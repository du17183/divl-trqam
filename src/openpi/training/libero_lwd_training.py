"""One LIBERO trainer: independent value and reference choices, frozen by default."""
from dataclasses import dataclass
from functools import partial
from pathlib import Path
import flax.nnx as nnx
from flax.traverse_util import flatten_dict
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from openpi.models.divl_config import DIVLStateEncoderConfig
from openpi.models.divl_heads import MLP, DistributionalVHead, TwinQHead
from openpi.models.divl_state_encoder import DIVLStateEncoder
from openpi.models.divl_losses import distributional_loss, divl_q_target, twin_q_loss, adaptive_quantile_level
from openpi.training.divl_lwd_training import initialize_training, _prepare_gradients, _guarded_update
from openpi.training.divl_weight_loaders import load_siglip_weights, resolve_checkpoint_dirs


@dataclass(frozen=True)
class LiberoTrainingConfig:
    value_type: str = 'scalar'
    reference_type: str = 'fixed'
    seed: int = 20261007
    gamma: float = .9999
    expectile: float = .6
    tau_base: float = .6
    alpha: float = .3
    ema_rate: float = .005
    max_grad_norm: float = 1.0
    learning_rate: float = .0002
    horizon: int = 10
    replan_steps: int = 5
    action_dim: int = 7
    actor_updates: bool = False
    smoke_steps: int = 0

    def __post_init__(self):
        if self.value_type not in ('scalar','divl') or self.reference_type not in ('fixed','dynamic'):
            raise ValueError('value_type/reference_type are the only algorithm switches')
        if (self.horizon,self.replan_steps,self.action_dim,self.gamma)!=(10,5,7,.9999):
            raise ValueError('Frozen LIBERO interface changed')
        if not 0<self.expectile<1 or not 0<=self.smoke_steps<=5:
            raise ValueError('Invalid expectile or disposable smoke budget')
        if self.actor_updates and not self.smoke_steps:
            raise ValueError('Formal actor updates are gated off in this stage')


class ScalarVHead(nnx.Module):
    def __init__(self,*,rngs):self.net=MLP(640,1,rngs=rngs)
    def __call__(self,z):return self.net(z)[...,0]


def build_critic(cfg, encoder_config):
    """Same pretrained encoder/Double-Q architecture and seed in both arms."""
    from gemma.gm.ckpts import _compat
    encoder=DIVLStateEncoder(encoder_config,rngs=nnx.Rngs(cfg.seed))
    q_head=TwinQHead(7,10,rngs=nnx.Rngs(cfg.seed+2))
    v_head=(ScalarVHead if cfg.value_type=='scalar' else DistributionalVHead)(rngs=nnx.Rngs(cfg.seed+1))
    critic=nnx.Dict(encoder=encoder,q_head=q_head,v_head=v_head)
    gemma,siglip=resolve_checkpoint_dirs(encoder_config)
    with ocp.StandardCheckpointer() as ck:
        metadata=ck.metadata(gemma);metadata=getattr(metadata,'item_metadata',metadata)
        spec=jax.tree.map(lambda x:jax.ShapeDtypeStruct(x.shape,x.dtype,sharding=jax.sharding.SingleDeviceSharding(jax.devices()[0])),metadata.tree)
        restored=ck.restore(gemma,target=spec)
    params=_compat.nest_params(_compat.param_remapper(restored))['transformer']
    state=nnx.state(critic.encoder.gemma.model,nnx.Param)
    actual,expected=flatten_dict(params),flatten_dict(state.to_pure_dict())
    if actual.keys()!=expected.keys() or any(actual[k].shape!=expected[k].shape for k in actual):
        raise ValueError('Pretrained Gemma parameter shape mismatch')
    state.replace_by_pure_dict(params);nnx.update(critic.encoder.gemma.model,state)
    load_siglip_weights(critic.encoder.vision,siglip)
    return critic


def encoder_config(root):
    # Public pretrained files are read-only. No source user's environment/code writes.
    return DIVLStateEncoderConfig(
        gemma_checkpoint='/home/zwl/openpi/.cache/gemma-data/checkpoints/gemma3-270m-it',
        siglip_checkpoint='/home/zwl/openpi/.cache/huggingface/hub/models--google--siglip-so400m-patch14-224/snapshots/d04cf29fca7b6374f74d8bea1969314492266b5e',
        gemma_tokenizer=str(Path(root)/'assets/critic_tokenizer/tokenizer_gemma3.model'),
        compute_dtype='bfloat16')


def expectile_loss(values,target,expectile):
    residual=jax.lax.stop_gradient(target)-values
    weight=jnp.where(residual>=0,expectile,1-expectile)
    return jnp.mean(weight*residual**2)


@partial(nnx.jit,static_argnames=('config',))
def critic_step(critic,target,value_opt,q_opt,batch,*,config):
    critic.train();target.eval()
    bundle=nnx.Dict(critic=critic,value_optimizer=value_opt,q_optimizer=q_opt)
    before=nnx.state(bundle)
    z=target.encoder(**batch['obs'],train=False)
    qtarget=jax.lax.stop_gradient(jnp.minimum(*target.q_head(z,batch['actions'],batch['action_mask'])))
    def vloss(model):
        encoded=model.encoder(**batch['obs'],train=True)
        values=model.v_head(encoded)
        if config.value_type=='scalar':return expectile_loss(values,qtarget,config.expectile)
        return distributional_loss(values,qtarget,model.v_head.atoms)
    lv,gv=nnx.value_and_grad(vloss,argnums=nnx.DiffState(0,value_opt.wrt))(critic)
    gv,nv,okv=_prepare_gradients(gv,lv,config.max_grad_norm)
    _guarded_update(value_opt,gv,okv)
    next_z=target.encoder(**batch['next_obs'],train=False)
    next_value=target.v_head(next_z)
    if config.value_type=='scalar':
        td=jax.lax.stop_gradient(batch['rewards']+batch['discounts']*next_value)
        entropy=jnp.asarray(0.0);tau=jnp.asarray(config.expectile)
    else:
        td=divl_q_target(batch['rewards'],batch['discounts'],next_value,target.v_head.atoms,
                        tau_base=config.tau_base,alpha=config.alpha,tau_min=0,tau_max=1)
        probs=jax.nn.softmax(next_value.astype(jnp.float32),axis=-1)
        entropy=jnp.mean(-jnp.sum(probs*jnp.log(jnp.maximum(probs,1e-10)),axis=-1))
        tau=jnp.mean(adaptive_quantile_level(next_value,tau_base=config.tau_base,alpha=config.alpha,tau_min=0,tau_max=1))
    def qloss(model):
        encoded=model.encoder(**batch['obs'],train=True)
        q1,q2=model.q_head(encoded,batch['actions'],batch['action_mask'])
        return twin_q_loss(q1,q2,td),{'q1_mean':q1.mean(),'q2_mean':q2.mean(),'td_error':jnp.mean(jnp.abs(jnp.minimum(q1,q2)-td))}
    (lq,metrics),gq=nnx.value_and_grad(qloss,argnums=nnx.DiffState(0,q_opt.wrt),has_aux=True)(critic)
    gq,nq,okq=_prepare_gradients(gq,lq,config.max_grad_norm)
    _guarded_update(q_opt,gq,okv&okq)
    candidate=nnx.state(bundle)
    finite=jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(candidate)]))
    applied=okv&okq&finite
    nnx.update(bundle,jax.lax.cond(applied,lambda:candidate,lambda:before))
    nnx.update(target,jax.tree.map(lambda old,new:jnp.where(applied,(1-config.ema_rate)*old+config.ema_rate*new,old),nnx.state(target,nnx.Param),nnx.state(critic,nnx.Param)))
    return {**metrics,'loss_v':lv,'loss_q':lq,'value_grad_norm':nv,'q_grad_norm':nq,'update_applied':applied,
            'value_entropy':entropy,'tau':tau,'critic_k_mean':batch['critic_k'].mean(),'discount_mean':batch['discounts'].mean()}


def training_step(critic,target,value_opt,q_opt,batch,*,config,actor=None,reference=None,actor_opt=None,
                  beta=None,beta_opt=None,action_stats=None,rng=None,qam_config=None):
    metrics=critic_step(critic,target,value_opt,q_opt,batch.critic,config=config)
    if not config.actor_updates:
        if beta is not None or actor_opt is not None:
            raise ValueError('Actor/beta optimizer passed to a frozen-actor phase')
        return rng,metrics
    from openpi.training.qam_training import actor_train_step
    from openpi.training import beta_training
    if not bool(metrics['update_applied']):raise FloatingPointError('Critic transaction rejected')
    rng,actor_metrics=actor_train_step(actor,reference,critic,actor_opt,batch.actor_observation,
        batch.critic['obs'],action_stats,rng,config=qam_config)
    if config.reference_type=='dynamic':
        if beta is None or beta_opt is None:raise ValueError('Dynamic reference requires an independent BC beta')
        rng,beta_rng=jax.random.split(rng)
        metrics.update(beta_training.beta_step(beta,reference,beta_opt,batch.actor_observation,
            beta_training.bc_targets(batch.critic,actor.action_dim),beta_rng,action_mask=batch.critic['actor_bc_mask']))
    elif beta is not None:raise ValueError('Fixed reference arm cannot update beta')
    metrics.update(actor_metrics)
    return rng,metrics
