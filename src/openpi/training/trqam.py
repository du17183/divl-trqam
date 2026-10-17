# Adapted from yonghdong/trqam, commit 8775edfd019fe02d7468153af09700f633a61ab0.
# TRQAM-derived portions: MIT; see licenses/TRQAM-MIT.txt.
# Published from the audited LIBERO experiment; mathematical body unchanged.
"""Full-parameter FP32 native FM/QAM and an audited official-TRQAM port.

Critics stay fixed during this critic-pretrained exploratory Actor pilot.
TRQAM solver/loss/dual formulas follow yonghdong/trqam commit8775edf;
VLA H10, min DoubleQ and IQL pretrained critics are explicit adaptations.
"""
import functools
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax
from openpi.models import model as model_lib
from openpi.training import qam
from openpi.training.qam_cached_training import CachedVelocity,prefix_cache,sample_with_velocities,cached_actor_loss
from openpi.training.divl_lwd_training import _prepare_gradients,_guarded_update


def guarded_ema(reference,beta,weight,valid):
    nnx.update(reference,jax.tree.map(lambda old,new:jnp.where(valid,(1-weight)*old+weight*new,old),
        nnx.state(reference,nnx.Param),nnx.state(beta,nnx.Param)))


def tr_path(vfast,vslow,qgrad,rng,shape,steps=10):
    """Same shifted grid/noise/random-key allocation as official adj_matching."""
    h=1/steps;x=jax.random.normal(rng,shape,dtype=jnp.float32);xs=[x];ts=[]
    for i,key in zip(range(steps),jax.random.split(rng,steps),strict=True):
        t=jnp.asarray(i/steps,jnp.float32);g=jnp.sqrt(2*(1-t+h)/(t+h))
        if i!=steps-1:x=x+h*(2*vfast(x,t)-x/(t+h))+jnp.sqrt(h)*g*jax.random.normal(key,shape,dtype=jnp.float32)
        else:x=x+h*vslow(x,t)
        x=jax.lax.stop_gradient(x);xs.append(x);ts.append(t)
    terminal=jax.lax.stop_gradient(qgrad(xs[-1]));adj=terminal;adjs=[]
    for i in reversed(range(steps)):
        t=jnp.asarray(i/steps,jnp.float32)
        _,pullback=jax.vjp(lambda xi:2*vslow(xi,t+h)-xi/(t+h),xs[i])
        adj=jax.lax.stop_gradient(adj+h*pullback(adj)[0]);adjs.append(adj)
    return jnp.stack(xs[:-1]),jnp.stack(list(reversed(adjs))),jnp.stack(ts),xs[-1],terminal


def tr_local_loss(vfast,vslow,xs,adjs,ts,lam,steps=10,horizon=10):
    h=1/steps
    def each(_,items):
        x,a,t=items;vf=vfast(x,t);vb=jax.lax.stop_gradient(vslow(x,t));g2=2*(1-t+h)/(t+h)
        sigma=jnp.sqrt(g2/jnp.maximum(lam,1e-8));delta=vf-vb
        # Sum all H*D coordinates, mean states, sum solver steps: official loss.
        loss=jnp.mean(jnp.sum((2*delta/sigma+sigma*a)**2,axis=(-2,-1)))
        kl=jnp.mean(2*h/g2*jnp.sum(delta**2,axis=(-2,-1)))/horizon
        return None,(loss,kl)
    _,(loss,kl)=jax.lax.scan(jax.checkpoint(each),None,(xs,adjs,ts))
    return jnp.sum(loss),jnp.sum(kl)


def tr_dual(lam,ema,path_kl,*,budget=1.,scale=3.,rho=.1,eta=.01,minimum=.01,maximum=100.,clip_coefficient=2.):
    clipped=jnp.minimum(path_kl,clip_coefficient*budget)
    newema=(1-rho)*ema+rho*clipped
    newlam=jnp.clip(lam+eta*(newema-budget),minimum,maximum)
    return newlam,newema,{'path_kl':path_kl,'path_kl_ema':newema,'path_kl_clipped':clipped,
        'lambda_parameter':newlam,'lambda_effective':scale*newlam,'kl_budget':jnp.asarray(budget),
        'kl_constraint_violation':path_kl-budget,'kl_over_budget':path_kl>budget}


@nnx.jit
def bc_step(actor,reference,opt,obs,actions,rng,ema_weight):
    actor.train();reference.eval()
    loss,grads=nnx.value_and_grad(lambda m:jnp.mean(m.compute_loss(rng,obs,actions,train=True)),argnums=nnx.DiffState(0,opt.wrt))(actor)
    ready,norm,valid=_prepare_gradients(grads,loss,1.)
    _guarded_update(opt,ready,valid)
    # Only blend after a valid full optimizer transaction.
    guarded_ema(reference,actor,ema_weight,valid)
    actor.eval()
    return {'fm_loss':loss,'grad_norm':norm,'update_applied':valid}


@functools.partial(nnx.jit,static_argnames=('config','explicit_bc'))
def qam_step(actor,reference,critic,opt,obs,cobs,stats,rng,bcobs,bcactions,bcrng,*,config,explicit_bc=False):
    actor.eval();reference.eval();critic.eval()
    obs=model_lib.preprocess_observation(None,obs,train=False)
    z=jax.lax.stop_gradient(critic.encoder(**cobs,train=False))
    ref=CachedVelocity(reference,jax.tree.map(jax.lax.stop_gradient,prefix_cache(reference,obs)))
    # C/D/E/F fixed SFT reference sampler and endpoint; no beta update here.
    times,path,velocities=sample_with_velocities(ref,obs,rng,config=config)
    adj,terminal=qam.qam_terminal_adjoint(critic,z,path[-1],stats.actor_q01,stats.actor_q99,stats.critic_q01,stats.critic_q99,
        temperature=config.temperature,replan_steps=5,critic_horizon=5,action_dim=7,action_clip=None)
    adjs=qam.qam_backward_adjoint(ref,obs,times,path,adj)
    def loss_fn(m):
        loss,metrics=cached_actor_loss(m,obs,times,path,adjs,velocities)
        fm=jnp.mean(m.compute_loss(bcrng,bcobs,bcactions,train=True)) if explicit_bc else jnp.asarray(0.,jnp.float32)
        return loss+fm,{**metrics,'fm_loss':fm,'total_loss':loss+fm}
    (loss,metrics),grads=nnx.value_and_grad(loss_fn,argnums=nnx.DiffState(0,opt.wrt),has_aux=True)(actor)
    ready,norm,valid=_prepare_gradients(grads,loss,1.);_guarded_update(opt,ready,valid)
    return {**metrics,**terminal,'grad_norm':norm,'update_applied':valid}


@nnx.jit
def tr_step(actor,beta,reference,critic,opt,betaopt,dual,obs,cobs,stats,rng,bcobs,bcactions,bcrng):
    actor.eval();beta.eval();reference.eval();critic.eval()
    obs=model_lib.preprocess_observation(None,obs,train=False)
    fast=CachedVelocity(actor,jax.tree.map(jax.lax.stop_gradient,prefix_cache(actor,obs)))
    slow=CachedVelocity(reference,jax.tree.map(jax.lax.stop_gradient,prefix_cache(reference,obs)))
    z=jax.lax.stop_gradient(critic.encoder(**cobs,train=False))
    vf=lambda x,t:qam.qam_velocity(fast,obs,x,t)
    vb=lambda x,t:qam.qam_velocity(slow,obs,x,t)
    def terminal(x):
        return qam.qam_terminal_adjoint(critic,z,x,stats.actor_q01,stats.actor_q99,stats.critic_q01,stats.critic_q99,
            temperature=1.,replan_steps=5,critic_horizon=5,action_dim=7,action_clip=None)[0]
    shape=(obs.state.shape[0],10,32)
    xs,adjs,ts,endpoint,lastadj=tr_path(vf,vb,terminal,rng,shape)
    effective=3.*dual.lam.value
    def fast_loss(m):
        # Trainable prefix is recreated INSIDE differentiation; no frozen VLA encoder.
        cached=CachedVelocity(m,prefix_cache(m,obs))
        loss,kl=tr_local_loss(lambda x,t:qam.qam_velocity(cached,obs,x,t),vb,xs,adjs,ts,effective)
        return loss,kl
    (loss,kl),grad=nnx.value_and_grad(fast_loss,argnums=nnx.DiffState(0,opt.wrt),has_aux=True)(actor)
    beta.train()
    fm,bcgrad=nnx.value_and_grad(lambda m:jnp.mean(m.compute_loss(bcrng,bcobs,bcactions,train=True)),argnums=nnx.DiffState(0,betaopt.wrt))(beta)
    g,n,valid=_prepare_gradients(grad,loss,1.);bg,bn,bvalid=_prepare_gradients(bcgrad,fm,1.);valid=valid & bvalid
    _guarded_update(opt,g,valid);_guarded_update(betaopt,bg,valid)
    # Both policy updates use the PREVIOUS slow reference. EMA follows both updates.
    guarded_ema(reference,beta,.005,valid);beta.eval()
    lam,ema,dm=tr_dual(dual.lam.value,dual.kl_ema.value,kl)
    dual.lam.value=jnp.where(valid,lam,dual.lam.value);dual.kl_ema.value=jnp.where(valid,ema,dual.kl_ema.value)
    return {'trqam_loss':loss,'fm_loss':fm,'grad_norm':n,'beta_grad_norm':bn,'update_applied':valid,
        'terminal_adjoint_norm':jnp.mean(jnp.linalg.norm(lastadj.reshape(shape[0],-1),axis=-1)),
        'path_max_abs':jnp.max(jnp.abs(xs)),**dm}
