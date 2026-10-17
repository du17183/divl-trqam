"""Shared native Scalar/DIVL state encoder and Double-Q; planned fixed H5 actions."""
from dataclasses import dataclass
from pathlib import Path
import flax.nnx as nnx
from flax.traverse_util import flatten_dict
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from openpi.models.divl_config import DIVLStateEncoderConfig
from openpi.models.divl_heads import MLP, DistributionalVHead, TwinQHead
from openpi.models.divl_state_encoder import DIVLStateEncoder
from openpi.training.divl_weight_loaders import load_siglip_weights, resolve_checkpoint_dirs
from openpi.training.divl_lwd_training import initialize_training
from openpi.training.libero_lwd_training import critic_step

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CriticConfig:
    value_type: str
    seed: int = 20261009
    expectile: float = .6
    tau_base: float = .6
    alpha: float = .3
    ema_rate: float = .005
    max_grad_norm: float = 1.
    learning_rate: float = .0002
    horizon: int = 5
    action_dim: int = 7
    updates: int = 2000
    gamma: float = .99

    def __post_init__(self):
        assert self.value_type in ['scalar', 'divl']
        assert (self.horizon, self.action_dim, self.updates, self.gamma) == (5, 7, 2000, .99)


class ScalarVHead(nnx.Module):
    def __init__(self, *, rngs):
        self.net = MLP(640, 1, rngs=rngs)

    def __call__(self, state):
        return self.net(state)[..., 0]


def encoder_config():
    return DIVLStateEncoderConfig(
        gemma_checkpoint='/home/zwl/openpi/.cache/gemma-data/checkpoints/gemma3-270m-it',
        siglip_checkpoint='/home/zwl/openpi/.cache/huggingface/hub/models--google--siglip-so400m-patch14-224/snapshots/d04cf29fca7b6374f74d8bea1969314492266b5e',
        gemma_tokenizer=str(ROOT / 'assets/critic_tokenizer/tokenizer_gemma3.model'),
        compute_dtype='float32', siglip_attention_dtype='float32')


def heads(cfg, encoder):
    # Independent RNG streams ensure the differing V output dimensions cannot shift Q initialization.
    q_head = TwinQHead(7, 5, rngs=nnx.Rngs(cfg.seed + 2))
    v_head = (ScalarVHead if cfg.value_type == 'scalar' else DistributionalVHead)(rngs=nnx.Rngs(cfg.seed + 1))
    return nnx.Dict(encoder=encoder, q_head=q_head, v_head=v_head)


def build_pretrained(cfg):
    """Public pretrained Gemma3/SigLIP only; never load any old task-trained Critic."""
    from gemma.gm.ckpts import _compat
    enc = encoder_config()
    model = heads(cfg, DIVLStateEncoder(enc, rngs=nnx.Rngs(cfg.seed)))
    gemma, siglip = resolve_checkpoint_dirs(enc)
    with ocp.StandardCheckpointer() as checkpointer:
        metadata = checkpointer.metadata(gemma)
        metadata = getattr(metadata, 'item_metadata', metadata)
        specs = jax.tree.map(lambda item: jax.ShapeDtypeStruct(item.shape, item.dtype,
                            sharding=jax.sharding.SingleDeviceSharding(jax.devices()[0])), metadata.tree)
        restored = checkpointer.restore(gemma, target=specs)
    params = _compat.nest_params(_compat.param_remapper(restored))['transformer']
    state = nnx.state(model.encoder.gemma.model, nnx.Param)
    actual, expected = flatten_dict(params), flatten_dict(state.to_pure_dict())
    assert actual.keys() == expected.keys() and all(actual[key].shape == expected[key].shape for key in actual)
    state.replace_by_pure_dict(params)
    nnx.update(model.encoder.gemma.model, state)
    load_siglip_weights(model.encoder.vision, siglip)
    nnx.update(model, jax.tree.map(lambda item: item.astype(jnp.float32), nnx.state(model, nnx.Param)))
    return model


def action_gradient(model, batch):
    """Gradient with respect to exactly the planned normalized H5 real7 input."""
    state = jax.lax.stop_gradient(model.encoder(**batch['obs'], train=False))
    def value(actions):
        q1, q2 = model.q_head(state, actions, batch['action_mask'])
        return jnp.minimum(q1, q2).sum()
    return jax.grad(value)(batch['actions'])
