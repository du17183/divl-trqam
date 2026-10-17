"""Master-parameter precision shared by actor, beta and reference initialization."""
import flax.nnx as nnx
import jax
import jax.numpy as jnp


def ensure_fp32_master(model):
    params = nnx.state(model, nnx.Param)
    nnx.update(model, jax.tree.map(lambda x: x.astype(jnp.float32), params))
    assert_fp32_master(model)
    return model


def assert_fp32_master(model):
    leaves = jax.tree.leaves(nnx.state(model, nnx.Param))
    if not leaves or any(x.dtype != jnp.float32 for x in leaves):
        raise ValueError("All optimizer/EMA master parameters must be FP32")


def reference_ema(reference, beta, weight=0.005):
    assert_fp32_master(reference)
    assert_fp32_master(beta)
    if not 0 < weight < 1:
        raise ValueError("EMA new-parameter weight must be in (0,1)")
    nnx.update(reference, jax.tree.map(
        lambda old, new: (1 - weight) * old + weight * new,
        nnx.state(reference, nnx.Param), nnx.state(beta, nnx.Param)))
