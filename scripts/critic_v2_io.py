"""Use the validated full restore for checkpoints with shared NNX variable links."""
import flax.nnx as nnx
import jax,jax.numpy as jnp
import orbax.checkpoint as ocp

def restore_critic(critic,folder):
    with ocp.PyTreeCheckpointer() as c:
        restored=c.restore(folder.resolve())
    params=nnx.state(critic,nnx.Param)
    params.replace_by_pure_dict(restored['critic'])
    # Portable checkpoints contain NumPy FP32 arrays; materialize them on the
    # current device once, retaining their dtype and all global values.
    nnx.update(critic,jax.tree.map(jax.device_put,params));critic.eval();return critic
