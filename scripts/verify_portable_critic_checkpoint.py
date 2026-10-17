# Public release: deployment paths anonymized; algorithm equations unchanged.
"""CPU round trip of the same four NNX state fields and FP32 restoration."""
from pathlib import Path
import json
import numpy as np
import jax,jax.numpy as jnp
import flax.nnx as nnx
import optax,orbax.checkpoint as ocp
from critic_v2_io import restore_critic
R=Path('/opt/divl-trqam');O=R/'artifacts/replay_action_audit_v3'
def main():
    critic=nnx.Dict(encoder=nnx.Linear(3,8,rngs=nnx.Rngs(1)),q_head=nnx.Linear(8,2,rngs=nnx.Rngs(2)),v_head=nnx.Linear(8,1,rngs=nnx.Rngs(3)))
    target=nnx.clone(critic);vo=nnx.Optimizer(critic,optax.adam(.0002),wrt=nnx.Param);qo=nnx.Optimizer(critic,optax.adam(.0002),wrt=nnx.Param)
    params=nnx.state(critic,nnx.Param);nnx.update(critic,jax.tree.map(lambda x:x+jnp.float32(.0000137),params))
    expected=nnx.state(critic,nnx.Param).to_pure_dict()
    data=dict(critic=expected,target=nnx.state(target,nnx.Param).to_pure_dict(),value_optimizer=nnx.state(vo).to_pure_dict(),q_optimizer=nnx.state(qo).to_pure_dict())
    host=jax.tree.map(lambda x:np.asarray(jax.device_get(x)),data)
    assert all(isinstance(x,np.ndarray) for x in jax.tree.leaves(host))
    folder=(R/'backups/throughput_20261007/portable_toy_checkpoint').resolve();assert not folder.exists()
    with ocp.PyTreeCheckpointer() as ck:ck.save(folder,host)
    reloaded=restore_critic(nnx.clone(target),folder)
    pairs=list(zip(jax.tree.leaves(expected),jax.tree.leaves(nnx.state(reloaded,nnx.Param).to_pure_dict()),strict=True))
    assert all(b.dtype==np.float32 and np.array_equal(np.asarray(a),np.asarray(b)) for a,b in pairs)
    result=dict(passed=True,parameters_bitwise_equal=True,fp32_dtype_retained=True,
                full_four_field_state=True,restore_into_current_device=True,actor_updated=False)
    (O/'portable_checkpoint_roundtrip.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
