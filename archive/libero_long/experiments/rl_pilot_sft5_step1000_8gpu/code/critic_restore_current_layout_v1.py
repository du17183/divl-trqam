"""Explicit consumer sharding; checkpoint bytes and all numerical values unchanged."""
from pilot8_common_v1 import *

def verify_repair():
    verify_worker()
    for name in ['critic_4plus4_sources003.json','critic_4plus4_sources004.json','postcritic_restore_sources005.json']:
        proof=read(EXP/'provenance'/name)
        for rel,h in proof['files'].items():assert sha(ROOT/rel)==h,rel
    from batch128_common_v1 import protected
    protected()

def restore_tree(path,template):
    import jax
    import numpy as np
    import orbax.checkpoint as ocp
    def arguments(x):
        if isinstance(x,jax.Array):
            assert isinstance(x.sharding,jax.sharding.Sharding)
            return ocp.ArrayRestoreArgs(restore_type=jax.Array,dtype=x.dtype,sharding=x.sharding,global_shape=x.shape,strict=True)
        if isinstance(x,np.ndarray):return ocp.RestoreArgs(restore_type=np.ndarray,dtype=x.dtype)
        return ocp.RestoreArgs()
    args=jax.tree.map(arguments,template)
    with ocp.PyTreeCheckpointer() as loader:
        loaded=loader.restore(str(Path(path).resolve()),item=template,restore_args=args)
    assert jax.tree.structure(loaded)==jax.tree.structure(template)
    for value,reference in zip(jax.tree.leaves(loaded),jax.tree.leaves(template),strict=True):
        assert value.shape==reference.shape and value.dtype==reference.dtype
        if isinstance(reference,jax.Array):assert value.sharding==reference.sharding
    return loaded

def restore(bundle,path,*,expected_metadata=None):
    import flax.nnx as nnx
    import jax
    from critic_state_v1 import payload
    verify_repair();path=Path(path)
    assert path.resolve().is_relative_to((EXP/'checkpoints/critic').resolve())
    manifest=read(path/'manifest.json')
    assert manifest['complete'] and manifest['next_sampler_update']==manifest['update']+1
    for rel,rec in manifest['files'].items():assert sha(path/rel)==rec['sha256'],rel
    for key,value in (expected_metadata or {}).items():assert manifest[key]==value,key
    state=nnx.state(bundle)
    loaded=restore_tree(path/'full_train_state',payload(bundle,manifest['update']))
    assert int(loaded['update'])==manifest['update']
    state.replace_by_pure_dict(loaded['state']);nnx.update(bundle,state)
    jax.block_until_ready(nnx.state(bundle))
    return manifest
