# Public release: deployment paths anonymized; algorithm equations unchanged.
"""Verify host prefetch tensors and unchanged RNG/sample order, CPU only."""
from pathlib import Path
import json,hashlib,time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import jax
from openpi.shared import normalize
from openpi.models.divl_tokenizer import DIVLTokenizer
from openpi.training.libero_lwd_training import encoder_config
from openpi.training.divl_libero_adapter import make_batch
from train_libero_lwd import CRITIC_REPLAY_FIELDS

R=Path('/opt/divl-trqam');O=R/'artifacts/replay_action_audit_v3'
def main():
    original=json.loads((R/'data/pilot_libero_v1/manifest.json').read_text())['records']
    rows=original+json.loads((R/'data/action_sensitive_replay_v2/manifest.json').read_text())['records']
    def length(row):
        with np.load(row['path'],allow_pickle=False) as f:return len(f['rewards'])
    with ThreadPoolExecutor(max_workers=8) as pool:lengths=list(pool.map(length,rows))
    orig=[];anchor=[];continuation=[]
    for e,n in enumerate(lengths):
        if e<len(original):orig.extend((e,i) for i in range(n))
        else:anchor.append((e,0));continuation.extend((e,i) for i in range(1,n))
    orig=np.asarray(orig);anchor=np.asarray(anchor);continuation=np.asarray(continuation)
    rng=np.random.default_rng(20261007);sha=hashlib.sha256();checks={};first=None
    for step in range(1,101):
        index=np.concatenate([anchor[rng.integers(len(anchor),size=128)],orig[rng.integers(len(orig),size=64)],continuation[rng.integers(len(continuation),size=64)]])
        index=index[rng.permutation(256)]
        if first is None:first=index
        sha.update(json.dumps([(rows[e]['episode_id'],int(i)) for e,i in index],separators=(',',':')).encode())
        if step in [25,50,100]:checks[str(step)]=sha.hexdigest()
    assert checks['25']=='fc229dca5d6ffb16ed2058501d8e264ef6803fb35b084453f2bd170e4cd90bd1'
    assert checks['50']=='476e8780e044ff8b1ef6445921616d9f42666313d0090f7b8dbfbec2001e1912'
    cache={};samples=[]
    for e,i in first:
        if e not in cache:
            with np.load(rows[e]['path'],allow_pickle=False) as f:cache[e]={k:f[k] for k in CRITIC_REPLAY_FIELDS}
        samples.append((cache[e],int(i)))
    tokenizer=DIVLTokenizer(encoder_config(R));norm=normalize.load(R/'checkpoints/pi05_libero/assets/physical-intelligence/libero')
    tick=time.monotonic();host=make_batch(samples,tokenizer,norm,td_steps=10,return_numpy=True);seconds=time.monotonic()-tick
    old=jax.device_get(make_batch(samples,tokenizer,norm,td_steps=10))
    comparisons=[]
    for a,b in zip(jax.tree.leaves(host),jax.tree.leaves(old),strict=True):
        assert a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a,b)
        comparisons.append(dict(shape=list(a.shape),dtype=str(a.dtype),equal=True))
    result=dict(passed=True,full_global_batch=256,host_vs_previous_device_arrays=comparisons,
                original_sample_order_matches=True,sample_hashes=checks,host_prepare_seconds=seconds,
                frozen_global_batch=True,actor_updated=False)
    (O/'prefetch_batch_contract.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
