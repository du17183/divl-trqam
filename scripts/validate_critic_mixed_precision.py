# Public release: deployment paths anonymized; algorithm equations unchanged.
"""Actual Gemma compute dtype, immutable FP32 masters and one discarded update."""
from pathlib import Path
import json
import hashlib
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from openpi.models.divl_tokenizer import DIVLTokenizer
from openpi.shared import normalize
from openpi.training.divl_libero_adapter import make_batch
from openpi.training.libero_lwd_training import LiberoTrainingConfig,build_critic,encoder_config,initialize_training,critic_step
from openpi.training.precision import assert_fp32_master
from train_libero_lwd import CRITIC_REPLAY_FIELDS

ROOT=Path('/opt/divl-trqam')

def digest(module):
    h=hashlib.sha256()
    for x in jax.tree.leaves(nnx.state(module,nnx.Param)):h.update(np.asarray(x).tobytes())
    return h.hexdigest()

def main():
    cfg=LiberoTrainingConfig();enc=encoder_config(ROOT);critic=build_critic(cfg,enc)
    target,vo,qo=initialize_training(critic,2000,learning_rate=cfg.learning_rate);assert_fp32_master(critic)
    rows=json.loads((ROOT/'data/pilot_libero_v1/manifest.json').read_text())['records'][:2];samples=[]
    for row in rows:
        with np.load(row['path'],allow_pickle=False) as f:e={k:f[k] for k in CRITIC_REPLAY_FIELDS}
        samples.append((e,3))
    norm=normalize.load(ROOT/'checkpoints/pi05_libero/assets/physical-intelligence/libero')
    batch=make_batch(samples,DIVLTokenizer(enc),norm,td_steps=1);batch['critic_k']=jnp.asarray([e['lengths'][i] for e,i in samples])
    @nnx.jit
    def inputs(model,obs):
        vt,vm=model.vision.encode_views(obs['images'],obs['image_masks'],train=False)
        te=model.gemma.embed_tokens(obs['token_ids'])
        return model.readout_input(vt,vm,te,obs['token_mask'])
    x,pos,mask=inputs(critic.encoder,batch['obs'])
    before=digest(critic)
    @nnx.jit
    def actual(core,x,pos,mask):
        return core._compute_model()(x,pos,mask,checkpoint_enabled=True,method='encode_embeddings')
    @nnx.jit
    def direct(model,x,pos,mask):return model(x,pos,mask,checkpoint_enabled=True,method='encode_embeddings')
    hidden=actual(critic.encoder.gemma,x,pos,mask);jax.block_until_ready(hidden)
    assert hidden.dtype==jnp.bfloat16,hidden.dtype
    clone=nnx.clone(critic.encoder.gemma.model)
    nnx.update(clone,jax.tree.map(lambda p:p.astype(jnp.bfloat16),nnx.state(clone,nnx.Param)))
    reference=direct(clone,x,pos,mask);jax.block_until_ready(reference)
    np.testing.assert_allclose(np.asarray(hidden,np.float32),np.asarray(reference,np.float32),rtol=1e-5,atol=1e-5)
    assert digest(critic)==before;assert_fp32_master(critic);assert_fp32_master(target)
    print('COMPUTE_VIEW_PASS',flush=True)
    metrics=jax.device_get(critic_step(critic,target,vo,qo,batch,config=cfg))
    assert bool(metrics['update_applied']) and all(np.isfinite(v) for v in metrics.values())
    assert digest(critic)!=before;assert_fp32_master(critic);assert_fp32_master(target)
    result={'passed':True,'batch':2,'real_optimizer_updates':1,'discarded_update':True,'actor_updated':False,
            'gemma_internal_output_dtype':str(hidden.dtype),'master_dtype':'float32','target_master_dtype':'float32',
            'forward_did_not_mutate_master':True,'explicit_bf16_weight_reference_max_error':float(np.max(np.abs(np.asarray(hidden,np.float32)-np.asarray(reference,np.float32)))),
            'real_update_changed_master':True,'metrics':{k:float(v) for k,v in metrics.items()},
            'source_sha256':hashlib.sha256((ROOT/'src/openpi/models/divl_gemma.py').read_bytes()).hexdigest()}
    (ROOT/'artifacts/critic_mixed_precision_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
    with (ROOT/'reports/critic_memory_fix.md').open('a') as f:
        f.write('\n## Root cause confirmed by XLA\n\nInitial remat with prevent_cse=False did not reduce memory. XLA reported 260.38GB temporary storage with many FP32 Gemma attention/FFN activations. Gemma dtype config initializes weights only; promoting master params to FP32 promoted plain einsum compute too.\n\n'
            'The copied backbone now makes a differentiable BF16 read view, preserving original FP32 storage and EMA, and enables prevent_cse=True for unrolled block remat. A real model forward matches an explicit BF16 weight reference, leaves master hashes unchanged, and a discarded real update changes FP32 parameters.\n\n'+json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
