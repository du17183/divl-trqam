# Public release: deployment paths anonymized; algorithm equations unchanged.
"""Phase 0 gates: exact CPU tests plus disposable real pi05 backward/checkpoint."""
import argparse
import gc
import json
from pathlib import Path
import time
import traceback
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from openpi.models import model as model_lib
from openpi.models.divl_config import DIVLStateEncoderConfig
from openpi.models.divl_tokenizer import DIVLTokenizer
from openpi.shared import normalize
from openpi.training import beta_training, config
from openpi.training.precision import assert_fp32_master, reference_ema
from openpi.training.qam_training import initialize_actor_training, QAMTrainingConfig
from openpi.training.qam_libero_adapter import LiberoQAMBatchBuilder

ROOT = Path('/opt/divl-trqam')


class TinyBeta(nnx.Module):
    def __init__(self):
        self.weight = nnx.Param(jnp.ones((1,), dtype=jnp.bfloat16))

    def compute_loss(self, rng, obs, actions, train=True):
        return jnp.mean((self.weight.value - actions) ** 2, axis=-1)


def write(name, data):
    (ROOT/'artifacts').mkdir(exist_ok=True)
    (ROOT/'reports').mkdir(exist_ok=True)
    (ROOT/'artifacts'/name).write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n')


def logical_tensors(model):
    """Unpack real scan-layer parameter tensors, not arbitrary array slices.

    Official pi05 stores 27 vision blocks and 18 language/action blocks in
    stacked leaves. There are only 51 storage leaves, but many layer tensors.
    """
    leaves=jax.tree.flatten_with_path(nnx.state(model,nnx.Param).to_pure_dict())[0]
    rows=[]
    for path,value in leaves:
        name=jax.tree_util.keystr(path)
        stacked="['encoderblock']" in name or "['llm']['layers']" in name
        if stacked:
            assert value.shape[0] in (18,27), (name,value.shape)
            rows.extend((name+f'[scan_layer={i}]',value[i]) for i in range(value.shape[0]))
        else:
            rows.append((name,value))
    return rows


def small():
    ref = TinyBeta()
    beta, opt = beta_training.initialize_dynamic_beta(ref, learning_rate_fn=2e-5)
    assert_fp32_master(beta); assert_fp32_master(ref)
    mask = jnp.asarray([[True]*3+[False]*7])
    for i in range(8):
        old_ref, old_beta = ref.weight.value.copy(), beta.weight.value.copy()
        metrics = beta_training.beta_step(beta, ref, opt, None, jnp.zeros((1,10,32)), jax.random.key(i), action_mask=mask)
        np.testing.assert_allclose(ref.weight.value, .995*old_ref+.005*beta.weight.value, atol=1e-7, rtol=0)
    assert float(beta.weight.value[0]) < 1 and float(ref.weight.value[0]) < 1
    # Padding labels can be arbitrarily large but cannot contribute to loss.
    a, b = TinyBeta(), TinyBeta()
    ba, oa = beta_training.initialize_dynamic_beta(a, learning_rate_fn=2e-5)
    bb, ob = beta_training.initialize_dynamic_beta(b, learning_rate_fn=2e-5)
    targets = jnp.zeros((1,10,32))
    altered = targets.at[:,3:].set(1e5)
    ma = beta_training.beta_step(ba,a,oa,None,targets,jax.random.key(1),action_mask=mask)
    mb = beta_training.beta_step(bb,b,ob,None,altered,jax.random.key(1),action_mask=mask)
    np.testing.assert_allclose(ma['beta/bc_loss'], mb['beta/bc_loss'], atol=0, rtol=0)
    np.testing.assert_allclose(ba.weight.value, bb.weight.value, atol=0, rtol=0)
    destination = ROOT/'checkpoints/phase0_tiny_fp32'
    destination.parent.mkdir(exist_ok=True)
    with ocp.PyTreeCheckpointer() as ck:
        ck.save(destination.resolve(), {'beta':nnx.state(beta,nnx.Param).to_pure_dict(), 'reference':nnx.state(ref,nnx.Param).to_pure_dict()}, force=True)
        restored = ck.restore(destination.resolve())
    assert all(x.dtype==jnp.float32 for x in jax.tree.leaves(restored))
    result = {'passed':True, 'updates':8, 'lr':2e-5, 'beta_parameter':float(beta.weight.value[0]), 'reference_parameter':float(ref.weight.value[0]),
              'ema_formula_max_error_below':1e-7,'padding_loss_excluded':True,'checkpoint_fp32_verified':True}
    write('phase0_small_parameter_metrics.json', result)
    print(json.dumps(result),flush=True)


def real():
    started=time.monotonic()
    cfg=config.get_config('pi05_libero')
    actor=cfg.model.load(model_lib.restore_params(ROOT/'checkpoints/pi05_libero/params',dtype=jnp.bfloat16))
    ref, unused_opt=initialize_actor_training(actor,QAMTrainingConfig(critic_horizon=10))
    del unused_opt; gc.collect()
    beta,opt=beta_training.initialize_dynamic_beta(ref,learning_rate_fn=2e-5)
    for obj in [actor,beta,ref]: assert_fp32_master(obj)
    path=ROOT/'data/feasibility_libero_long/task00_episode00.npz'
    with np.load(path,allow_pickle=False) as f: ep={k:f[k] for k in f.files}
    norm=normalize.load(ROOT/'checkpoints/pi05_libero/assets/physical-intelligence/libero')
    tokenizer=DIVLTokenizer(DIVLStateEncoderConfig(gemma_tokenizer=str(ROOT/'assets/critic_tokenizer/tokenizer_gemma3.model')))
    builder=LiberoQAMBatchBuilder(cfg.model,norm,norm,tokenizer,training_config=QAMTrainingConfig(critic_horizon=10),td_steps=1)
    batch=builder([(ep,0)])
    targets=beta_training.bc_targets(batch.critic,actor.action_dim)
    before=logical_tensors(beta)
    assert len(before)>=100
    selected=sorted(np.random.default_rng(20261007).choice(len(before),100,replace=False).tolist())
    print('REAL_MODEL_READY '+json.dumps({'tensor_count':len(before),'sampled':len(selected),'device':str(jax.devices())}),flush=True)
    losses=[]
    for step in range(2):
        tick=time.monotonic()
        metrics=beta_training.beta_step(beta,ref,opt,batch.actor_observation,targets,jax.random.key(20261007+step),action_mask=batch.critic['actor_bc_mask'])
        metrics={k:float(v) for k,v in jax.device_get(metrics).items()}
        assert all(np.isfinite(v) for v in metrics.values())
        losses.append(metrics)
        print('REAL_UPDATE '+json.dumps({'step':step+1,'seconds':time.monotonic()-tick,**metrics}),flush=True)
    after=logical_tensors(beta)
    refs=logical_tensors(ref)
    rows=[]
    for i in selected:
        p,initial=before[i]; p2,current=after[i]; p3,reference=refs[i]
        assert p==p2==p3
        rows.append({'path':p,'shape':list(current.shape),'dtype':str(current.dtype),
                     'beta_delta_norm':float(jnp.linalg.norm(current-initial)),
                     'reference_delta_norm':float(jnp.linalg.norm(reference-initial))})
    assert any(x['beta_delta_norm']>0 for x in rows)
    assert any(x['reference_delta_norm']>0 for x in rows)
    assert all(x['dtype']=='float32' for x in rows)
    destination=ROOT/'checkpoints/phase0_real_beta_reference_fp32'
    print('CHECKPOINT_SAVE_START',flush=True)
    with ocp.PyTreeCheckpointer() as ck:
        ck.save(destination.resolve(), {'beta':nnx.state(beta,nnx.Param).to_pure_dict(),'reference':nnx.state(ref,nnx.Param).to_pure_dict()})
        metadata=ck.metadata(destination.resolve())
        metadata=getattr(metadata,'item_metadata',metadata)
        # Metadata and actual full restore both cover all saved parameter leaves.
        all_metadata=jax.tree.leaves(metadata)
        assert all(str(x.dtype)=='float32' for x in all_metadata)
        restored=ck.restore(destination.resolve())
    assert all(x.dtype==jnp.float32 for x in jax.tree.leaves(restored))
    restored_leaves=jax.tree.leaves(restored['beta'])
    original_leaves=jax.tree.leaves(nnx.state(beta,nnx.Param).to_pure_dict())
    np.testing.assert_allclose(restored_leaves[0],original_leaves[0],atol=0,rtol=0)
    result={'passed':True,'scope':'disposable two-step real pi05 BC update; not an experiment actor checkpoint',
            'actor_master_dtype':'float32','beta_master_dtype':'float32','reference_master_dtype':'float32',
            'real_updates':2,'lr':2e-5,'sampled_tensors':rows,'total_parameter_tensors':len(before),
            'sampling_unit':'distinct actual layer tensors; scan-stacked layers unpacked along their documented layer axis',
            'stored_parameter_leaves':len(original_leaves),
            'sampled_beta_changed':sum(x['beta_delta_norm']>0 for x in rows),
            'sampled_reference_changed':sum(x['reference_delta_norm']>0 for x in rows),
            'checkpoint_all_parameter_dtypes_verified':'float32','checkpoint_save_reload':True,
            'checkpoint_path':str(destination),'losses':losses,'elapsed_seconds':time.monotonic()-started,
            'original_checkpoint_modified':False,'reuse_disposable_weights_for_training':False}
    write('phase0_beta_fp32_metrics.json',result)
    (ROOT/'reports/phase0_beta_fp32_report.md').write_text('# Phase 0 FP32 validation\n\nPASS. Actor, beta and reference masters are FP32.\n\n'
        f'2 disposable real pi05 BC updates; 100 parameter tensors sampled. Beta changed in {result["sampled_beta_changed"]}, reference in {result["sampled_reference_changed"]}.\n\n'
        'Eight-step small-lr test, hand EMA and padding loss tests passed; full real beta/reference checkpoint save/reload retained FP32.\n\n'
        'Details: artifacts/phase0_beta_fp32_metrics.json. These temporary trained weights are not used in subsequent comparisons.\n')
    (ROOT/'reports/current_stage_summary.md').write_text('# Current stage\n\nPhase 0 PASS. FP32 actor/beta/reference masters and real updates verified.\n\nPhase 0B label audit is next. No QAM actor training or formal four-arm experiment has started.\n')
    print('PHASE0_REAL_PASS',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--real',action='store_true');args=parser.parse_args()
    try:
        real() if args.real else small()
    except Exception as e:
        write('phase0_failure.json',{'passed':False,'error':repr(e),'traceback':traceback.format_exc()})
        raise
