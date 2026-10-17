# Public release: deployment paths anonymized; algorithm equations unchanged.
"""Disposable real replay/global-update benchmark; never writes formal parameters."""
from pathlib import Path
import argparse, hashlib, json, time, subprocess
import numpy as np
import flax.nnx as nnx
import jax, jax.numpy as jnp
from openpi.shared import normalize
from openpi.models.divl_tokenizer import DIVLTokenizer
from openpi.training.divl_libero_adapter import make_batch
from openpi.training.libero_lwd_training import LiberoTrainingConfig, build_critic, encoder_config, initialize_training, critic_step
from train_libero_lwd import CRITIC_REPLAY_FIELDS
from critic_data_parallel import CriticDataParallel

R=Path('/opt/divl-trqam'); O=R/'artifacts/replay_action_audit_v3'

def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['scalar','divl'],required=True)
    p.add_argument('--counts',default='1,2,4');p.add_argument('--tag',required=True)
    p.add_argument('--repeats',type=int,default=6);args=p.parse_args()
    counts=[int(x) for x in args.counts.split(',')]; assert max(counts)<=jax.device_count()
    original=json.loads((R/'data/pilot_libero_v1/manifest.json').read_text())['records']
    branches=json.loads((R/'data/action_sensitive_replay_v2/manifest.json').read_text())['records']
    rng=np.random.default_rng(20261007);selection=[]
    # Representative fixed global batch: same 128/64/64 strata as formal training.
    for rows,n,kind in [(branches,128,'anchor'),(original,64,'original'),(branches,64,'continuation')]:
        for j in rng.integers(len(rows),size=n):selection.append((rows[j],kind))
    selection=[selection[i] for i in rng.permutation(256)];cache={};samples=[]
    for row,kind in selection:
        path=Path(row['path'])
        if str(path) not in cache:
            with np.load(path,allow_pickle=False) as f:cache[str(path)]={k:f[k] for k in CRITIC_REPLAY_FIELDS}
        e=cache[str(path)];n=len(e['rewards'])
        i=0 if kind=='anchor' else int(rng.integers(1,n)) if kind=='continuation' and n>1 else int(rng.integers(n))
        samples.append((e,i))
    cfg=LiberoTrainingConfig(value_type=args.method,reference_type='fixed',actor_updates=False)
    ec=encoder_config(R);tokenizer=DIVLTokenizer(ec);norm=normalize.load(R/'checkpoints/pi05_libero/assets/physical-intelligence/libero')
    t=time.monotonic();batch=make_batch(samples,tokenizer,norm,td_steps=10)
    batch['critic_k']=jnp.asarray([int(e['lengths'][i]) for e,i in samples]);jax.block_until_ready(batch)
    preprocessing=time.monotonic()-t
    critic=build_critic(cfg,ec);target,vo,qo=initialize_training(critic,2000,learning_rate=.0002)
    bundle=nnx.Dict(critic=critic,target=target,value_optimizer=vo,q_optimizer=qo)
    initial=jax.tree.map(lambda x:np.asarray(jax.device_get(x)),nnx.state(bundle))
    assert len(counts)==1,'Use one independent process per device count to release XLA arenas completely'
    results=[];reference=None;reference_params=None
    reference_path=O/f'critic_dp_reference_{args.method}.npz'
    for count in counts:
        dp=CriticDataParallel(critic,target,vo,qo,jax.devices()[:count]);dp.restore(initial)
        b=dp.batch(batch);jax.block_until_ready(b)
        tick=time.monotonic();m=jax.device_get(critic_step(critic,target,vo,qo,b,config=cfg));compile_s=time.monotonic()-tick
        first={k:float(v) for k,v in m.items()};assert all(np.isfinite(v) for v in first.values()) and first['update_applied']
        # Compare identical full global batch / initial FP32 model and optimizer.
        # Probe every parameter leaf, uniformly over its coordinates. Full
        # global loss and gradient norms are compared separately.
        probes=[]
        for leaf in jax.tree.leaves(nnx.state(critic,nnx.Param)):
            ids=np.linspace(0,leaf.size-1,min(64,leaf.size),dtype=np.int32)
            probes.append(np.asarray(jax.device_get(leaf.reshape(-1)[ids]),dtype=np.float32))
        params=np.concatenate(probes)
        if count==1:
            reference=first;reference_params=params;equivalence=dict(reference=True,passed=True)
            np.savez(reference_path,metrics=np.asarray(json.dumps(first)),parameter_samples=params)
        else:
            with np.load(reference_path,allow_pickle=False) as f:
                reference=json.loads(str(f['metrics']));reference_params=f['parameter_samples']
            diffs={k:abs(first[k]-reference[k]) for k in first}
            relative={k:diffs[k]/max(abs(reference[k]),1e-5) for k in ['loss_q','loss_v','td_error','q_grad_norm','value_grad_norm']}
            absmax=float(np.max(np.abs(params-reference_params)))
            rms=float(np.sqrt(np.mean((params-reference_params).astype(np.float64)**2)))
            equivalence=dict(metric_abs_diff=diffs,metric_relative_diff=relative,param_max_abs_diff=absmax,param_rms_diff=rms,
                             sampled_parameter_coordinates=len(params),parameter_probe='Uniform64 coordinates in EVERY parameter leaf; full losses/gradient norms also compared',
                             passed=all(x<.03 for x in relative.values()) and absmax<.001 and rms<1e-5,
                             tolerance='BF16 reduction order: primary metric relative<3%; FP32 parameter max abs<1e-3, RMS<1e-5')
        del params
        for _ in range(2):jax.block_until_ready(critic_step(critic,target,vo,qo,b,config=cfg))
        times=[]
        for _ in range(args.repeats):
            tick=time.monotonic();jax.block_until_ready(critic_step(critic,target,vo,qo,b,config=cfg));times.append(time.monotonic()-tick)
        result=dict(method=args.method,device_count=count,global_batch=256,compile_and_first_update_seconds=compile_s,
                    warm_update_seconds=times,median_update_seconds=float(np.median(times)),
                    gpu_samples_per_second=float(256/np.median(times)),preprocess_seconds=preprocessing,
                    estimated_end_to_end_samples_per_second=float(256/(np.median(times)+preprocessing)),
                    first_metrics=first,equivalence=equivalence,sharding=dp.evidence(b),
                    gpu_snapshot=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,power.draw','--format=csv,noheader'],capture_output=True,text=True).stdout)
        results.append(result);print('BENCHMARK',json.dumps(result),flush=True)
        (O/f'critic_dp_benchmark_{args.tag}.json').write_text(json.dumps(dict(actor_updated=False,disposable=True,results=results),indent=2))
    print('BENCHMARK_COMPLETE',args.tag,flush=True)

if __name__=='__main__':main()
