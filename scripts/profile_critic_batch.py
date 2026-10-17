# Public release: deployment paths anonymized; algorithm equations unchanged.
"""Disposable critic-only batch profile, no formal checkpoint overwrite."""
import argparse
from pathlib import Path
import json
import time
import jax
import jax.numpy as jnp
import numpy as np
from openpi.models.divl_tokenizer import DIVLTokenizer
from openpi.shared import normalize
from openpi.training.divl_libero_adapter import EpisodeReplay,make_batch
from openpi.training.libero_lwd_training import LiberoTrainingConfig,encoder_config,build_critic,initialize_training,critic_step
from train_libero_lwd import CRITIC_REPLAY_FIELDS

ROOT=Path('/opt/divl-trqam')

def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=int,required=True);p.add_argument('--method',choices=['scalar','divl'],required=True);args=p.parse_args()
    assert json.loads((ROOT/'artifacts/pilot_replay_audit.json').read_text())['passed']
    # Profile only after at least one matched 250-update correctness checkpoint.
    assert all((ROOT/'checkpoints/critic_pilot_v1'/m/'metrics_step0250.json').exists() for m in ['scalar','divl'])
    cfg=LiberoTrainingConfig(value_type=args.method);enc=encoder_config(ROOT);tok=DIVLTokenizer(enc)
    norm=normalize.load(ROOT/'checkpoints/pi05_libero/assets/physical-intelligence/libero')
    replay=EpisodeReplay(ROOT/'data/pilot_libero_v1/episodes',seed=cfg.seed,fields=CRITIC_REPLAY_FIELDS);replay.refresh()
    critic=build_critic(cfg,enc);target,vo,qo=initialize_training(critic,2000,learning_rate=cfg.learning_rate)
    times=[];io=[]
    for step in range(8):
        tick=time.monotonic();samples=replay.sample(args.batch);batch=make_batch(samples,tok,norm,td_steps=1)
        batch['critic_k']=jnp.asarray([int(e['lengths'][i]) for e,i in samples]);jax.block_until_ready(batch);reading=time.monotonic()-tick
        tick=time.monotonic();metrics=jax.device_get(critic_step(critic,target,vo,qo,batch,config=cfg));elapsed=time.monotonic()-tick
        if not bool(metrics['update_applied']) or not all(np.isfinite(v) for v in metrics.values()):raise FloatingPointError('Profile update rejected')
        if step>=3:times.append(elapsed);io.append(reading)
    result={'method':args.method,'batch':args.batch,'compute_seconds':float(np.mean(times)),'preprocessing_seconds':float(np.mean(io)),
            'samples_per_second':args.batch/float(np.mean(times)+np.mean(io)),'gpu_only_samples_per_second':args.batch/float(np.mean(times)),
            'scope':'8 disposable critic steps, first 3 excluded as warmup; all backbone parameters train; no actor/beta/QAM'}
    (ROOT/'artifacts'/f'critic_profile_{args.method}_b{args.batch}.json').write_text(json.dumps(result,indent=2));print('PROFILE_COMPLETE '+json.dumps(result),flush=True)

if __name__=='__main__':main()
