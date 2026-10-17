# Public release: deployment paths anonymized; algorithm equations unchanged.
"""LIBERO-only CLI. Frozen-actor critic training or <=5 disposable smoke steps."""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import time
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from openpi.models import model as model_lib
from openpi.models.divl_tokenizer import DIVLTokenizer
from openpi.shared import normalize
from openpi.training import beta_training, config as actor_config
from openpi.training.divl_libero_adapter import EpisodeReplay,make_batch
from openpi.training.libero_lwd_training import LiberoTrainingConfig, build_critic, encoder_config, initialize_training, training_step
from openpi.training.qam_libero_adapter import LiberoQAMBatchBuilder
from openpi.training.qam_training import QAMTrainingConfig, QAMBatch, initialize_actor_training
from openpi.training.precision import assert_fp32_master

ROOT=Path('/opt/divl-trqam')
CRITIC_REPLAY_FIELDS=('base','wrist','state','actions','action_mask','lengths','rewards','discounts','terminal','prompt',
                     'task_id','task_name','init_id','seed','split','success')


def digest(model):
    h=hashlib.sha256()
    for x in jax.tree.leaves(nnx.state(model,nnx.Param)):
        h.update(np.asarray(x).tobytes())
    return h.hexdigest()


def execute(args, value_type, reference_type):
    cfg=LiberoTrainingConfig(value_type=value_type,reference_type=reference_type,
        actor_updates=args.smoke,smoke_steps=args.steps if args.smoke else 0)
    aq=actor_config.get_config('pi05_libero').model
    qcfg=QAMTrainingConfig(critic_horizon=10,replan_steps=5,sample_from_reference=True)
    norm=normalize.load(ROOT/'checkpoints/pi05_libero/assets/physical-intelligence/libero')
    enc_cfg=encoder_config(ROOT)
    builder=LiberoQAMBatchBuilder(aq,norm,norm,DIVLTokenizer(enc_cfg),training_config=qcfg,td_steps=1)
    directory=ROOT/'data/feasibility_libero_long' if args.smoke else Path(args.replay)
    if args.smoke:
        samples=[]
        for p in sorted(directory.glob('*.npz'))[:2]:
            with np.load(p,allow_pickle=False) as f: ep={k:f[k] for k in f.files}
            samples.append((ep,0))
    else:
        split=json.loads((ROOT/'configs/splits/libero_factorial_v1.json').read_text())
        manifest=json.loads((directory/'manifest.json').read_text())
        if manifest['split_hash']!=split['sha256']:raise ValueError('Replay split changed')
        for record in manifest['records']:
            if hashlib.sha256(Path(record['path']).read_bytes()).hexdigest()!=record['sha256']:raise ValueError('Frozen replay file hash mismatch')
        replay=EpisodeReplay(directory/'episodes',split='train',seed=cfg.seed,fields=CRITIC_REPLAY_FIELDS);replay.refresh()
        for ep in replay.episodes:
            task=str(ep['task_name']);init=int(ep['init_id'])
            if init not in split['tasks'][task]['train']:raise ValueError('Initial state leakage')
        samples=replay.sample(args.batch_size)
    critic=build_critic(cfg,enc_cfg)
    target,vopt,qopt=initialize_training(critic,args.steps,learning_rate=cfg.learning_rate)
    assert_fp32_master(critic);assert_fp32_master(target)
    q_before=digest(critic.q_head)
    encoder_before=digest(critic.encoder)
    actor=reference=actor_opt=beta=beta_opt=None
    actor_before=reference_before=None
    if args.smoke:
        actor=aq.load(model_lib.restore_params(ROOT/'checkpoints/pi05_libero/params',dtype=jnp.bfloat16))
        reference,actor_opt=initialize_actor_training(actor,qcfg)
        if reference_type=='dynamic':beta,beta_opt=beta_training.initialize_dynamic_beta(reference,learning_rate_fn=qcfg.learning_rate)
        actor_before=digest(actor);reference_before=digest(reference)
    rng=jax.random.key(cfg.seed)
    metrics=[];sample_hash=hashlib.sha256();started=time.monotonic()
    checkpoints={250,500,1000,2000}
    for step in range(1,args.steps+1):
        if not args.smoke and step>1:samples=replay.sample(args.batch_size)
        sample_hash.update(np.asarray([(int(e.get('task_id',0)),int(e.get('init_id',0)),int(e.get('seed',0)),i) for e,i in samples],dtype=np.int64).tobytes())
        if args.smoke:
            batch=builder(samples)
        else:
            data=make_batch(samples,builder.critic_tokenizer,norm,td_steps=1)
            data['critic_k']=jnp.asarray([int(e['lengths'][i]) for e,i in samples])
            batch=QAMBatch(actor_observation=None,critic=data)
        if not np.array_equal(np.asarray(batch.critic['critic_k']),[e['lengths'][i] for e,i in samples]):raise AssertionError('Actual k changed')
        tick=time.monotonic()
        rng,m=training_step(critic,target,vopt,qopt,batch,config=cfg,actor=actor,reference=reference,
            actor_opt=actor_opt,beta=beta,beta_opt=beta_opt,action_stats=builder.action_stats,rng=rng,qam_config=qcfg)
        m={k:float(v) for k,v in jax.device_get(m).items()}
        if not all(np.isfinite(v) for v in m.values()) or not m['update_applied']:raise FloatingPointError(m)
        row={'step':step,'update_seconds':time.monotonic()-tick,**m};metrics.append(row)
        if step==1 or step%25==0:print('UPDATE '+json.dumps({'arm':value_type+'_'+reference_type,**row}),flush=True)
        if not args.smoke and step in checkpoints:
            out=ROOT/'checkpoints/critic_pilot_v1'/value_type/f'step{step:04d}'
            out.parent.mkdir(parents=True,exist_ok=True)
            with ocp.PyTreeCheckpointer() as ck:
                ck.save(out.resolve(),{'critic':nnx.state(critic,nnx.Param).to_pure_dict(),'target':nnx.state(target,nnx.Param).to_pure_dict(),
                                      'value_optimizer':nnx.state(vopt).to_pure_dict(),'q_optimizer':nnx.state(qopt).to_pure_dict(),'rng':jax.random.key_data(rng)})
            (out.parent/f'metrics_step{step:04d}.json').write_text(json.dumps({'config':asdict(cfg),'batch_size':args.batch_size,'sample_order_sha256':sample_hash.hexdigest(),'metrics':metrics},indent=2))
            (out.parent/f'replay_rng_step{step:04d}.json').write_text(json.dumps(replay.rng.bit_generator.state))
        if not args.smoke:
            (ROOT/'artifacts'/f'critic_{value_type}_progress.json').write_text(json.dumps({'step':step,'total_steps':args.steps,'batch':args.batch_size,'elapsed_seconds':time.monotonic()-started,'latest':row,'sample_order_hash':sample_hash.hexdigest(),'q_initial_hash':q_before,'encoder_initial_hash':encoder_before}))
    changed=digest(critic.q_head)!=q_before
    assert changed
    result={'passed':True,'config':asdict(cfg),'batch_size':len(samples),'steps':args.steps,'q_initial_hash':q_before,
            'encoder_initial_hash':encoder_before,'sample_order_hash':sample_hash.hexdigest(),'q_changed':changed,
            'actor_initial_hash':actor_before,'actor_changed':digest(actor)!=actor_before if actor is not None else False,
            'reference_changed':digest(reference)!=reference_before if reference is not None else False,
            'beta_enabled':beta is not None,'k_definition':'actual executed prefix','td_chunks':1,'gamma':.9999,
            'elapsed_seconds':time.monotonic()-started,'metrics':metrics}
    if args.smoke:
        assert result['actor_changed']
        assert result['reference_changed']==(reference_type=='dynamic')
    else:
        assert not result['actor_changed'] and not result['beta_enabled']
    out=ROOT/'artifacts'/f'{"smoke" if args.smoke else "critic"}_{value_type}_{reference_type}.json'
    out.write_text(json.dumps(result,indent=2))
    print('ARM_COMPLETE '+json.dumps({k:v for k,v in result.items() if k!='metrics'}),flush=True)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--value-type',choices=['scalar','divl'],default='scalar')
    p.add_argument('--reference-type',choices=['fixed','dynamic'],default='fixed');p.add_argument('--steps',type=int,default=2000)
    p.add_argument('--batch-size',type=int,default=256);p.add_argument('--replay',default=str(ROOT/'data/pilot_libero_v1'))
    p.add_argument('--smoke',action='store_true');p.add_argument('--four-arm-smoke',action='store_true');args=p.parse_args()
    for name in ['phase0_beta_fp32_metrics.json','beta_target_audit.json']:
        assert json.loads((ROOT/'artifacts'/name).read_text())['passed'],name
    if args.four_arm_smoke:
        args.smoke=True
        if not 1<=args.steps<=5:raise ValueError('Four-arm smoke must be 1–5 disposable steps')
        rows=[]
        for value in ['scalar','divl']:
            for ref in ['fixed','dynamic']:
                rows.append(execute(args,value,ref));gc.collect()
        assert len({x['q_initial_hash'] for x in rows})==1
        assert len({x['encoder_initial_hash'] for x in rows})==1
        assert len({x['actor_initial_hash'] for x in rows})==1
        assert len({x['sample_order_hash'] for x in rows})==1
        (ROOT/'artifacts/four_arm_smoke_test.json').write_text(json.dumps({'passed':True,'arms':rows,'scope':'disposable optimizer tests; no performance comparison'},indent=2))
        (ROOT/'reports/phase1_libero_training_entry.md').write_text('# Phase 1 LIBERO entry\n\nPASS. One trainer with independent value_type/reference_type switches.\n\n'
            'Four arms completed disposable real LIBERO forward/backward/optimizer steps. Encoder, Double-Q, actor initialization and sample order hashes match.\n\n'
            'Scalar uses expectile V; DIVL uses 201 bins/adaptive quantile. Actor remains frozen by default; formal actor training is prohibited by config.\n')
        (ROOT/'reports/current_stage_summary.md').write_text('# Current stage\n\nPhases 0, 0B, 1 and 1B passed. Next: immutable initial-state split and expert provenance audit.\n\nNo formal actor/QAM training has started.\n')
    else:
        if not args.smoke and args.reference_type!='fixed':raise ValueError('Dynamic reference is disabled in critic-only phase')
        execute(args,args.value_type,args.reference_type)


if __name__=='__main__':main()
