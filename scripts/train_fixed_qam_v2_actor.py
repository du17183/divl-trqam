# Public release: deployment paths anonymized; algorithm equations unchanged.
"""One smoke update plus a bounded fixed-reference actor pilot, after real gates."""
from pathlib import Path
import argparse,json,time,hashlib,dataclasses
import numpy as np
import flax.nnx as nnx
import jax,jax.numpy as jnp
import orbax.checkpoint as ocp
from openpi.models import model as model_lib
from openpi.models.divl_tokenizer import DIVLTokenizer
from openpi.shared import normalize
from openpi.training import config as actor_config
from openpi.training.libero_lwd_training import LiberoTrainingConfig,encoder_config,build_critic
from openpi.training.qam_libero_adapter import LiberoQAMBatchBuilder
from openpi.training.qam_training import QAMTrainingConfig,initialize_actor_training,actor_train_step
from openpi.training.precision import assert_fp32_master
from train_libero_lwd import digest
from critic_v2_io import restore_critic

R=Path('/opt/divl-trqam');O=R/'artifacts/replay_action_audit_v3'
def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['scalar','divl'],required=True);args=p.parse_args()
    gate=json.loads((O/'v2_new_gradient_metrics.json').read_text());assert gate['methods'][args.method]['passed']
    cfg=LiberoTrainingConfig(value_type=args.method);enc=encoder_config(R);critic=build_critic(cfg,enc)
    critic=restore_critic(critic,R/'checkpoints/critic_replay_v2'/args.method/'step2000');critic_initial=digest(critic)
    aq=actor_config.get_config('pi05_libero').model;norm=normalize.load(R/'checkpoints/pi05_libero/assets/physical-intelligence/libero')
    qcfg=QAMTrainingConfig(critic_horizon=10,replan_steps=5,sample_from_reference=True)
    actor=aq.load(model_lib.restore_params(R/'checkpoints/pi05_libero/params',dtype=jnp.bfloat16));reference,opt=initialize_actor_training(actor,qcfg)
    assert_fp32_master(actor);reference_initial=digest(reference);actor_initial=digest(actor)
    builder=LiberoQAMBatchBuilder(aq,norm,norm,DIVLTokenizer(enc),training_config=qcfg,td_steps=10)
    manifest=json.loads((R/'data/action_sensitive_replay_v2/manifest.json').read_text());records=manifest['records'];split=json.loads((R/'configs/splits/libero_factorial_v1.json').read_text())
    out=R/'checkpoints/fixed_qam_v2'/args.method;assert not out.exists();out.mkdir(parents=True)
    protocol=dict(method=args.method,steps=50,smoke_steps=1,batch=2,actor_lr=qcfg.learning_rate,seed=cfg.seed,qam_config=dataclasses.asdict(qcfg),critic_steps=2000,critic_frozen=True,
        reference='independent immutable official pi05_libero, fixed; no beta optimizer',training='original actor_train_step: action derivative, reference VJP adjoint, velocity regression; not -Q',
        primary_actor_checkpoint=50,replay_sha256=hashlib.sha256((R/'data/action_sensitive_replay_v2/manifest.json').read_bytes()).hexdigest(),reference_initial_hash=reference_initial,actor_initial_hash=actor_initial,critic_initial_hash=critic_initial)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));rng=jax.random.key(cfg.seed);replay_rng=np.random.default_rng(cfg.seed);order=hashlib.sha256();metrics=[];start=time.monotonic();cache={}
    probe=None;before=None
    @nnx.jit
    def actions(model,obs):return model.sample_actions(jax.random.key(100),obs,noise=jnp.asarray(np.random.default_rng(100).normal(size=(2,10,32)),jnp.float32),num_steps=10)
    for step in range(1,51):
        idx=replay_rng.integers(len(records),size=2);samples=[]
        for n in idx:
            row=records[n];assert row['initial_state_id'] in split['tasks'][row['task_name']]['train']
            if row['episode_id'] not in cache:
                assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256']
                with np.load(row['path'],allow_pickle=False) as f:cache[row['episode_id']]={k:f[k] for k in f.files}
            samples.append((cache[row['episode_id']],0))
        order.update(json.dumps([records[n]['episode_id'] for n in idx],separators=(',',':')).encode());batch=builder(samples)
        if probe is None:probe=batch.actor_observation;before=np.asarray(actions(actor,probe))
        rng,m=actor_train_step(actor,reference,critic,opt,batch.actor_observation,batch.critic['obs'],builder.action_stats,rng,config=qcfg)
        m={k:float(v) for k,v in jax.device_get(m).items()};assert all(np.isfinite(v) for v in m.values()),m
        if step==1:
            assert m['actor/terminal_adjoint_norm']>0 and m['actor/grad_norm']>0,'QAM smoke lacks an actual critic adjoint or policy parameter gradient'
        m.update(step=step,elapsed_seconds=time.monotonic()-start);metrics.append(m)
        (O/f'fixed_qam_{args.method}_progress.json').write_text(json.dumps(dict(step=step,total_steps=50,latest=m)))
        if step in [1,5,50]:
            drift=float(np.linalg.norm(np.asarray(actions(actor,probe))-before));assert drift>0,'Optimizer did not change actions at FP32 master precision'
            assert digest(reference)==reference_initial and digest(critic)==critic_initial
            params=nnx.state(actor,nnx.Param).to_pure_dict()
            with ocp.PyTreeCheckpointer() as c:c.save((out/f'step{step:04d}').resolve(),dict(params=params,optimizer=nnx.state(opt).to_pure_dict(),rng=jax.random.key_data(rng)))
            (out/f'metrics_step{step:04d}.json').write_text(json.dumps(dict(metrics=metrics,action_drift=drift,sample_order_hash=order.hexdigest(),reference_unchanged=True,critic_unchanged=True),indent=2))
            if step==50:
                inferdir=out/'inference_step0050';inferdir.mkdir()
                with ocp.PyTreeCheckpointer() as c:c.save((inferdir/'params').resolve(),dict(params=params))
                (inferdir/'assets').symlink_to(R/'checkpoints/pi05_libero/assets',target_is_directory=True)
        print('FIXED_QAM_UPDATE',args.method,json.dumps(m),flush=True)
    result=dict(passed=True,protocol=protocol,steps=50,smoke_passed=True,reference_unchanged=True,critic_unchanged=True,actor_changed=digest(actor)!=actor_initial,
        action_drift=drift,sample_order_hash=order.hexdigest(),elapsed_seconds=time.monotonic()-start,metrics=metrics)
    assert result['actor_changed'];(O/f'fixed_qam_{args.method}_complete.json').write_text(json.dumps(result,indent=2))
    print('FIXED_QAM_ACTOR_COMPLETE',args.method,flush=True)

if __name__=='__main__':main()
