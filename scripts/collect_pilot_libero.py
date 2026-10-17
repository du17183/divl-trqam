# Public release: deployment paths anonymized; algorithm equations unchanged.
"""Train-only 200 rollout pilot and independent 100-episode dev characterization."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
from libero.libero import benchmark
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from examples.libero.main import _get_libero_env,LIBERO_ENV_RESOLUTION,LIBERO_DUMMY_ACTION
from examples.libero.collect_divl import collect_episode
from openpi.training.libero_replay_contract import behavior_sequences,validate_episode
from libero_snapshot import take_snapshot

ROOT=Path('/opt/divl-trqam')


class EnvAudit:
    def __init__(self,env):self.env=env;self.step_rewards=[]
    def step(self,action):
        out=self.env.step(action);self.step_rewards.append(float(out[1]));return out
    def check_success(self):return self.env.check_success()


class PolicyAudit:
    def __init__(self,client,env,seed,low,high):
        self.client=client;self.env=env;self.seed=seed;self.low=low;self.high=high
        self.plans=[];self.snapshots=[];self.hidden=[];self.noise_seeds=[]
    def infer(self,obs):
        flat,meta=take_snapshot(self.env);self.snapshots.append(flat);self.hidden.append(meta)
        seed=self.seed*1000+len(self.plans);self.noise_seeds.append(seed)
        out=self.client.infer({**obs,'_noise_seed':seed})
        plan=np.asarray(out['actions'],dtype=np.float32)
        assert plan.shape==(10,7) and np.isfinite(plan).all()
        self.plans.append(plan.copy());return out


def task_group(task_id,split_name,port):
    suite=benchmark.get_benchmark_dict()['libero_10']();task=suite.get_task(task_id)
    split=json.loads((ROOT/'configs/splits/libero_factorial_v1.json').read_text())
    ids=split['tasks'][task.name][split_name][:10];repetitions=2 if split_name=='train' else 1
    initial=suite.get_task_init_states(task_id)
    directory=ROOT/('data/pilot_libero_v1' if split_name=='train' else 'data/dev_libero_v1')
    (directory/'episodes').mkdir(parents=True,exist_ok=True)
    norm=json.loads((ROOT/'checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json').read_text())['norm_stats']['actions']
    low=np.asarray(norm['q01'],np.float32)[:7];high=np.asarray(norm['q99'],np.float32)[:7]
    client=WebsocketClientPolicy('127.0.0.1',port)
    env,prompt=_get_libero_env(task,LIBERO_ENV_RESOLUTION,20261007+task_id)
    records=[]
    try:
        for init_id in ids:
            for rep in range(repetitions):
                seed=2026100700+task_id*10000+int(init_id)*10+rep+(500000 if split_name=='dev' else 0)
                path=directory/'episodes'/f'task{task_id:02d}_init{init_id:03d}_rep{rep}.npz'
                if path.exists():
                    with np.load(path,allow_pickle=False) as f:e={k:f[k] for k in f.files}
                    validate_episode(e)
                else:
                    env.seed(seed);env.reset();obs=env.set_init_state(initial[init_id])
                    for _ in range(10):obs,_,_,_=env.step(LIBERO_DUMMY_ACTION)
                    audited=EnvAudit(env);policy=PolicyAudit(client,env,seed,low,high)
                    e=collect_episode(audited,policy,obs,prompt,max_steps=520,replan_steps=5,action_horizon=10,gamma=.9999)
                    full,mask=behavior_sequences(e)
                    actual=np.concatenate([a[m] for a,m in zip(e['actions'],e['action_mask'],strict=True)])
                    e.update(actor_bc_executed_sequence=full,actor_bc_mask=mask,
                        critic_executed_actions=e['actions'].copy(),critic_action_mask=e['action_mask'].copy(),critic_k=e['lengths'].copy(),
                        raw_env_actions=actual,normalized_actor_actions=2*(actual-low)/(high-low+1e-6)-1,
                        policy_action_plan=np.stack(policy.plans),policy_action_normalized=2*(np.stack(policy.plans)-low)/(high-low+1e-6)-1,
                        env_step_rewards=np.asarray(audited.step_rewards,np.float32),snapshot_states=np.stack(policy.snapshots),snapshot_metadata=np.asarray(policy.hidden),
                        policy_noise_seeds=np.asarray(policy.noise_seeds,np.int64),task_id=np.asarray(task_id),task_name=np.asarray(task.name),
                        suite=np.asarray('libero_10'),init_id=np.asarray(init_id),seed=np.asarray(seed),source=np.asarray('rollout'),split=np.asarray(split_name),
                        episode_id=np.asarray(path.stem),dataset_version=np.asarray('pilot_libero_v1'),split_hash=np.asarray(split['sha256']))
                    e['model_xml']=np.asarray(env.sim.model.get_xml())
                    validate_episode(e)
                    assert len(e['env_step_rewards'])==int(e['lengths'].sum())
                    offset=0
                    for reward,length in zip(e['rewards'],e['lengths'],strict=True):
                        native=sum(.9999**j*float(e['env_step_rewards'][offset+j]) for j in range(int(length)))
                        if abs(native-float(reward))>1e-6:raise AssertionError('Collector reward differs from native environment reward')
                        offset+=int(length)
                    tmp=path.with_suffix('.partial')
                    with tmp.open('wb') as f:np.savez_compressed(f,**e)
                    tmp.replace(path)
                row={'episode_id':path.stem,'task_id':task_id,'task_name':task.name,'initial_state_id':int(init_id),'seed':seed,'repetition':rep,
                     'split':split_name,'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'success':bool(e['success']),
                     'environment_steps':int(e['lengths'].sum()),'chunks':len(e['rewards']),'return':float(e['mc_returns'][0])}
                records.append(row)
                print('EPISODE '+json.dumps(row),flush=True)
    finally:env.close()
    return records


def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8027);p.add_argument('--workers',type=int,default=3);args=p.parse_args()
    assert json.loads((ROOT/'artifacts/four_arm_smoke_test.json').read_text())['passed']
    assert json.loads((ROOT/'artifacts/pilot_snapshot_preflight.json').read_text())['passed']
    split=json.loads((ROOT/'configs/splits/libero_factorial_v1.json').read_text())
    records=[];start=time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as pool:
        futures=[pool.submit(task_group,i,s,args.port) for s in ['train','dev'] for i in range(10)]
        for future in as_completed(futures):
            records.extend(future.result())
            (ROOT/'artifacts/pilot_collection_progress.json').write_text(json.dumps({'completed':len(records),'target':300,'elapsed_seconds':time.monotonic()-start,'task_groups_completed':len(records)//10},indent=2))
    assert len({(e['split'],e['episode_id']) for e in records})==300
    assert len({e['seed'] for e in records})==300
    for name in ['train','dev']:
        items=sorted([e for e in records if e['split']==name],key=lambda e:e['episode_id'])
        expected=200 if name=='train' else 100;assert len(items)==expected
        manifest={'version':'pilot_libero_v1' if name=='train' else 'dev_libero_v1','split_hash':split['sha256'],'episodes':len(items),
                  'transitions':sum(e['environment_steps'] for e in items),'chunks':sum(e['chunks'] for e in items),
                  'successes':sum(e['success'] for e in items),'failures':sum(not e['success'] for e in items),'expert_included':0,
                  'task_distribution':{str(i):sum(e['task_id']==i for e in items) for i in range(10)},'records':items}
        directory=ROOT/('data/pilot_libero_v1' if name=='train' else 'data/dev_libero_v1')
        (directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        if name=='dev':(ROOT/'artifacts/frozen_dev_baseline.json').write_text(json.dumps(manifest,indent=2)+'\n')
        else:(ROOT/'reports/pilot_replay_report.md').write_text('# Pilot replay\n\n'
            f'{len(items)} train deployment episodes, {manifest["transitions"]} environment steps; successes {manifest["successes"]}, failures {manifest["failures"]}.\n\n'
            'Frozen official pi05_libero; all 10 tasks; 10 train initial states/task, two repetitions. No experts mixed. Full RGB/proprio, real actions, k/masks, full beta labels, snapshots and controller/physics state saved.\n')
    (ROOT/'reports/current_stage_summary.md').write_text('# Current stage\n\nPhase 3 pilot complete: 200 train episodes and 100 independent dev characterization episodes. Final eval IDs unused.\n\n'
        'Actor, checkpoint, reward, controller and normalization unchanged. Next: matched critic-only Scalar/DIVL updates and learned action diagnostics.\n')
    print('PILOT_COMPLETE '+json.dumps({'episodes':300,'elapsed_seconds':time.monotonic()-start}),flush=True)


if __name__=='__main__':main()
