# Public release: deployment paths anonymized; algorithm equations unchanged.
"""A preregistered20-case paired dev pilot on immutable original initial scenes."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,json,multiprocessing as mp,hashlib
import numpy as np
from libero.libero import benchmark
from examples.libero.main import _get_libero_env,LIBERO_ENV_RESOLUTION
from examples.libero.collect_divl import collect_episode
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from libero_snapshot_v2 import migrate_v1_for_branch,restore_snapshot
from collect_pilot_libero import EnvAudit
from run_replay_v2_pipeline import atomic

R=Path('/opt/divl-trqam');O=R/'artifacts/replay_action_audit_v3'
class ExplicitNoise:
    def __init__(self,client,seed):self.client=client;self.seed=seed;self.index=0
    def infer(self,obs):
        result=self.client.infer({**obs,'_noise_seed':self.seed*1000+self.index});self.index+=1;return result
def evaluate(row,method,port):
    with np.load(row['path'],allow_pickle=False) as f:source={k:f[k] for k in f.files}
    assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256']
    env,prompt=_get_libero_env(benchmark.get_benchmark_dict()['libero_10']().get_task(row['task_id']),LIBERO_ENV_RESOLUTION,row['seed']);env.seed(row['seed']);env.reset()
    try:
        hidden=migrate_v1_for_branch(env,source['snapshot_states'][0],source['snapshot_metadata'][0],source['model_xml'])
        obs=restore_snapshot(env,source['snapshot_states'][0],hidden);assert np.array_equal(env.get_sim_state(),source['snapshot_states'][0])
        audit=EnvAudit(env);policy=ExplicitNoise(WebsocketClientPolicy('127.0.0.1',port),row['seed'])
        ep=collect_episode(audit,policy,obs,prompt,max_steps=520,replan_steps=5,action_horizon=10,gamma=.9999)
        first_delta=float(np.linalg.norm(ep['actions'][0]-source['actions'][0]))
        ep.update(task_id=np.asarray(row['task_id']),task_name=np.asarray(row['task_name']),init_id=np.asarray(row['initial_state_id']),seed=np.asarray(row['seed']),split=np.asarray('dev'),episode_id=np.asarray(row['episode_id']),env_step_rewards=np.asarray(audit.step_rewards,np.float32))
        folder=R/'data/fixed_qam_v2_pilot'/method;folder.mkdir(parents=True,exist_ok=True);path=folder/f'{row["episode_id"]}.npz';assert not path.exists()
        np.savez_compressed(path,**ep)
        result=dict(task_id=row['task_id'],task_name=row['task_name'],initial_state_id=row['initial_state_id'],seed=row['seed'],episode_id=row['episode_id'],baseline_success=bool(source['success']),updated_success=bool(ep['success']),baseline_return=float(source['mc_returns'][0]),updated_return=float(ep['mc_returns'][0]),
            baseline_steps=int(source['lengths'].sum()),updated_steps=int(ep['lengths'].sum()),first_action_change=first_delta,initial_state_error=0.,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        print('FIXED_ACTOR_DEV_EPISODE',json.dumps(result),flush=True);return result
    finally:env.close()
def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['scalar','divl'],required=True);p.add_argument('--port',type=int,default=8030);args=p.parse_args()
    assert json.loads((O/f'fixed_qam_{args.method}_complete.json').read_text())['passed']
    manifest=json.loads((R/'data/dev_libero_v1/manifest.json').read_text());selected=[]
    for t in range(10):selected.extend(sorted([v for v in manifest['records'] if v['task_id']==t],key=lambda r:r['initial_state_id'])[:2])
    assert len(selected)==20;records=[]
    with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn')) as pool:
        for job in as_completed([pool.submit(evaluate,r,args.method,args.port) for r in selected]):records.append(job.result())
    records.sort(key=lambda r:(r['task_id'],r['initial_state_id']));base=np.array([r['baseline_success'] for r in records],float);after=np.array([r['updated_success'] for r in records],float);delta=after-base;rng=np.random.default_rng(20261007)
    index=np.array([np.concatenate([rng.choice(np.arange(2*t,2*t+2),2,replace=True) for t in range(10)]) for _ in range(5000)])
    output=dict(method=args.method,checkpoint_step=50,episodes=20,protocol='First2 dev initial IDs/task chosen without outcomes; same source scene/explicit noise/controller/reward; final split not accessed; no checkpoint selection',baseline_success=float(base.mean()),updated_success=float(after.mean()),success_delta=float(delta.mean()),paired_success_delta_ci95=np.quantile(delta[index].mean(axis=1),[.025,.975]).tolist(),
        task_wise={str(t):dict(baseline=float(base[t*2:t*2+2].mean()),updated=float(after[t*2:t*2+2].mean())) for t in range(10)},mean_initial_action_change=float(np.mean([r['first_action_change'] for r in records])),records=records,
        note='Small engineering pilot, not final LIBERO-Long evidence or a DIVL advantage claim. Baseline outcomes are from the existing exactly reproduced immutable frozen dataset.')
    atomic(O/f'fixed_qam_{args.method}_dev_pilot.json',output);(R/'reports'/f'{args.method}_fixed_qam_v2_pilot.md').write_text('# Fixed-reference actor pilot\n\n'+json.dumps(output,indent=2)+'\n')

if __name__=='__main__':main()
