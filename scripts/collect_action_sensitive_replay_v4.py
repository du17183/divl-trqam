"""Seven real branches from fixed train anchors, with strict snapshot gates."""
import argparse,json,os,time,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from libero.libero import benchmark
from examples.libero.main import _get_libero_env,LIBERO_ENV_RESOLUTION
from examples.libero.collect_divl import collect_episode,policy_observation
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from libero_snapshot_v2 import restore_snapshot
from collect_action_sensitive_replay_v2 import BranchPolicy,NativeAudit
from collect_action_sensitive_replay_v3 import add_contract,save_npz
from action_sensitive_v4_common import R,D,O,atomic,sha,action_bounds,gt_features
WORKER_PORT=None
def initialize(ports,gpus):
 global WORKER_PORT
 slot=(mp.current_process()._identity[0]-1)%len(ports);WORKER_PORT=ports[slot]
 os.environ['MUJOCO_EGL_DEVICE_ID']=str(gpus[slot]);os.environ['OMP_NUM_THREADS']='2'

def branch(row):
 sid=row['state_id'];dest=D/'states'/f'{sid}.json';protocol_hash=sha(D/'protocol.json')
 if dest.exists():
  saved=json.loads(dest.read_text());assert saved['protocol_sha256']==protocol_hash;return saved
 assert sha(row['path'])==row['sha256'] and sha(row['candidate_path'])==row['candidate_sha256']
 with np.load(row['path'],allow_pickle=False) as f:source={k:f[k] for k in f.files}
 with np.load(row['candidate_path'],allow_pickle=False) as f:plans=f['raw'];norms=f['normalized'];padded=f['padded32']
 assert plans.shape==(7,10,7) and np.count_nonzero(padded[:,:,7:])==0
 i=row['source_index'];k=int(source['lengths'][i]);assert k==5
 flat=source['snapshot_states'][i];hidden=source['snapshot_metadata'][i];low,high=action_bounds()
 task=benchmark.get_benchmark_dict()['libero_10']().get_task(row['task_id'])
 env,prompt=_get_libero_env(task,LIBERO_ENV_RESOLUTION,row['seed']);env.seed(row['seed']);env.reset()
 client=WebsocketClientPolicy('127.0.0.1',WORKER_PORT);records=[];gt=None;gt_names=None
 try:
  for j,spec in enumerate(row['candidate_specs']):
   obs=restore_snapshot(env,flat,hidden);state_error=float(abs(env.get_sim_state()-flat).max());assert state_error==0
   po=policy_observation(obs,prompt)
   assert np.array_equal(po['observation/image'],source['base'][i])
   assert np.array_equal(po['observation/wrist_image'],source['wrist'][i])
   assert np.array_equal(po['observation/state'],source['state'][i])
   current_gt,names=gt_features(env,obs)
   if gt is None:gt=current_gt;gt_names=names
   else:assert np.array_equal(gt,current_gt) and names==gt_names
   policy=BranchPolicy(client,plans[j],row['seed'],i,env,source,j==0);audit=NativeAudit(env,k)
   episode=collect_episode(audit,policy,obs,prompt,row['remaining_steps'],replan_steps=5,action_horizon=10,gamma=.9999)
   episode_id=sid+f'_cf{j}';add_contract(episode,audit.native,low,high,row,episode_id,'counterfactual_train_v4')
   episode['dataset_version']=np.asarray('action_sensitive_replay_v4')
   error=abs(float(episode['mc_returns'][0])-float(source['mc_returns'][i])) if j==0 else None
   if j==0 and (error>1e-5 or policy.max_state_error>1e-7 or policy.max_pixel_error!=0 or policy.max_plan_error>1e-6):
    debug=dict(state_id=sid,source=row,baseline_reproduction_error=error,max_state_error=policy.max_state_error,
      max_pixel_error=policy.max_pixel_error,max_plan_error=policy.max_plan_error,reason='Original behavior continuation restore gate failed')
    save_npz(D/'invalid'/f'{sid}_baseline.npz',**episode);atomic(D/'invalid'/f'{sid}.json',debug)
    return dict(state_id=sid,valid=False,records=[],debug=debug)
   episode.update(state_id=np.asarray(sid),candidate_index=np.asarray(j),phase=np.asarray(row['achieved_phase']),
     initial_candidate_raw=plans[j],initial_candidate_normalized=norms[j],initial_candidate_padded32=padded[j],
     initial_planned_mask=np.arange(10)<5,perturbation_direction=np.asarray(spec['direction'],np.float32),perturbation_scale=np.asarray(spec['scale']))
   path=D/'episodes'/f'{episode_id}.npz';save_npz(path,**episode)
   records.append(dict(episode_id=episode_id,state_id=sid,task_id=row['task_id'],task_name=row['task_name'],initial_state_id=row['initial_state_id'],
    source_episode_id=row['trajectory_id'],source_index=i,source_path=row['path'],phase=row['achieved_phase'],assigned_phase=row['assigned_phase'],
    phase_fallback=row['phase_fallback'],candidate_index=j,candidate_name=spec['name'],candidate_kind=spec['kind'],candidate_geometry=spec,
    support_bucket=spec['bucket'],local_action_rms=spec['support_distance'],behavior_rms=spec['rms_from_behavior'],
    path=str(path),sha256=sha(path),future_return=float(episode['mc_returns'][0]),success=bool(episode['success']),
    terminal=bool(episode['terminal'][-1]),chunks=len(episode['rewards']),steps=int(episode['lengths'].sum()),
    initial_state_error=state_error,baseline_reproduction_error=error,k=k,remaining_steps=row['remaining_steps'],snapshot_id=sid,branch_seed=row['branch_seed']))
  save_npz(D/'states'/f'{sid}.npz',flat=flat,metadata=np.asarray(hidden),base=source['base'][i],wrist=source['wrist'][i],proprio=source['state'][i],
    candidate_raw=plans,candidate_normalized=norms,candidate_padded32=padded,valid_action_mask=np.arange(10)<5,
    instruction=source['prompt'],snapshot_id=np.asarray(sid),gt_diagnostic=gt,gt_names=np.asarray(gt_names),GT_formal_input=np.asarray(False))
  saved=dict(state_id=sid,valid=True,source=row,records=records,protocol_sha256=protocol_hash);atomic(dest,saved)
  print('V4_STATE_DONE',sid,[r['future_return'] for r in records],flush=True);return saved
 except Exception as exc:
  atomic(D/'invalid'/f'{sid}.exception.json',dict(error=repr(exc),source=row,completed_records=records));raise
 finally:env.close()

def main():
 p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=16);p.add_argument('--ports',default='8040,8041,8042,8043,8044,8045,8046,8047')
 p.add_argument('--gpus',default='0,1,2,3,4,5,6,7');p.add_argument('--limit',type=int);args=p.parse_args()
 ports=list(map(int,args.ports.split(',')));gpus=list(map(int,args.gpus.split(',')));assert len(ports)==len(gpus)
 protocol=json.loads((D/'protocol.json').read_text());assert sha(Path(__file__))==protocol['code_hashes'][Path(__file__).name]
 rows=json.loads((D/'state_selection_manifest.json').read_text())['records']
 if args.limit:rows=rows[:args.limit]
 started=time.monotonic();results=[]
 with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(ports,gpus)) as pool:
  jobs=[pool.submit(branch,row) for row in rows]
  for job in as_completed(jobs):
   results.append(job.result());elapsed=time.monotonic()-started
   atomic(D/'branches_progress.json',dict(status='collecting',completed=len(results),target=len(rows),elapsed_seconds=elapsed,workers=args.workers,inference_gpus=gpus))
 if args.limit:atomic(O/'branches_preflight.json',dict(passed=all(x['valid'] for x in results),records=results));return
 valid=[x for x in results if x['valid']];records=[r for x in valid for r in x['records']];assert len(records)==len(valid)*7
 atomic(D/'manifest.json',dict(version='action_sensitive_replay_v4',states=len(valid),attempted_states=len(rows),invalid_states=len(rows)-len(valid),outcomes=len(records),
  records=sorted(records,key=lambda x:x['episode_id']),split_hash=protocol['split_hash'],config_hash=sha(D/'protocol.json'),
  selection_hash=sha(D/'state_selection_manifest.json'),all_outcomes_retained=True,actor_updated=False,GT_formal_input=False,elapsed_seconds=time.monotonic()-started))
 atomic(D/'branches_progress.json',dict(status='complete',completed=len(results),target=len(rows),elapsed_seconds=time.monotonic()-started))
 print('V4_COLLECTION_COMPLETE',len(valid),len(records),flush=True)

if __name__=='__main__':main()
