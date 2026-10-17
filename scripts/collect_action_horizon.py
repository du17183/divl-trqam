"""Same100 snapshots/directions, real H1/H3/H5 branches and paired recovery traces."""
import argparse,json,os,time,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from libero.libero import benchmark
from examples.libero.main import _get_libero_env,LIBERO_ENV_RESOLUTION
from examples.libero.collect_divl import policy_observation
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from libero_snapshot_v2 import restore_snapshot
from horizon_common import R,D,O,SEED,sha,atomic,save_npz,load
from action_sensitive_v4_common import action_bounds,normalized
from openpi.training.libero_replay_contract import behavior_sequences,validate_episode
PORT=None
def initialize(ports,gpus):
 global PORT
 i=(mp.current_process()._identity[0]-1)%len(ports);PORT=ports[i]
 os.environ['MUJOCO_EGL_DEVICE_ID']=str(gpus[i]);os.environ['OMP_NUM_THREADS']='2'

def physical_state(env,obs,names):
 sim=env.sim
 pos=np.stack([sim.data.body_xpos[sim.model.body_name2id(n)] for n in names]) if names else np.zeros((0,3))
 quat=np.stack([sim.data.body_xquat[sim.model.body_name2id(n)] for n in names]) if names else np.zeros((0,4))
 return (np.r_[obs['robot0_eef_pos'],obs['robot0_eef_quat'],obs['robot0_gripper_qpos']],
   np.asarray(obs['robot0_eef_pos']),np.asarray(obs['robot0_eef_quat']),pos,quat,np.asarray(env.get_sim_state()).copy())

def collect(env,client,source,row,candidate,h,names):
 i=row['source_index'];prompt=str(source['prompt']);remaining=row['remaining_steps']
 obs=restore_snapshot(env,source['snapshot_states'][i],source['snapshot_metadata'][i])
 start_error=float(abs(env.get_sim_state()-source['snapshot_states'][i]).max());assert start_error==0
 frame=policy_observation(obs,prompt)
 assert np.array_equal(frame['observation/image'],source['base'][i]) and np.array_equal(frame['observation/wrist_image'],source['wrist'][i])
 assert np.array_equal(frame['observation/state'],source['state'][i])
 frames=[frame];traces=[physical_state(env,obs,names)]
 acts=[];masks=[];rewards=[];discounts=[];terminal=[];lengths=[];native=[];plans=[];replan_times=[]
 steps=0;call=0;success=False;max_source_state_error=0.;max_source_plan_error=0.;max_source_pixel_error=0
 while steps<remaining:
  if call==0:plan=candidate.copy();count=h
  else:
   plan=np.asarray(client.infer({**frames[-1],'_noise_seed':row['seed']*1000+i+call})['actions'],np.float32);count=5
  assert plan.shape==(10,7) and np.isfinite(plan).all()
  if h==5 and np.array_equal(candidate,source['actions'][i]) and i+call<len(source['rewards']):
   max_source_state_error=max(max_source_state_error,float(abs(env.get_sim_state()-source['snapshot_states'][i+call]).max()))
   max_source_plan_error=max(max_source_plan_error,float(abs(np.clip(plan[:5],-1,1)-np.clip(source['policy_action_plan'][i+call,:5],-1,1)).max()))
   max_source_pixel_error=max(max_source_pixel_error,int(abs(frames[-1]['observation/image'].astype(int)-source['base'][i+call].astype(int)).max()))
  plans.append(plan);execute=[];rew=0.;ended=False
  for j in range(min(count,remaining-steps)):
   action=np.clip(plan[j],-1,1).astype(np.float32);obs,r,done,_=env.step(action.tolist());success=bool(env.check_success())
   # Native environment reward only; compare the existing binary-success collection contract.
   if abs(float(r)-float(success))>1e-9:raise AssertionError(('native_reward_differs_from_binary_success',float(r),success))
   native.append(float(r));rew+=.9999**j*float(r);steps+=1;execute.append(action)
   traces.append(physical_state(env,obs,names));ended=success or bool(done) or steps>=remaining
   if ended:break
  k=len(execute);a=np.zeros((10,7),np.float32);a[:k]=execute
  acts.append(a);masks.append(np.arange(10)<k);rewards.append(rew);discounts.append(0. if ended else .9999**k);terminal.append(ended);lengths.append(k)
  frames.append(policy_observation(obs,prompt));replan_times.append(steps);call+=1
  if ended:break
 G=np.zeros(len(rewards),np.float32);v=0.
 for n in range(len(G)-1,-1,-1):v=rewards[n]+discounts[n]*v;G[n]=v
 executed=np.concatenate([a[m] for a,m in zip(acts,masks,strict=True)]);low,high=action_bounds()
 e=dict(base=np.stack([f['observation/image'] for f in frames]),wrist=np.stack([f['observation/wrist_image'] for f in frames]),
  state=np.stack([f['observation/state'] for f in frames]),actions=np.stack(acts),action_mask=np.stack(masks),rewards=np.asarray(rewards,np.float32),
  discounts=np.asarray(discounts,np.float32),terminal=np.asarray(terminal,bool),lengths=np.asarray(lengths,np.int32),mc_returns=G,
  prompt=np.asarray(prompt),success=np.asarray(success),gamma=np.asarray(.9999),env_step_rewards=np.asarray(native,np.float32),
  raw_env_actions=executed,normalized_actor_actions=normalized(executed,low,high),policy_action_plan=np.stack(plans),
  task_id=np.asarray(row['task_id']),task_name=np.asarray(row['task_name']),init_id=np.asarray(row['initial_state_id']),seed=np.asarray(row['seed']),
  split=np.asarray('train'),source=np.asarray('action_horizon_diagnostic'),split_hash=np.asarray(json.loads((R/'configs/splits/libero_factorial_v1.json').read_text())['sha256']),
  action_intervention_horizon=np.asarray(h),replan_env_steps=np.asarray(replan_times,np.int32),
  physical_proprio=np.stack([x[0] for x in traces]),physical_eef_pos=np.stack([x[1] for x in traces]),physical_eef_quat=np.stack([x[2] for x in traces]),
  object_positions=np.stack([x[3] for x in traces]),object_quaternions=np.stack([x[4] for x in traces]),object_body_names=np.asarray(names),
  simulator_flat_states=np.stack([x[5] for x in traces]),initial_snapshot=source['snapshot_states'][i],initial_snapshot_metadata=source['snapshot_metadata'][i],
  candidate_plan_raw=candidate,start_state_error=np.asarray(start_error))
 full,mask=behavior_sequences(e);e.update(actor_bc_executed_sequence=full,actor_bc_mask=mask,
  critic_executed_actions=e['actions'].copy(),critic_action_mask=e['action_mask'].copy(),critic_k=e['lengths'].copy())
 validate_episode(e)
 return e,dict(initial_state_error=start_error,source_state_error=max_source_state_error,source_plan_error=max_source_plan_error,source_pixel_error=max_source_pixel_error)

def branch(row):
 sid=row['state_id'];out=D/'states'/f'{sid}.json';ph=sha(D/'protocol.json')
 if out.exists():
  obj=json.loads(out.read_text());assert obj['protocol_hash']==ph;return obj
 assert sha(row['path'])==row['sha256'] and sha(row['candidate_path'])==row['candidate_sha256']
 source=load(row['path']);plans=load(row['candidate_path'])['raw'];assert plans.shape==(5,10,7)
 task=benchmark.get_benchmark_dict()['libero_10']().get_task(row['task_id'])
 env,_=_get_libero_env(task,LIBERO_ENV_RESOLUTION,row['seed']);env.seed(row['seed']);env.reset()
 names=[obj.root_body for _,obj in sorted(env.env.objects_dict.items())]
 client=WebsocketClientPolicy('127.0.0.1',PORT);records=[];audits=[];started=time.monotonic()
 try:
  for h in [5,1,3]:
   base,check=collect(env,client,source,row,plans[0],h,names)
   source_error=abs(float(base['mc_returns'][0])-float(source['mc_returns'][row['source_index']])) if h==5 else None
   if h==5:
    assert source_error<=1e-5 and check['source_state_error']<=1e-7 and check['source_pixel_error']==0 and check['source_plan_error']<=1e-6,(sid,source_error,check)
   # Every horizon's behavior is independently repeated from the exact hidden snapshot.
   repeat,recheck=collect(env,client,source,row,plans[0],h,names)
   repeat_error=abs(float(base['mc_returns'][0])-float(repeat['mc_returns'][0]))
   assert repeat_error<=1e-5 and np.array_equal(base['raw_env_actions'],repeat['raw_env_actions']) and np.array_equal(base['simulator_flat_states'],repeat['simulator_flat_states'])
   repeatpath=D/'episodes'/f'{sid}_H{h}_behavior_repeat.npz';repeat['episode_id']=np.asarray(repeatpath.stem);save_npz(repeatpath,**repeat)
   audits.append(dict(horizon=h,repeat_return_error=repeat_error,source_return_error=source_error,
     repeat_action_error=0.,repeat_sim_state_error=0.,repeat_path=str(repeatpath),**check))
   for j in range(5):
    e=base if j==0 else collect(env,client,source,row,plans[j],h,names)[0]
    eid=f'{sid}_H{h}_cf{j}';path=D/'episodes'/f'{eid}.npz';e['episode_id']=np.asarray(eid);e['candidate_index']=np.asarray(j);save_npz(path,**e)
    spec=row['candidate_specs'][j];support=spec['support'][str(h)]
    records.append(dict(state_id=sid,episode_id=eid,task_id=row['task_id'],task_name=row['task_name'],initial_state_id=row['initial_state_id'],
     source_index=row['source_index'],source_path=row['path'],horizon=h,H_QAM_alias=h==5,candidate_index=j,candidate_name=spec['name'],
     path=str(path),sha256=sha(path),future_return=float(e['mc_returns'][0]),success=bool(e['success']),steps=int(e['lengths'].sum()),
     first_k=int(e['lengths'][0]),chunk_return=float(e['rewards'][0]),initial_state_error=float(e['start_state_error']),restore_repeat_error=repeat_error,
     source_restore_error=source_error,phase=row['achieved_phase'],assigned_phase=row['assigned_phase'],branch_seed=row['seed']*1000+row['source_index'],**support))
  saved=dict(state_id=sid,records=records,audits=audits,valid=True,elapsed_seconds=time.monotonic()-started,protocol_hash=ph)
  atomic(out,saved);print('HORIZON_STATE_DONE',sid,'seconds',round(saved['elapsed_seconds'],2),flush=True);return saved
 except BaseException as exc:
  atomic(D/'invalid'/f'{sid}.json',dict(error=repr(exc),records=records,audits=audits,source=row,protocol_hash=ph));raise
 finally:env.close()

def main():
 p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=16);p.add_argument('--ports',default=','.join(str(i) for i in range(8050,8058)))
 p.add_argument('--gpus',default='0,1,2,3,4,5,6,7');p.add_argument('--limit',type=int);a=p.parse_args()
 rows=json.loads((D/'state_selection_manifest.json').read_text())['records'];rows=rows[:a.limit] if a.limit else rows
 ports=list(map(int,a.ports.split(',')));gpus=list(map(int,a.gpus.split(',')));assert len(ports)==len(gpus)
 started=time.monotonic();result=[]
 with ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(ports,gpus)) as pool:
  jobs=[pool.submit(branch,row) for row in rows]
  for job in as_completed(jobs):
   result.append(job.result());atomic(O/'collection_progress.json',dict(states=len(result),target=len(rows),branches=len(result)*15,elapsed_seconds=time.monotonic()-started))
 if a.limit:atomic(O/'preflight.json',dict(passed=all(r['valid'] for r in result),states=len(result)));return
 records=[r for s in result for r in s['records']];assert len(records)==1500
 atomic(D/'manifest.json',dict(states=100,outcomes=1500,behavior_reproduction_rollouts=300,records=sorted(records,key=lambda r:r['episode_id']),
  restore_audits=[r for s in result for r in s['audits']],elapsed_seconds=time.monotonic()-started,protocol_hash=sha(D/'protocol.json'),
  all_outcomes_retained=True,actor_updated=False,H_QAM_alias=5))
 print('HORIZON_BRANCH_COLLECTION_COMPLETE',len(records),flush=True)
if __name__=='__main__':main()
