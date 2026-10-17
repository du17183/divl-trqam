"""Detached, bounded ownership-aware Expert Transition audit; no model training."""
from mixed_common_v2 import *
import traceback
def main():
 lock=EXP/'provenance/expert_audit_coordinator003.lock'
 with lock.open('x') as f:f.write(str(os.getpid()))
 state={'stage':'VERIFY_FROZEN_INPUTS','started_utc':stamp(),'coordinator':identity(os.getpid()),'active_jobs':[],'finished_jobs':[x for x in read(EXP/'metrics/pipeline_state001.json')['finished_jobs'] if x.get('exit_code')==0 and x.get('kind')=='expert_audit'],'critic_updates':{},'actor_updates':{},'completed_evaluation_episodes':0}
 progress(state);jobs={}
 try:
  verify_sources();protected_audit()
  p=read(EXP/'configs/protocol001.json')['inherited']
  for name in ['sft_checkpoint','normalization','train_manifest']:
   if name=='sft_checkpoint':
    cp=Path(p[name]);m=read(cp/'manifest.json');assert sha(cp/'manifest.json')==p['sft_checkpoint_manifest_sha256'] and m['complete']
    for f,v in m['files'].items():assert sha(cp/f)==v['sha256'] and (cp/f).stat().st_size==v['bytes']
   else:assert sha(p[name])==p[name+'_sha256']
  from pilot8_critic_v1 import Replay
  replay=Replay('train',verify_files=True);assert len(replay.by_id)==240
  assert sum(x['success'] for x in replay.by_id.values())==109
  replay.close();write(EXP/'provenance/frozen_inputs_verified003.json',{'passed':True,'utc':stamp(),'protected':protected_audit(),'old_replay_episodes':240,'sft_manifest_sha256':p['sft_checkpoint_manifest_sha256']},True)
  state['stage']='EXPERT_TRANSITION_REPLAY';progress(state);pending=[]
  for tid in range(10):
   pth=EXP/'metrics/expert_audit'/f'task_{tid:02d}.json'
   if pth.exists() and read(pth).get('passed'):
    j=read(pth);assert sha(j['output'])==j['output_sha256'];continue
   assert not (EXP/'data/performance2/expert_transitions'/f'task_{tid:02d}.hdf5').exists(),'Partial expert audit output requires separate recovery; never overwrite'
   pending.append(tid)
  while pending or jobs:
   for pid,(proc,rec) in list(jobs.items()):
    code=proc.poll()
    if code is None:assert_owned(proc,rec);continue
    ended={**rec,'exit_code':code,'ended_utc':stamp()};state['finished_jobs'].append(ended);state['active_jobs'].remove(rec);del jobs[pid];progress(state)
   occupied={g for r in state['active_jobs'] for g in r['physical_gpus']}
   for gpu in range(8):
    if not pending or gpu in occupied:continue
    try:receipt=resources(state['active_jobs'],(gpu,))
    except RuntimeError:continue
    if any(x['free_mib']<65000 for x in receipt['gpus']):continue
    tid=pending.pop(0);name=f'expert_audit_task{tid:02d}_resume003';argv=[SIM,'-u',str(CODE/'mixed_expert_audit_v1.py'),'--task-id',str(tid)]
    with (EXP/'logs'/f'{name}.log').open('x') as log:proc=subprocess.Popen(argv,cwd=ROOT,env=env((gpu,),sim=True),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    deadline=time.monotonic()+10
    while True:
     rec=identity(proc.pid)
     if rec['argv']==argv:break
     assert proc.poll() is None and time.monotonic()<deadline;time.sleep(.02)
    rec.update(name=name,kind='expert_audit',physical_gpus=[gpu],task_id=tid,started_utc=stamp(),initial_resource=receipt)
    jobs[proc.pid]=(proc,rec);state['active_jobs'].append(rec);progress(state)
   time.sleep(3)
  results=[read(EXP/'metrics/expert_audit'/f'task_{t:02d}.json') for t in range(10)]
  assert all(r['passed'] for r in results),'Expert Transition audit failed; Mixed training remains barred'
  records=[e for r in results for e in r['records']];assert len(records)==50
  from pilot8_critic_v1 import Replay
  replay=Replay('train',verify_files=True);rr=[{**e,'source':'successful_rollout' if e['success'] else 'failed_rollout'} for e in replay.by_id.values()];replay.close()
  m={'schema':'mixed_expert_rollout290_v1','audit_passed':True,'complete':True,'expert_records':records,'rollout_records':rr,'episodes':290,'source_quota':None,'sampling':'original draw_indices: task balanced; trajectory uniform within task; chunk uniform within trajectory','expert_action_semantics':'full executed behavior H5; no synthetic issued plan; partial terminal H5 excluded','files':[{'hdf5':r['output'],'sha256':r['output_sha256'],'bytes':r['output_bytes']} for r in results],'audit_result_sha256':{str(t):sha(EXP/'metrics/expert_audit'/f'task_{t:02d}.json') for t in range(10)},'created_utc':stamp()}
  write(EXP/'data/performance2/mixed_manifest001.json',m,True)
  import csv
  with (EXP/'metrics/mixed_replay_manifest.csv').open('x') as f:
   w=csv.DictWriter(f,fieldnames=['episode_id','task_id','source','success','state_source_id','transitions','trajectory_path']);w.writeheader();w.writerows({k:e[k] for k in w.fieldnames} for e in records+rr)
  lines=['# 01 专家 Transition 审计','',f"审计通过：50/50，真实执行 {sum(e['executed_steps'] for e in records)} 步。",'','真实回放恢复N+1观测，逐帧核对已冻结执行前物理状态、RGB、proprio、native奖励/done；初态误差<1e-12，历史物理状态误差<1e-9；不生成奖励或动作。','', '专家H5是连续真实行为动作，Replay H5是π0.5已发出计划。专家不足5个真实动作的终止尾部排除，Critic fixed-H5接口和TD不变。','',f"排除终止尾部Chunk：{sum(e['excluded_partial_h5_chunks'] for e in records)}；真实尾部动作：{sum(e['excluded_tail_actions'] for e in records)}；原完整BC H10协议不变。",'', '|任务|专家轨迹|合法H5|native回报和|终止尾部排除|','|---|---:|---:|---:|---:|']
  for t in range(10):
   es=[e for e in records if e['task_id']==t];lines.append(f"|{t}|{len(es)}|{sum(e['transitions'] for e in es)}|{sum(e['native_reward_sum'] for e in es)}|{sum(e['excluded_partial_h5_chunks'] for e in es)}|")
  lines+=['',f"Mixed Manifest SHA256：{sha(EXP/'data/performance2/mixed_manifest001.json')}"]
  (EXP/'reports/01_expert_transition_audit.md').write_text('\n'.join(lines)+'\n')
  (EXP/'reports/02_mixed_replay_manifest.md').write_text('# 02 Mixed Replay\n\n50专家＋240 Train Rollout全部进入索引。Held-out60与评测数据排除。\n\n采样保留任务均衡→任务内轨迹均匀→轨迹内Chunk均匀，不设来源Quota。每任务5专家＋24Rollout；其专家抽样期望为5/29，但有限训练实际比例另记，不能当实测比例。\n')
  state.update(stage='EXPERT_AUDIT_PASSED_READY_FOR_MIXED_CRITIC',audit_passed=True,mixed_manifest_sha256=sha(EXP/'data/performance2/mixed_manifest001.json'));progress(state)
 except Exception:
  state.update(stage='ENGINEERING_STOP_EXPERT_AUDIT',error=traceback.format_exc());progress(state);print(state['error'],flush=True)
 finally:write(EXP/'metrics/expert_audit_coordinator_result003.json',{**state,'ended_utc':stamp(),'signals_sent':0},True)
if __name__=='__main__':main()
