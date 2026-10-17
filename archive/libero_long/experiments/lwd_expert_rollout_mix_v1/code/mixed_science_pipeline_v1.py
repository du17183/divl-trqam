"""Detached own-project queue; all completion comes from actual child waitpid."""
from mixed_common_v2 import *
import traceback

def main():
 with (EXP/'provenance/mixed_science_pipeline001.lock').open('x') as f:f.write(str(os.getpid()))
 s={'stage':'MIXED_CRITIC_RESOURCE_WAIT','started_utc':stamp(),'coordinator':identity(os.getpid()),'active_jobs':[],'finished_jobs':read(EXP/'metrics/pipeline_state001.json')['finished_jobs'],'critic_updates':{},'actor_updates':{},'completed_evaluation_episodes':0,'expert_audit_tasks_passed':10};jobs={}
 def refresh():
  for method in ['scalar','divl']:
   p=EXP/'metrics/critic'/('mixed_'+method+'_train001')/'lifecycle.json'
   if p.exists():s['critic_updates'][method]=read(p)['scientific_critic_updates']
  progress(s)
 def start(name,script,args,gpus,kind):
  receipt=resources(s['active_jobs'],gpus)
  if any(v['free_mib']<150000 for v in receipt['gpus']):return False
  argv=[ACTOR,'-u',str(CODE/script),*map(str,args)]
  with (EXP/'logs'/f'{name}.log').open('x') as f:p=subprocess.Popen(argv,cwd=ROOT,env=env(gpus),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  deadline=time.monotonic()+10
  while True:
   rec=identity(p.pid)
   if rec['argv']==argv:break
   assert p.poll() is None and time.monotonic()<deadline;time.sleep(.02)
  rec.update(name=name,kind=kind,physical_gpus=list(gpus),started_utc=stamp(),initial_resource=receipt);jobs[p.pid]=(p,rec);s['active_jobs'].append(rec);refresh();return True
 try:
  verify_sources();protected_audit()
  m=read(EXP/'data/performance2/mixed_manifest001.json');assert m['audit_passed'] and m['complete'] and m['episodes']==290
  for t in range(10):j=read(EXP/'metrics/expert_audit'/f'task_{t:02d}.json');assert j['passed'] and sha(j['output'])==j['output_sha256']
  for f,h in read(EXP/'provenance/mixed_training_sources009.json')['files'].items():assert sha(EXP/f)==h,f
  pending=[('scalar',(0,1,2,3)),('divl',(4,5,6,7))];failed=False
  while pending or jobs:
   for item in list(pending):
    method,gpus=item
    try:
     if start('mixed_'+method+'_train001','mixed_critic_train_v1.py',['--method',method],gpus,'mixed_critic'):pending.remove(item);s['stage']='MIXED_CRITICS_TRAINING'
    except RuntimeError:s['stage']='MIXED_CRITIC_RESOURCE_WAIT'
   for pid,(proc,rec) in list(jobs.items()):
    code=proc.poll()
    if code is None:assert_owned(proc,rec);continue
    s['finished_jobs'].append({**rec,'exit_code':code,'ended_utc':stamp()});s['active_jobs'].remove(rec);del jobs[pid]
    if code:failed=True;pending=[]
   refresh();time.sleep(5)
  assert not failed,'A Mixed Critic worker stopped; preserve all evidence, no automatic retraining'
  for method in ['scalar','divl']:
   j=read(EXP/'metrics/critic'/('mixed_'+method+'_train001')/'result.json');assert j['passed'] and j['scientific_critic_updates']==2000 and j['same_configuration_gate_passed']
  x=read(EXP/'metrics/critic/mixed_scalar_train001/result.json');y=read(EXP/'metrics/critic/mixed_divl_train001/result.json');assert x['coverage']['sample_sequence_sha256']==y['coverage']['sample_sequence_sha256']
  s['stage']='MIXED_CRITICS_COMPLETED_AWAITING_ACTOR_QUEUE';refresh()
  # Actor orchestration is installed separately and must carry an audited source receipt.
  while not (EXP/'provenance/mixed_actor_queue_ready010.json').exists():protected_audit();refresh();time.sleep(20)
  p=read(EXP/'provenance/mixed_actor_queue_ready010.json')
  for f,h in p['files'].items():assert sha(EXP/f)==h,f
  assert p['entrypoint']=='mixed_actor_pipeline_v1.py'
  argv=[ACTOR,'-u',str(CODE/p['entrypoint'])]
  with (EXP/'logs/mixed_actor_pipeline001.log').open('x') as f:child=subprocess.Popen(argv,cwd=ROOT,env=env(),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  time.sleep(.1);rec=identity(child.pid);assert rec['argv']==argv;s['stage']='HANDED_OFF_ACTOR_QUEUE';write(EXP/'provenance/mixed_actor_handoff010.json',rec,True)
 except Exception:s.update(stage='ENGINEERING_STOP_MIXED_PIPELINE',error=traceback.format_exc());print(s['error'],flush=True);refresh()
 finally:write(EXP/'metrics/mixed_science_pipeline_result001.json',{**s,'ended_utc':stamp(),'signals_sent':0},True)
if __name__=='__main__':main()
