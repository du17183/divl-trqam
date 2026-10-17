"""Resume only six BC evaluations; reuse frozen evaluation/report functions."""
from mixed_resume_guard041 import install_guard,audit_historical_protected,PROOF
install_guard()
import mixed_actor_pipeline_v3 as original
from mixed_common_v2 import *
import copy,traceback

class EvalOnlyQueue(original.CoUseQueue):
 def train_pair(self,*args,**kwargs):raise AssertionError('Resume041 forbids all training')
 def launch(self,script,args,gpus,name,python=ACTOR,shared=False):
  mapping={'mixed_eval_inference_v1.py':'mixed_resume_eval_inference041.py','mixed_eval_shard_v1.py':'mixed_resume_eval_shard041.py','mixed_results_report_v1.py':'mixed_resume_report041.py'}
  assert script in mapping,('Resume041 forbids this executable',script)
  return super().launch(mapping[script],args,gpus,name+'_resume041',python,shared)

def main():
 deadline=time.monotonic()+30
 while os.getppid()!=1 and time.monotonic()<deadline:time.sleep(.1)
 assert os.getppid()==1
 pre=read(PROOF/'preflight_readonly.json');assert pre['passed']
 admission=read(PROOF/'admission041.json');assert admission['cpu_tests_passed']
 assert not (EXP/'human_hold.json').exists()
 original.protected_audit=audit_historical_protected
 verify_sources();audit_historical_protected()
 with (PROOF/'coordinator041.lock').open('x') as f:f.write(str(os.getpid()))
 old=read(PROOF/'pipeline_state_before041.json')
 assert not old['active_jobs']
 complete={k:v for k,v in old['evaluations'].items() if v.get('status')=='AUDITED_COMPLETE500'}
 assert complete==pre['audited500'] and len(complete)==42
 q=EvalOnlyQueue();current=q.state['coordinator'];q.state.update(copy.deepcopy(old))
 q.state.update(coordinator=current,active_jobs=[],error=None,complete=False,all_scientific_jobs_complete=False,stage='RESUMED_BC_EVALUATION_ONLY041',engineering_revision=41,resumed_from_dead_coordinator=751318,resume041_scope='BC params/EMA CP100/250/500 only; no training or completed-episode reruns')
 write(PROOF/'coordinator_active041.json',current,True);q.refresh()
 try:
  for n in [100,250,500]:
   q.evaluate([('BC-Cosine500',n,'params'),('BC-Cosine500',n,'ema_params')]);q.report()
   assert all(q.state['evaluations'][k]==v for k,v in complete.items())
  assert sum(v['episodes'] for v in q.state['evaluations'].values())==24000
  assert len([v for v in q.state['evaluations'].values() if v['status']=='AUDITED_COMPLETE500'])==48
  q.state.update(stage='ALL24000_EPISODES_COMPLETE_AWAITING_ARTIFACT_ACCEPTANCE',all_scientific_jobs_complete=True)
  q.report();q.refresh()
 except Exception:
  q.state.update(stage='ENGINEERING_STOP_BC_EVALUATION041',error=traceback.format_exc());print(q.state['error'],flush=True);q.refresh()
 finally:
  write(PROOF/'mixed_bc_eval_pipeline_result041.json',{**q.state,'ended_utc':stamp(),'wall_s':time.monotonic()-q.started},True)

if __name__=='__main__':main()
