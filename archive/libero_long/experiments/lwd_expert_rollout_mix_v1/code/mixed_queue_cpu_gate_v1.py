"""Real CPU child startup-failure cleanup and closed-prefix sampling checks."""
from mixed_common_v2 import *
import traceback,ast,unittest.mock as mock

def main():
 out={'utc':stamp(),'passed':False,'scientific_updates':0,'GPU_launches':0,'signals_sent':0}
 try:
  verify_sources();protected_audit()
  for rel,h in read(EXP/'provenance/mixed_actor_sources010.json')['files'].items():assert sha(EXP/rel)==h,rel
  prior=read(EXP/'metrics/mixed_queue_cpu_preflight017.json');assert prior['passed'] and prior['BC_failures_used']==0 and prior['native_evaluation_main_AST_identical_after_provenance_guard']
  # All training/math, replay, native inference, shard and audit source bytes are
  # exactly those admitted by017; only coordinator/reporting engineering changed.
  revision=read(EXP/'provenance/mixed_queue_engineering018.json')
  before=read(EXP/'provenance/queue_engineering018/mixed_actor_sources010.before018.json')
  after=read(EXP/'provenance/mixed_actor_sources010.json')
  changed={k for k,h in before['files'].items() if after['files'].get(k)!=h}
  assert changed=={'code/mixed_actor_pipeline_v1.py','code/mixed_results_report_v1.py'}
  import mixed_actor_pipeline_v1 as module
  q=module.Queue();q.refresh=lambda:None;count=[0];child_pids=[]
  def fake_read(p):
   p=Path(p)
   if p.name=='result.json':return {'passed':True,'all_saved_checkpoints_numerically_admitted':True,'checkpoint_gate_results':{'100':{'passed':True}},'scheduler':'cosine100'}
   if p.name=='manifest.json':return {'complete':True,'update':100,'files':{}}
   if p.name=='evaluation_protocol_original001.json':return {}
   if p.name.startswith('evaluation_'):return {'evaluation_id':'CPU018_step000100','checkpoint':str(EXP/'checkpoints/performance2/actor/CPU018/step_000100'),'checkpoint_manifest_sha256':'cpu-test','actor_updates':100,'method':'CPU018','params_variant':'params','source_reuse':'original official50 development protocol; no outcome-dependent changes','training_scheduler':'cosine100'}
   raise AssertionError(str(p))
  def launch(*args,**kwargs):
   count[0]+=1
   if count[0]==2:raise RuntimeError('CPU018 intentional second-service startup failure')
   argv=[ACTOR,'-c','import time;time.sleep(.25)'];p=subprocess.Popen(argv,cwd=ROOT,env=env(),stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
   deadline=time.monotonic()+10
   while True:
    r=identity(p.pid)
    if r['argv']==argv:break
    assert p.poll() is None and time.monotonic()<deadline;time.sleep(.01)
   r.update(name='CPU018_registered_child',physical_gpus=[],started_utc=stamp());q.jobs[p.pid]=(p,r,time.monotonic());q.state['active_jobs'].append(r);child_pids.append(p.pid);return p.pid
  q.launch=launch
  with mock.patch.object(module,'read',side_effect=fake_read),mock.patch.object(module,'sha',return_value='cpu-test'),mock.patch.object(module,'write'),mock.patch('mixed_eval_audit_v1.completed_shards',return_value={}),mock.patch.object(Path,'exists',return_value=True):
   # Avoid treating a mocked socket as real. The registered CPU child has already
   # naturally exited before cleanup; every completion still requires real waitpid.
   original=launch
   def slow_second(*a,**k):
    if count[0]==1:time.sleep(.4)
    return original(*a,**k)
   q.launch=slow_second
   try:q.evaluate([('CPU018',100,'params')]);raise AssertionError('Expected deliberate failure')
   except RuntimeError as e:assert str(e)=='CPU018 intentional second-service startup failure'
  assert len(child_pids)==1 and not q.jobs and len(q.state['finished_jobs'])==1
  assert q.state['finished_jobs'][0]['exit_code']==0 and not Path('/proc',str(child_pids[0])).exists()
  assert len(q.state['evaluation_resource_intervals'])==1
  interval=q.state['evaluation_resource_intervals'][0]
  assert interval['reserved_union_gpu_hours']==sum(r['wall_s'] for r in interval['server_intervals'])/3600
  from mixed_sampling_report_v1 import summarize
  sampling=summarize();assert len(sampling['per_task'])==20
  out.update(passed=True,registered_cpu_children=1,actual_waitpid_completions=1,remaining_registered_children=0,startup_exception_cleanup_passed=True,changed_pre_admission_files=sorted(changed),original017_preflight_sha256=sha(EXP/'metrics/mixed_queue_cpu_preflight017.json'),sourceproof_sha256=sha(EXP/'provenance/mixed_actor_sources010.json'),sample_updates={m:max(r['last_update'] for r in sampling['per_task'] if r['method']==m) for m in ['scalar','divl']})
 except Exception:out['error']=traceback.format_exc()
 write(EXP/'metrics/mixed_queue_cpu_gate018.json',out,True);print(json.dumps(out));sys.exit(0 if out['passed'] else 1)

if __name__=='__main__':main()
