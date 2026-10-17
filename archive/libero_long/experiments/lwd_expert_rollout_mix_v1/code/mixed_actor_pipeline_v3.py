"""Revision023: user-authorized GPU co-use; frozen science stays inherited."""
from mixed_actor_pipeline_v2 import *
import itertools

REPAIR = EXP/'provenance/resource_continuation023'

def public_resources(owned=(), requested=tuple(range(8))):
 """Read foreign PID/UUID/memory only; verify every protected identity."""
 from pilot8_common_v1 import UUIDS
 protected_audit()
 rows=[];processes=[];requested=set(requested)
 out=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 for line in out.splitlines():
  i,u,f,m,z=[x.strip() for x in line.split(',')];i=int(i)
  if i in requested:
   assert u==UUIDS[i]
   rows.append({'gpu':i,'uuid':u,'free_mib':float(f),'used_mib':float(m),'util_percent':float(z)})
 assert {r['gpu'] for r in rows}==requested
 out=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True)
 selected={UUIDS[g] for g in requested};known={r['pid'] for r in owned}
 for line in out.splitlines():
  pid,u,m=[x.strip() for x in line.split(',')]
  if u in selected:processes.append({'pid':int(pid),'uuid':u,'memory_mib':float(m),'registered_to_this_queue':int(pid) in known})
 return {'utc':stamp(),'gpus':rows,'processes':processes,'foreign_private_identity_reads':0,'policy':'User authorized co-use; unchanged 150000/35000 MiB launch thresholds'}

def select_gpus(script, requested, owned, resource, shared=False, python=ACTOR):
 requested=tuple(requested)
 need=150000 if script=='mixed_actor_train_v1.py' else 35000 if python==ACTOR and requested else 0
 occupied={g for r in owned for g in r['physical_gpus']}
 available={r['gpu'] for r in resource['gpus'] if r['free_mib']>=need and (shared or r['gpu'] not in occupied)}
 if set(requested)<=available:return requested
 # Four-device math/sharding/batch remain unchanged; remap only physical devices.
 if script=='mixed_actor_train_v1.py' and len(requested)==4:
  for candidate in itertools.combinations(sorted(available),4):return candidate
 return None

class CoUseQueue(ResumeQueue):
 def launch(self,script,args,gpus,name,python=ACTOR,shared=False):
  assert not (EXP/'human_hold.json').exists(),'HUMAN_HOLD'
  contract=read(REPAIR/'admission023.json')
  assert contract['user_authorized_gpu_co_use'] and contract['cpu_tests_passed']
  requested=tuple(gpus)
  scan=tuple(range(8)) if script=='mixed_actor_train_v1.py' and len(requested)==4 else requested
  r=public_resources(self.state['active_jobs'],scan)
  gpus=select_gpus(script,requested,self.state['active_jobs'],r,shared,python)
  if gpus is None:return None
  if not shared:assert not any(set(gpus)&set(rec['physical_gpus']) for rec in self.state['active_jobs'])
  for pid,(proc,rec,_) in list(self.jobs.items()):
   if not set(gpus)&set(rec['physical_gpus']) or proc.poll() is not None:continue
   assert_owned(proc,rec);s=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
   if s[0]=='Z' or int(s[6])&4:return None
  r['requested_physical_gpus']=list(requested);r['selected_physical_gpus']=list(gpus)
  r['gpus']=[x for x in r['gpus'] if x['gpu'] in gpus]
  argv=[python,'-u',str(CODE/script),*map(str,args)]
  lp=EXP/'logs'/f'{name}.log'
  if lp.exists():
   lp=EXP/'logs'/f'{name}_resource023.log'
  with lp.open('x') as f:
   p=subprocess.Popen(argv,cwd=ROOT,env=env(gpus,sim=python==SIM),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  deadline=time.monotonic()+10
  while True:
   rec=identity(p.pid)
   if rec['argv']==argv:break
   assert p.poll() is None and time.monotonic()<deadline;time.sleep(.02)
  rec.update(name=name,physical_gpus=list(gpus),log=str(lp),started_utc=stamp(),shared_gpu=shared,initial_resource=r,resource_engineering_revision=23)
  self.jobs[p.pid]=(p,rec,time.monotonic());self.state['active_jobs'].append(rec);self.refresh();return p.pid

def main():
 deadline=time.monotonic()+30
 while os.getppid()!=1 and time.monotonic()<deadline:time.sleep(.1)
 assert os.getppid()==1 and not Path('/proc/2125838').exists()
 admission=read(REPAIR/'admission023.json')
 assert admission['user_authorized_gpu_co_use'] and admission['cpu_tests_passed']
 for path,digest in admission['frozen_files'].items():assert sha(path)==digest,path
 assert sha(CODE/'mixed_actor_pipeline_v3.py')==admission['wrapper_sha256']
 assert read(REPAIR/'cpu_tests023.json')['passed']
 verify_sources();protected_audit()
 original=read(REPAIR/'terminal_state_before023.json')
 assert not original['active_jobs'] and 'HUMAN_HOLD' in original['error']
 assert read(REPAIR/'handoff_complete023.json')['old_coordinator_and_children_absent']
 with (REPAIR/'mixed_actor_pipeline023.lock').open('x') as f:f.write(str(os.getpid()))
 q=CoUseQueue();new_coordinator=q.state['coordinator'];q.state.update(original)
 q.state.update(coordinator=new_coordinator,active_jobs=[],error=None,stage='ACTOR_QUEUE_RESOURCE_CONTINUATION023',engineering_revision=23,signals_sent=0,complete=False,resumed_from_dead_coordinator=2125838)
 write(REPAIR/'coordinator_active023.json',new_coordinator,True);q.refresh()
 try:
  for methods in [('C','D'),('E','F'),('G','H')]:
   ids=[m+'-Mix100' for m in methods]
   q.train_pair(ids);q.evaluate([(x,100,'params') for x in ids]);q.report()
  q.report(True);assert read(EXP/'metrics/mixed100_report_barrier015.json')['complete']
  priority_report_addendum()
  for data in ['R0','R3']:
   for methods in [('C','D'),('E','F'),('G','H')]:
    ids=[m+'-'+data+'-Cosine500' for m in methods];q.train_pair(ids)
    for step in [100,250,500]:
     q.evaluate([(x,step,'params') for x in ids]);q.report()
  q.train_pair(['BC-Cosine500'])
  for step in [100,250,500]:
   q.evaluate([('BC-Cosine500',step,'params'),('BC-Cosine500',step,'ema_params')]);q.report()
  q.state.update(stage='ALL24000_EPISODES_COMPLETE_AWAITING_ARTIFACT_ACCEPTANCE',all_scientific_jobs_complete=True)
  q.report();q.refresh()
 except Exception:
  q.state.update(stage='ENGINEERING_STOP_ACTOR_QUEUE',error=traceback.format_exc())
  print(q.state['error'],flush=True);q.refresh()
 finally:
  write(REPAIR/'mixed_actor_pipeline_result023.json',{**q.state,'ended_utc':stamp(),'wall_s':time.monotonic()-q.started},True)

if __name__=='__main__':main()
