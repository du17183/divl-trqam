"""Detached fixed-budget training/evaluation; actual registered waitpid only."""
from mixed_common_v2 import *
import collections,copy,traceback

class Queue:
 def __init__(self):
  self.jobs={};self.started=time.monotonic()
  self.state={'stage':'ACTOR_QUEUE_INITIALIZING','coordinator':identity(os.getpid()),'active_jobs':[],'finished_jobs':[], 'critic_updates':{'scalar':2000,'divl':2000},'actor_updates':{},'evaluations':{},'completed_evaluation_episodes':0,'expert_audit_tasks_passed':10,'started_utc':stamp(),'signals_sent':0,'complete':False}
 def refresh(self):
  for p in (EXP/'metrics/actor').glob('*/lifecycle.json'):
   j=read(p);self.state['actor_updates'][p.parent.name]=j.get('scientific_actor_updates',0)
  self.state['completed_evaluation_episodes']=sum(v.get('episodes',0) for v in self.state['evaluations'].values())
  progress(self.state)
  p=EXP/'reports/current_progress.md';text=p.read_text();rates={k:round(v['success_rate']*100,3) for k,v in self.state['evaluations'].items() if v.get('episodes')==500}
  text=text.replace('专家审计任务完成：0/10','专家审计任务完成：10/10')
  text=text.replace('最新真实成功率：新实验尚无完成500回合结果。历史九组保持原值。','最新完整500成功率：'+json.dumps(rates,ensure_ascii=False)+'。历史九组保持原值。')
  p.write_text(text)
  (ROOT/'reports/current_stage_summary.md').write_bytes((EXP/'reports/current_progress.md').read_bytes())
 def launch(self,script,args,gpus,name,python=ACTOR,shared=False):
  assert not (EXP/'human_hold.json').exists(),'HUMAN_HOLD'
  if not shared:assert not any(set(gpus)&set(r['physical_gpus']) for r in self.state['active_jobs'])
  # During a registered child's kernel teardown, wait for its real waitpid.
  for pid,(proc,rec,_) in list(self.jobs.items()):
   if not set(gpus)&set(rec['physical_gpus']):continue
   if proc.poll() is not None:continue
   assert_owned(proc,rec);s=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
   if s[0]=='Z' or int(s[6])&4:return None
  # Unknown resource owners are only enumerated by PID/UUID; never inspect private argv.
  from batch128_common_v1 import protected
  from pilot8_common_v1 import UUIDS
  known=set(protected())|set(self.jobs);requested={UUIDS[g] for g in gpus}
  for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True).splitlines():
   pid,u=[x.strip() for x in line.split(',')]
   if u in requested and int(pid) not in known:return None
  try:r=resources(self.state['active_jobs'],gpus)
  except RuntimeError:return None
  need=150000 if script=='mixed_actor_train_v1.py' else 35000 if python==ACTOR and gpus else 0
  if any(x['free_mib']<need for x in r['gpus']):return None
  argv=[python,'-u',str(CODE/script),*map(str,args)]
  lp=EXP/'logs'/f'{name}.log'
  with lp.open('x') as f:p=subprocess.Popen(argv,cwd=ROOT,env=env(gpus,sim=python==SIM),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  deadline=time.monotonic()+10
  while True:
   rec=identity(p.pid)
   if rec['argv']==argv:break
   assert p.poll() is None and time.monotonic()<deadline;time.sleep(.02)
  rec.update(name=name,physical_gpus=list(gpus),log=str(lp),started_utc=stamp(),shared_gpu=shared,initial_resource=r)
  self.jobs[p.pid]=(p,rec,time.monotonic());self.state['active_jobs'].append(rec);self.refresh();return p.pid
 def poll(self,pid):
  p,r,t=self.jobs[pid];code=p.poll()
  if code is None:assert_owned(p,r);return None
  out={**r,'exit_code':code,'ended_utc':stamp(),'wall_s':time.monotonic()-t}
  self.state['finished_jobs'].append(out);self.state['active_jobs'].remove(r);del self.jobs[pid];self.refresh();return out
 def wait_all(self,pids):
  failures=[]
  while any(p in self.jobs for p in pids):
   for pid in list(pids):
    if pid not in self.jobs:continue
    out=self.poll(pid)
    if out and out['exit_code']:failures.append(out)
   self.refresh();time.sleep(5)
  assert not failures,failures
 def train_pair(self,run_ids):
  self.state['stage']='ACTOR_TRAINING_'+','.join(run_ids);pending=list(enumerate(run_ids));pids=[]
  while pending:
   for slot,rid in list(pending):
    result=EXP/'metrics/actor'/rid/'result.json'
    if result.exists():
     j=read(result);assert j['passed'] and j['all_saved_checkpoints_numerically_admitted'];pending.remove((slot,rid));continue
    p=self.launch('mixed_actor_train_v1.py',['--run-id',rid],tuple(range(slot*4,(slot+1)*4)),rid+'_train001')
    if p is not None:pids.append(p);pending.remove((slot,rid))
   for pid in list(pids):
    if pid in self.jobs:
     done=self.poll(pid)
     if done:assert done['exit_code']==0,done
   self.refresh();time.sleep(5)
  self.wait_all(pids)
  for rid in run_ids:
   j=read(EXP/'metrics/actor'/rid/'result.json');assert j['passed'] and j['all_saved_checkpoints_numerically_admitted']
 def evaluate(self,requests):
  from mixed_eval_audit_v1 import completed_shards
  from multiprocessing.connection import Client
  from pilot8_common_v1 import AUTHKEY
  self.state['stage']='NATIVE_EVALUATING_'+','.join(x[0] for x in requests)
  protocols={};servers={};pending=collections.deque();workers={};idle=[];failed=[];t0=time.monotonic()
  try:
   for slot,(rid,n,variant) in enumerate(requests):
    result=read(EXP/'metrics/actor'/rid/'result.json');assert result['passed'] and result['all_saved_checkpoints_numerically_admitted'] and result['checkpoint_gate_results'][str(n)]['passed']
    cp=EXP/'checkpoints/performance2/actor'/rid/f'step_{n:06d}';m=read(cp/'manifest.json');assert m['complete'] and m['update']==n
    for f,v in m['files'].items():assert sha(cp/f)==v['sha256'],f
    key=f'{rid}_step{n:06d}'+('_EMA' if variant=='ema_params' else '')
    proto=copy.deepcopy(read(OLD/'configs/evaluation_protocol_original001.json'))
    proto.update(evaluation_id=key,checkpoint=str(cp),checkpoint_manifest_sha256=sha(cp/'manifest.json'),actor_updates=n,method=rid,params_variant=variant,source_reuse='original official50 development protocol; no outcome-dependent changes',training_scheduler=result['scheduler'])
    path=EXP/'configs'/f'evaluation_{key}.json'
    if path.exists():assert read(path)==proto
    else:write(path,proto,True)
    protocols[key]=path;used=completed_shards(key,path);self.state['evaluations'][key]={'episodes':10*len(used),'protocol_sha256':sha(path),'status':'IN_PROGRESS'}
    if len(used)==50:self.admit_eval(key,path,used);continue
    for shard in range(5):
     for task in range(10):
      if (task,shard) not in used:pending.append((key,task,shard))
    for g in range(slot,8,len(requests)):
     sock=f'/tmp/pi05mix_{key}_g{g}.sock';ready=EXP/'metrics/eval_ready'/f'{key}_g{g}.json';ready.parent.mkdir(exist_ok=True)
     pid=None
     while pid is None:
      pid=self.launch('mixed_eval_inference_v1.py',['--gpu',g,'--checkpoint',cp,'--checkpoint-sha',sha(cp/'manifest.json'),'--params-folder',variant,'--socket',sock,'--ready',ready],(g,),f'{key}_infer_g{g}')
      if pid is None:self.refresh();time.sleep(10)
     servers[g]=(pid,sock,ready,key);idle.append(g)
   deadline=time.monotonic()+3600
   while not all(v[2].exists() for v in servers.values()):
    for pid,_,_,_ in servers.values():
     out=self.poll(pid);assert out is None,('Inference readiness failed',out)
    assert time.monotonic()<deadline,'Inference readiness timeout';self.refresh();time.sleep(5)
   while pending or workers:
    for g in list(idle):
     key=servers[g][3];item=next((x for x in pending if x[0]==key),None)
     if item is None:continue
     _,tid,shard=item;pid=self.launch('mixed_eval_shard_v1.py',['--gpu',g,'--task-id',tid,'--shard',shard,'--protocol',protocols[key],'--protocol-sha',sha(protocols[key]),'--socket',servers[g][1]],(g,),f'{key}_task{tid:02d}_shard{shard:02d}',SIM,shared=True)
     if pid is not None:pending.remove(item);workers[pid]=(g,key,tid,shard);idle.remove(g)
    for pid,(g,key,tid,shard) in list(workers.items()):
     out=self.poll(pid)
     if out:
      del workers[pid];idle.append(g)
      if out['exit_code']:
       failed.append(out);pending=collections.deque(x for x in pending if x[0]!=key)
      else:self.state['evaluations'][key]['episodes']+=10
    # Only servers with registered live workers may enter temporary main-thread teardown.
    for pid,_,_,_ in servers.values():
     assert pid in self.jobs
     p,r,_=self.jobs[pid];assert_owned(p,r);assert p.poll() is None,'Inference service exited during evaluation'
    self.refresh();time.sleep(3)
   assert not failed,failed
   for key,path in protocols.items():self.admit_eval(key,path,completed_shards(key,path))
  finally:
   # Wait only registered simulation children, preserving every failed output.
   for pid in list(workers):
    while pid in self.jobs:
     self.poll(pid);self.refresh();time.sleep(3)
   for pid,sock,ready,key in servers.values():
    if pid not in self.jobs:continue
    proc,rec,_=self.jobs[pid]
    if proc.poll() is None:
     exact(rec)
     if Path(sock).exists():
      with Client(sock,family='AF_UNIX',authkey=AUTHKEY) as c:c.send({'command':'stop'});assert c.recv()['stopping']
    while pid in self.jobs:self.poll(pid);self.refresh();time.sleep(3)
   intervals=[{'pid':pid,'gpu':g,'started_utc':next(r for r in self.state['finished_jobs'] if r['pid']==pid)['started_utc'],'ended_utc':next(r for r in self.state['finished_jobs'] if r['pid']==pid)['ended_utc'],'wall_s':next(r for r in self.state['finished_jobs'] if r['pid']==pid)['wall_s']} for g,(pid,_,_,_) in servers.items()]
   self.state.setdefault('evaluation_resource_intervals',[]).append({'requests':requests,'gpu_count':len(servers),'batch_wall_s':time.monotonic()-t0,'server_intervals':intervals,'reserved_union_gpu_hours':sum(r['wall_s'] for r in intervals)/3600,'scope':'actual registered inference Popen lifetimes; colocated simulation counted once; GPU busy time not separately measured'})
   self.refresh()
 def admit_eval(self,key,path,used):
  assert len(used)==50 and sum(v['episodes'] for v in used.values())==500
  rows=[r for p in sorted((EXP/'evaluation'/key).glob('task_*_shard_*.json')) for r in read(p)['records']]
  receipt={'passed':True,'evaluation_id':key,'episodes':500,'successes':sum(r['success'] for r in rows),'success_rate':sum(r['success'] for r in rows)/500,'protocol_sha256':sha(path),'shards':[{'task_id':t,'shard':s,**v} for (t,s),v in used.items()],'utc':stamp(),'action_reward_hdf5_protocol_audit':True}
  p=EXP/'metrics/eval_audit'/f'{key}.json'
  if not p.exists():write(p,receipt,True)
  else:assert read(p)['protocol_sha256']==receipt['protocol_sha256'] and read(p)['shards']==receipt['shards']
  self.state['evaluations'][key]={'episodes':500,'status':'AUDITED_COMPLETE500','successes':receipt['successes'],'success_rate':receipt['success_rate'],'audit_sha256':sha(p)}
  self.refresh()
 def report(self,barrier=False):
  pid=None
  while pid is None:pid=self.launch('mixed_results_report_v1.py',['--barrier'] if barrier else [],(),f'report_'+str(len(self.state['finished_jobs'])))
  self.wait_all([pid])

def main():
 # A science parent hands off; require adoption before recording coordinator PPID.
 deadline=time.monotonic()+30
 while os.getppid()!=1 and time.monotonic()<deadline:time.sleep(.1)
 assert os.getppid()==1
 with (EXP/'provenance/mixed_actor_pipeline001.lock').open('x') as f:f.write(str(os.getpid()))
 q=Queue();write(EXP/'provenance/mixed_actor_coordinator_active010.json',q.state['coordinator'],True)
 try:
  verify_sources();protected_audit()
  for f,h in read(EXP/'provenance/mixed_actor_queue_ready010.json')['files'].items():assert sha(EXP/f)==h,f
  # Frozen diagnostics never determine hyperparameters or remove methods.
  from mixed_diagnostics_pipeline_v1 import run as diagnostics
  diagnostics(q)
  assert read(EXP/'metrics/mixed_critic_comparison_complete015.json')['passed']
  for methods in [('C','D'),('E','F'),('G','H')]:
   ids=[m+'-Mix100' for m in methods];q.train_pair(ids);q.evaluate([(x,100,'params') for x in ids]);q.report()
  q.report(True)
  assert read(EXP/'metrics/mixed100_report_barrier015.json')['complete']
  # Independent new SFT initializations and cosine500 trajectories.
  for data in ['R0','R3']:
   for methods in [('C','D'),('E','F'),('G','H')]:
    ids=[m+'-'+data+'-Cosine500' for m in methods];q.train_pair(ids)
    for n in [100,250,500]:q.evaluate([(x,n,'params') for x in ids]);q.report()
  q.train_pair(['BC-Cosine500'])
  for n in [100,250,500]:q.evaluate([('BC-Cosine500',n,'params'),('BC-Cosine500',n,'ema_params')]);q.report()
  q.state['stage']='ALL24000_EPISODES_COMPLETE_AWAITING_ARTIFACT_ACCEPTANCE';q.state['all_scientific_jobs_complete']=True;q.report();q.refresh()
 except Exception:q.state.update(stage='ENGINEERING_STOP_ACTOR_QUEUE',error=traceback.format_exc());print(q.state['error'],flush=True);q.refresh()
 finally:write(EXP/'metrics/mixed_actor_pipeline_result001.json',{**q.state,'ended_utc':stamp(),'wall_s':time.monotonic()-q.started},True)
if __name__=='__main__':main()
