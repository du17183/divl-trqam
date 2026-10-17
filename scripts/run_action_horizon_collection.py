"""Eight owned frozen inference servers and16 sim workers; no training."""
import argparse,fcntl,json,os,socket,subprocess,time,shutil
from datetime import datetime,timezone
from horizon_common import R,D,O,atomic,sha
from run_critic_pilot_after_collection import common
from run_action_sensitive_v3_collection import sim_env
from run_replay_v2_pipeline import safe_stop_server
def now():return datetime.now(timezone.utc).isoformat()
def main():
 a=argparse.ArgumentParser();a.add_argument('--attempt',type=int,default=1);args=a.parse_args();suffix=f'_attempt{args.attempt}'
 O.mkdir(parents=True,exist_ok=True);lock=(O/'collection.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert json.loads((O/'audit_complete.json').read_text())['complete']
 if (O/'collection_status.json').exists():
  old=json.loads((O/'collection_status.json').read_text());assert old['stage']=='FAILED' and args.attempt>1
  atomic(O/f'collection_status_before_attempt{args.attempt}.json',old)
 state=dict(stage='setup',started_at=now(),workers=16,inference_gpus=list(range(8)),actor_updated=False,completed=[],attempt=args.attempt)
 atomic(O/f'collection_code_inventory{suffix}.json',{str(p):sha(p) for p in [R/'scripts/collect_action_horizon.py',R/'scripts/run_action_horizon_collection.py']})
 servers=[];handles=[]
 def stage(name,args):
  log=R/'logs/stage_v1'/f'action_horizon_{name}{suffix}.log';assert not log.exists()
  with log.open('w') as out:
   p=subprocess.Popen([str(R/'.venv_sim/bin/python'),'-u',str(R/'scripts/collect_action_horizon.py'),*args],cwd=R,env=sim_env(),stdout=out,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
   state.update(stage=name,pid=p.pid,log=str(log));atomic(O/'collection_status.json',state)
   while p.poll() is None:
    with (O/'runtime.jsonl').open('a') as f:f.write(json.dumps(dict(utc=now(),stage=name,load=os.getloadavg(),gpu=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,power.draw','--format=csv,noheader'],text=True,capture_output=True).stdout))+'\n')
    if shutil.disk_usage(R).free<30*1024**3:p.terminate();raise RuntimeError('Disk reserve reached')
    time.sleep(10)
   if p.returncode:raise RuntimeError(f'{name} failed: {log}')
  state['completed'].append(name);atomic(O/'collection_status.json',state)
 try:
  free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
  assert min(map(float,free.splitlines()))>65*1024
  for gpu,port in enumerate(range(8050,8058)):
   with socket.socket() as s:assert s.connect_ex(('127.0.0.1',port))!=0
   env=common();env.update(CUDA_VISIBLE_DEVICES=str(gpu),JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_MEM_FRACTION='.20')
   log=R/'logs/stage_v1'/f'action_horizon_policy_gpu{gpu}{suffix}.log';assert not log.exists();f=log.open('w');handles.append(f)
   p=subprocess.Popen([str(R/'.venv_actor/bin/python'),'-u',str(R/'scripts/serve_pilot_policy.py'),'--port',str(port)],cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
   servers.append(dict(gpu=gpu,port=port,pid=p.pid,process=p,log=str(log)))
  atomic(O/'servers_dispatch.json',[{k:v for k,v in s.items() if k!='process'} for s in servers]);deadline=time.monotonic()+900;ready=set()
  state.update(stage='loading_eight_frozen_policies');atomic(O/'collection_status.json',state)
  while len(ready)<8:
   for s in servers:
    if s['process'].poll() is not None:raise RuntimeError('Owned server failed: '+s['log'])
    try:
     with socket.create_connection(('127.0.0.1',s['port']),timeout=.2):ready.add(s['port'])
    except OSError:pass
   if time.monotonic()>deadline:raise TimeoutError('Frozen policy startup')
   state['ready_servers']=len(ready);atomic(O/'collection_status.json',state);time.sleep(2)
  stage('preflight',['--limit','1','--workers','1'])
  assert json.loads((O/'preflight.json').read_text())['passed']
  stage('branches',[])
  state.update(stage='COLLECTION_COMPLETE',finished_at=now());atomic(O/'collection_status.json',state)
  for p,h in json.loads((D/'protocol.json').read_text())['frozen_input_hashes'].items():assert sha(p)==h
 except BaseException as exc:
  state.update(stage='FAILED',error=repr(exc),failed_at=now());atomic(O/'collection_status.json',state);raise
 finally:
  for s in servers:safe_stop_server(s['pid'],s['port'])
  for f in handles:f.close()
if __name__=='__main__':main()
