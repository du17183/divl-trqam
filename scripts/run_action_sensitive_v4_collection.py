"""Eight owned frozen servers; independent sims; no actor updates."""
import json,fcntl,socket,subprocess,time,shutil
from action_sensitive_v4_common import R,D,O,REP,atomic,sha
from run_critic_pilot_after_collection import common
from run_action_sensitive_v3_collection import sim_env,now
from run_replay_v2_pipeline import safe_stop_server

def main():
 O.mkdir(parents=True,exist_ok=True);lock=(O/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (O/'collection_status.json').exists()
 assert (D/'protocol.json').exists()
 state=dict(stage='setup',started_at=now(),completed=[],actor_updates=False,gpus=list(range(8)),workers=16)
 atomic(O/'collection_status.json',state);servers=[];streams=[]
 def stage(name,args):
  log=R/'logs/stage_v1'/f'v4_{name}.log';assert not log.exists()
  with log.open('w') as f:
   child=subprocess.Popen([str(R/'.venv_sim/bin/python'),'-u',str(R/'scripts/collect_action_sensitive_replay_v4.py'),*args],cwd=R,env=sim_env(),stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
   state.update(stage=name,pid=child.pid,log=str(log),updated_at=now());atomic(O/'collection_status.json',state)
   while child.poll() is None:
    with (O/'runtime.jsonl').open('a') as rt:
     rt.write(json.dumps(dict(utc=now(),stage=name,load=os_load(),gpu=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,power.draw','--format=csv,noheader'],text=True,capture_output=True).stdout))+'\n')
    if shutil.disk_usage(R).free<25*1024**3:child.terminate();raise RuntimeError('Disk reserve')
    time.sleep(10)
   if child.returncode:raise RuntimeError(f'{name} exited{child.returncode}; {log}')
  state['completed'].append(name);atomic(O/'collection_status.json',state)
 try:
  free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
  assert min(float(v) for v in free.splitlines())>65*1024
  for gpu,port in enumerate(range(8040,8048)):
   with socket.socket() as s:assert s.connect_ex(('127.0.0.1',port))!=0
   env=common();env.update(CUDA_VISIBLE_DEVICES=str(gpu),JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_MEM_FRACTION='.20')
   log=R/'logs/stage_v1'/f'v4_frozen_policy_gpu{gpu}.log';assert not log.exists();f=log.open('w');streams.append(f)
   p=subprocess.Popen([str(R/'.venv_actor/bin/python'),'-u',str(R/'scripts/serve_pilot_policy.py'),'--port',str(port)],cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
   servers.append(dict(gpu=gpu,port=port,pid=p.pid,process=p,log=str(log)))
  atomic(O/'servers_dispatch.json',[{k:v for k,v in s.items() if k!='process'} for s in servers])
  state.update(stage='loading_eight_frozen_policies');atomic(O/'collection_status.json',state);ready=set();deadline=time.monotonic()+900
  while len(ready)<8:
   for s in servers:
    if s['process'].poll() is not None:raise RuntimeError('Owned frozen server exited: '+s['log'])
    if s['port'] in ready:continue
    try:
     with socket.create_connection(('127.0.0.1',s['port']),timeout=.2):ready.add(s['port'])
    except OSError:pass
   if time.monotonic()>deadline:raise TimeoutError('Frozen server load')
   state['ready_servers']=len(ready);atomic(O/'collection_status.json',state);time.sleep(2)
  stage('branch_preflight',['--limit','1']);assert json.loads((O/'branches_preflight.json').read_text())['passed']
  stage('same_state_branches',[])
  for s in servers:safe_stop_server(s['pid'],s['port'])
  state.update(stage='V4_COLLECTION_COMPLETE',finished_at=now());atomic(O/'collection_status.json',state)
  for p,h in json.loads((D/'protocol.json').read_text())['frozen_input_hashes'].items():assert sha(p)==h
 except BaseException as exc:
  state.update(stage='FAILED',error=repr(exc),failed_at=now());atomic(O/'collection_status.json',state);raise
 finally:
  for s in servers:safe_stop_server(s['pid'],s['port'])
  for stream in streams:stream.close()

def os_load():
 import os
 return os.getloadavg()
if __name__=='__main__':main()
