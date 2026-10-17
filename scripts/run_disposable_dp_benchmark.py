# Public release: deployment paths anonymized; algorithm equations unchanged.
from pathlib import Path
import json,os,signal,subprocess,sys,time
R=Path('/opt/divl-trqam');O=R/'artifacts/replay_action_audit_v3'
sys.path.insert(0,str(R/'scripts'));from run_critic_pilot_after_collection import common
jobs={};combined={m:dict(actor_updated=False,disposable=True,results=[]) for m in ['scalar','divl']}
try:
 for count in [1,2,4]:
  jobs={};handles=[]
  for method,gpus in [('scalar','1,0,2,3'),('divl','7,6,5,4')]:
   chosen=','.join(gpus.split(',')[:count]);env=common();env.update(CUDA_VISIBLE_DEVICES=chosen,JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_MEM_FRACTION='.50')
   tag=f'{method}_n{count}';f=(R/'logs/stage_v1'/f'critic_dp_benchmark_{tag}.log').open('w');handles.append(f)
   p=subprocess.Popen([str(R/'.venv_actor/bin/python'),'-u',str(R/'scripts/benchmark_critic_data_parallel.py'),'--method',method,'--tag',tag,'--counts',str(count),'--repeats','6'],cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT)
   jobs[method]=p
  (O/'critic_dp_benchmark_dispatch.json').write_text(json.dumps(dict(phase=count,pids={m:p.pid for m,p in jobs.items()},groups={'scalar':'1,0,2,3','divl':'7,6,5,4'})))
  for method,p in jobs.items():
   rc=p.wait(timeout=1200);print(method,'GPUS',count,'EXIT',rc,flush=True)
   if rc:raise RuntimeError(f'{method} n{count} failed')
   result=json.loads((O/f'critic_dp_benchmark_{method}_n{count}.json').read_text());combined[method]['results'].extend(result['results'])
   (O/f'critic_dp_benchmark_{method}.json').write_text(json.dumps(combined[method],indent=2))
  for f in handles:f.close()
finally:
 for p in jobs.values():
  if p.poll() is None:p.terminate()
 for pid,name in [(3746679,'train_critic_replay_v2.py'),(3746680,'train_critic_replay_v2.py'),(3348531,'run_replay_v2_pipeline.py')]:
  p=Path(f'/proc/{pid}')
  if p.exists() and str(R/'scripts'/name) in p.joinpath('cmdline').read_bytes().decode().replace('\0',' '):os.kill(pid,signal.SIGCONT)
 (O/'critic_dp_benchmark_status.json').write_text(json.dumps(dict(finished=True,old_jobs_resumed=True,finished_unix=time.time())))
