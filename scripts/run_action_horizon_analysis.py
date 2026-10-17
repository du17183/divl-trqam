"""CPU diagnostics then read-only frozen target evaluators on GPUs0/1."""
import json,os,subprocess,time
from datetime import datetime,timezone
from horizon_common import R,D,O,atomic
from run_critic_pilot_after_collection import common
def now():return datetime.now(timezone.utc).isoformat()
def main():
 status=dict(stage='CPU_analysis',started_at=now(),actor_updated=False,formal_training=False);atomic(O/'analysis_status.json',status)
 env=common();env.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 def run(script,tag,env,args=()):
  f=(R/'logs/stage_v1'/f'action_horizon_{tag}.log').open('w');p=subprocess.Popen([str(R/'.venv_actor/bin/python'),'-u',str(R/'scripts'/script),*args],cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL);return p,f
 p,f=run('analyze_action_horizon.py','analysis',env);ret=p.wait();f.close();assert ret==0,'CPU diagnostic failed'
 jobs=[]
 for method,gpu in [('scalar',0),('divl',1)]:
  ve=common();ve.update(CUDA_VISIBLE_DEVICES=str(gpu),JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_MEM_FRACTION='.25')
  p,f=run('evaluate_horizon_targets.py',f'target_{method}',ve,['--method',method]);jobs.append((method,p,f))
 status.update(stage='frozen_EMA_target_audit',pids={m:p.pid for m,p,_ in jobs});atomic(O/'analysis_status.json',status)
 for method,p,f in jobs:
  ret=p.wait();f.close();assert ret==0,f'{method} target audit failed'
 p,f=run('finalize_action_horizon.py','finalize',env);ret=p.wait();f.close();assert ret==0,'Final report failed'
 status.update(stage=json.loads((O/'round_complete.json').read_text())['decision'],finished_at=now());atomic(O/'analysis_status.json',status)
if __name__=='__main__':main()
