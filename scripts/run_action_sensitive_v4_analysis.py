"""Wait for real data; preregistered probes gate matched single-GPU critics."""
import json,subprocess,time,fcntl,shutil
from action_sensitive_v4_common import R,D,O,REP,atomic,sha
from run_action_sensitive_v3_collection import sim_env,now
from run_critic_pilot_after_collection import common

def cpu():
 e=common();e.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2');return e
def gpu(index):
 e=common();e.update(CUDA_VISIBLE_DEVICES=str(index),JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_MEM_FRACTION='.40');return e
def main():
 lock=(O/'analysis.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (O/'analysis_status.json').exists()
 state=dict(stage='waiting_collection',started_at=now(),completed=[],actor_updates=False,beta_dynamic=False);atomic(O/'analysis_status.json',state)
 def reports():subprocess.run([str(R/'.venv_actor/bin/python'),str(R/'scripts/finalize_action_sensitive_v4.py')],cwd=R,env=cpu(),check=True)
 def run(name,script,args=(),env=None,venv='.venv_actor'):
  log=R/'logs/stage_v1'/f'v4_{name}.log';assert not log.exists()
  with log.open('w') as f:
   p=subprocess.Popen([str(R/venv/'bin/python'),'-u',str(R/'scripts'/script),*args],cwd=R,env=env or cpu(),stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
   state.update(stage=name,current_pid=p.pid,current_log=str(log),updated_at=now());atomic(O/'analysis_status.json',state)
   while p.poll() is None:time.sleep(10)
   if p.returncode:raise RuntimeError(f'{name} failed exit{p.returncode}; {log}')
  state['completed'].append(name);atomic(O/'analysis_status.json',state)
 def train(mix):
  assert shutil.disk_usage(R).free>145*1024**3
  jobs={};streams=[]
  try:
   for method,index in [('scalar',0),('divl',1)]:
    log=R/'logs/stage_v1'/f'v4_critic_{mix}_{method}.log';assert not log.exists();f=log.open('w');streams.append(f)
    jobs[method]=subprocess.Popen([str(R/'.venv_actor/bin/python'),'-u',str(R/'scripts/train_critic_replay_v4.py'),'--method',method,'--mix',mix,'--device-count','1'],cwd=R,env=gpu(index),stdout=f,stderr=subprocess.STDOUT)
   atomic(O/f'training_{mix}_dispatch.json',{m:dict(pid=p.pid,single_gpu=True,gpu=0 if m=='scalar' else 1,global_batch=256) for m,p in jobs.items()})
   state.update(stage=f'{mix}_matched_single_gpu_critics');atomic(O/'analysis_status.json',state);reports()
   while any(p.poll() is None for p in jobs.values()):
    for m,p in jobs.items():
     if p.poll() is not None and p.returncode:raise RuntimeError(m+' formal critic failed, stop matched partner')
    time.sleep(20)
   for step in [250,500,1000,2000]:
    xs=[json.loads((R/'checkpoints/critic_replay_v4'/mix/m/f'metrics_step{step:04d}.json').read_text()) for m in jobs]
    assert len({x['sample_order_hash'] for x in xs})==1
    for field in ['q_initial_hash','encoder_initial_hash']:assert len({x['protocol'][field] for x in xs})==1
   state['completed'].append(f'{mix}_matched_training');atomic(O/'analysis_status.json',state)
  finally:
   for p in jobs.values():
    if p.poll() is None:p.terminate()
   for p in jobs.values():
    try:p.wait(timeout=30)
    except subprocess.TimeoutExpired:pass
   for f in streams:f.close()
 try:
  while not (D/'manifest.json').exists():
   s=json.loads((O/'collection_status.json').read_text())
   if s['stage']=='FAILED':raise RuntimeError('Collection failed: '+s['error'])
   time.sleep(10)
  while json.loads((O/'collection_status.json').read_text())['stage']!='V4_COLLECTION_COMPLETE':
   s=json.loads((O/'collection_status.json').read_text())
   if s['stage']=='FAILED':raise RuntimeError('Collection finalization failed: '+s['error'])
   time.sleep(2)
  run('audit_and_representation_probes','analyze_action_sensitive_replay_v4.py')
  gate=json.loads((R/'artifacts/v4_positive_control.json').read_text())
  if not gate['passed']:
   state.update(stage=gate['decision'],finished_at=now(),reason='Fixed public primary B control gate failed; no formal Scalar/DIVL, QAM or actor')
   atomic(O/'analysis_status.json',state);reports();return
  data=json.loads((D/'manifest.json').read_text());original=json.loads((R/'data/pilot_libero_v1/manifest.json').read_text())
  for mix,ratio in [('R1',.25),('R2',.50)]:
   atomic(R/'data/replay_manifests'/f'libero_v4_{mix}.json',dict(version='action_sensitive_replay_v4',mix=mix,deployment_ratio=1-ratio,branch_ratio=ratio,
    original_manifest_sha256=sha(R/'data/pilot_libero_v1/manifest.json'),branch_manifest_sha256=sha(D/'manifest.json'),deployment_records=original['records'],branch_records=data['records'],all_outcomes_retained=True))
   train(mix)
   env=gpu(0);env['V4_EVAL_MIX']=mix
   run(f'{mix}_fixed50_evaluation','evaluate_critic_replay_v4.py',env=env)
   e=json.loads((O/f'v4_counterfactual_{mix}.json').read_text());methods=[m for m,v in e['prerequisites'].items() if v['ready_for_new_gradient']]
   if methods:
    env=sim_env();env['V4_GRADIENT_MIX']=mix
    run(f'{mix}_new_real_gradient','run_new_v4_gradient_stage.py',['--methods',','.join(methods)],env=env,venv='.venv_sim')
   reports()
  grads=[json.loads(p.read_text()) for p in O.glob('v4_new_gradient_R*.json')]
  passed=any(v['passed'] for g in grads for v in g['methods'].values())
  evs=[json.loads(p.read_text()) for p in O.glob('v4_counterfactual_R*.json')]
  criticfail=not any(v['ready_for_new_gradient'] for e in evs for v in e['prerequisites'].values())
  state.update(stage='ALLOW_FIXED_REFERENCE_QAM_PILOT' if passed else 'STOP_CRITIC_ARCHITECTURE' if criticfail else 'STOP_ACTOR_QAM',finished_at=now())
  atomic(O/'analysis_status.json',state);reports()
 except BaseException as exc:
  state.update(stage='FAILED',error=repr(exc),failed_at=now());atomic(O/'analysis_status.json',state);reports();raise
 finally:
  if (D/'protocol.json').exists():
   for p,h in json.loads((D/'protocol.json').read_text())['frozen_input_hashes'].items():assert sha(p)==h
if __name__=='__main__':main()
