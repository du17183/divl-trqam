# Public release: deployment paths anonymized; algorithm equations unchanged.
"""Sequential, fail-closed diagnostic stages; no actor optimizer is invoked."""
from datetime import datetime,timezone
import argparse
import json
from pathlib import Path
import subprocess
import sys
from run_critic_pilot_after_collection import common

ROOT=Path('/opt/divl-trqam')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    env=common();sim=env.copy()
    sim.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',
      PYTHONPATH=env['PYTHONPATH']+':'+str(ROOT/'third_party/libero'),
      LIBERO_CONFIG_PATH=str(ROOT/'configs/libero_runtime'),TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',
      MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID='7',
      OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    actor=env.copy();actor.update(CUDA_VISIBLE_DEVICES='7',JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_MEM_FRACTION='.50')
    stages=[
      ('strict_gate100','.venv_sim',sim,'run_reproduction_gate.py',['--workers','3','--port','8027']),
      ('formal_fixed50','.venv_sim',sim,'run_counterfactual_branches_v2.py',['--workers','3','--port','8027']),
      ('action_evidence','.venv_actor',actor,'analyze_counterfactuals_v2.py',[]),
      ('frozen_neighbors','.venv_actor',actor,'analyze_frozen_neighbors.py',[]),
      ('final_reports','.venv_actor',actor,'finalize_counterfactual_recovery.py',[])]
    status={'started_at_utc':datetime.now(timezone.utc).isoformat(),'completed_stages':[],'actor_updates_enabled':False}
    p=ROOT/'artifacts/counterfactual_recovery_pipeline.json'
    if args.resume:
        status=json.loads(p.read_text())
        assert status['actor_updates_enabled'] is False
        if 'failed_stage' in status:
            status.setdefault('failure_history',[]).append({k:status[k] for k in ['failed_stage','returncode','stopped_at_utc','current_log']})
        for key in ['failed_stage','returncode','stopped_at_utc']:status.pop(key,None)
        status['resumed_at_utc']=datetime.now(timezone.utc).isoformat()
    assert (ROOT/'artifacts/reproduction_gate_selection.json').exists()
    for name,venv,stageenv,script,args in stages:
        if name in status['completed_stages']:continue
        log=ROOT/'logs/stage_v1'/('recovery_v2_'+name+'.log')
        if log.exists():log=log.with_name(log.stem+'_retry'+str(len(status.get('failure_history',[])))+'.log')
        status.update(current_stage=name,current_log=str(log));p.write_text(json.dumps(status,indent=2))
        with log.open('w') as f:
            result=subprocess.run([str(ROOT/venv/'bin/python'),'-u',str(ROOT/'scripts'/script),*args],cwd=ROOT,env=stageenv,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:
            status.update(failed_stage=name,returncode=result.returncode,stopped_at_utc=datetime.now(timezone.utc).isoformat())
            p.write_text(json.dumps(status,indent=2));raise SystemExit(result.returncode)
        if name=='frozen_neighbors':
            (ROOT/'reports/nearest_future_diversity_report.md').write_text((ROOT/'reports/similar_state_future_analysis.md').read_text())
        status['completed_stages'].append(name);p.write_text(json.dumps(status,indent=2))
    status.update(current_stage='complete',completed_at_utc=datetime.now(timezone.utc).isoformat());p.write_text(json.dumps(status,indent=2))
    print('PIPELINE_COMPLETE',json.dumps(status),flush=True)

if __name__=='__main__':main()
