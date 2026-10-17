# Public release: deployment paths anonymized; algorithm equations unchanged.
"""Finish the preregistered diagnostic; never start actor training automatically."""
from pathlib import Path
import json
import hashlib
import os
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from run_critic_pilot_after_collection import common

ROOT=Path('/opt/divl-trqam')

def run(script,extra=(),gpu=None,sim=False,timeout=3600):
    env=common();env['CUDA_VISIBLE_DEVICES']='' if gpu is None else str(gpu)
    if gpu is None:env['JAX_PLATFORMS']='cpu'
    if sim:
        env.update(PYTHONPATH=env['PYTHONPATH']+':'+str(ROOT/'third_party/libero'),
                   LIBERO_CONFIG_PATH=str(ROOT/'configs/libero_runtime'),TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',
                   MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID='7',
                   OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    tag=script.removesuffix('.py')+'_'.join(str(x).replace('/','_') for x in extra)
    log=(ROOT/'logs/stage_v1'/f'{tag}.log').open('a')
    command=[str(ROOT/('.venv_sim/bin/python' if sim else '.venv_actor/bin/python')),'-u',str(ROOT/'scripts'/script),*map(str,extra)]
    process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    print('DIAGNOSTIC_START',process.pid,script,list(extra),flush=True)
    try:code=process.wait(timeout=timeout)
    except BaseException:
        process.terminate();raise
    if code:raise RuntimeError(f'{script} failed exit {code}; inspect {log.name}')
    print('DIAGNOSTIC_DONE',script,list(extra),flush=True)

def ensure_policy_server():
    # The collector's frozen server may still be running; probe its WebSocket port
    # with HTTP (426 is a valid live WebSocket endpoint) before creating any job.
    import socket
    try:
        with socket.create_connection(('127.0.0.1',8027),timeout=2):
            dispatch=json.loads((ROOT/'artifacts/pilot_server_dispatch.json').read_text())
            command=Path(f'/proc/{dispatch["pid"]}/cmdline').read_bytes().replace(b'\0',b' ').decode()
            if str(ROOT/'scripts/serve_pilot_policy.py') not in command:raise RuntimeError('Port8027 is not the owned frozen policy server')
            return
    except OSError:pass
    env=common();env.update(CUDA_VISIBLE_DEVICES='7',XLA_PYTHON_CLIENT_MEM_FRACTION='.45')
    log=(ROOT/'logs/stage_v1/pilot_policy_server.log').open('a')
    process=subprocess.Popen([str(ROOT/'.venv_actor/bin/python'),'-u',str(ROOT/'scripts/serve_pilot_policy.py'),'--port','8027'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    (ROOT/'artifacts/pilot_server_dispatch.json').write_text(json.dumps({'pid':process.pid,'gpu':7,'port':8027,'frozen':True}))
    for _ in range(180):
        if process.poll() is not None:raise RuntimeError('Frozen policy server startup failed')
        try:
            with socket.create_connection(('127.0.0.1',8027),timeout=2):return
        except OSError:time.sleep(2)
    raise TimeoutError('Frozen policy server not ready')

def static_diagnostic(method,steps,gpu):
    path=ROOT/f'artifacts/static_action_{method}_step{steps}.json'
    if path.exists():
        data=json.loads(path.read_text())
        assert data['method']==method and data['steps']==steps and data['states']==100 and data['candidate_count']==8
        assert data['diagnostic_source_sha256']==hashlib.sha256((ROOT/'scripts/prepare_action_diagnostic.py').read_bytes()).hexdigest()
        checkpoint=ROOT/f'checkpoints/critic_pilot_v1/{method}/metrics_step{steps:04d}.json'
        assert data['checkpoint_metrics_sha256']==hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        for row in data['records']:
            assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==data['output_hashes'][row['state_id']]
        print('STATIC_DIAGNOSTIC_REUSED',method,steps,flush=True)
        return
    run('prepare_action_diagnostic.py',['--method',method,'--steps',steps],gpu=gpu,timeout=3600)

def main():
    gate=ROOT/'artifacts/critic_pilot_training_complete.json'
    for _ in range(2880):
        if gate.exists():break
        dispatch=ROOT/'artifacts/critic_pilot_orchestrator_dispatch.json'
        if dispatch.exists():
            pid=json.loads(dispatch.read_text())['pid'];status=Path(f'/proc/{pid}/stat')
            if not status.exists() or status.read_text().split()[2]=='Z':
                raise RuntimeError('Critic orchestrator stopped before complete gate; no diagnostic launched')
        time.sleep(30)
    else:raise TimeoutError('Critic phase incomplete after 24h')
    assert json.loads(gate.read_text())['passed']
    assert json.loads((ROOT/'artifacts/pilot_replay_audit.json').read_text())['passed']
    ensure_policy_server()
    # Shared candidate construction is deterministic and independent of learned Q.
    # All 4 budgets are reported; the final budget was selected before any result.
    for steps in [250,500,1000,2000]:
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs=[pool.submit(static_diagnostic,method,steps,gpu)
                  for method,gpu in [('scalar',0),('divl',5)]]
            for job in jobs:job.result()
    run('update_stage_summary.py',timeout=180)
    run('run_counterfactual_branches.py',['--workers','3','--steps','2000'],gpu=7,sim=True,timeout=14400)
    run('analyze_counterfactuals.py',timeout=1200)
    run('analyze_frozen_neighbors.py',gpu=0,timeout=2400)
    run('update_stage_summary.py',timeout=180)
    decision=json.loads((ROOT/'artifacts/critic_action_metrics.json').read_text())['decision']
    (ROOT/'artifacts/action_diagnostic_complete.json').write_text(json.dumps({'passed_execution':True,'decision':decision,
        'candidate_states':50,'candidate_branches':400,'gradient_branches':600,'static_states_per_method_per_budget':100,
        'budgets':[250,500,1000,2000],'formal_actor_training_started':False}))
    print('ACTION_DIAGNOSTIC_COMPLETE',decision,flush=True)

if __name__=='__main__':main()
