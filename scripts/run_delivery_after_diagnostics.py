# Public release: deployment paths anonymized; algorithm equations unchanged.
"""Finish reports/provenance after queued real jobs; never authorize actor training."""
from pathlib import Path
import hashlib
import json
import os
import signal
import subprocess
import time

ROOT=Path('/opt/divl-trqam')

def read(path):
    for _ in range(3):
        try:return json.loads(path.read_text())
        except (FileNotFoundError,json.JSONDecodeError):time.sleep(.1)
    return None

def alive(pid):
    path=Path(f'/proc/{pid}/stat')
    return path.exists() and path.read_text().split()[2]!='Z'

def refresh_code_record():
    path=ROOT/'artifacts/code_change_manifest.json';manifest=json.loads(path.read_text())
    for row in manifest['changes']:
        row['sha256']=hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()
    manifest['updated_utc_epoch']=time.time();path.write_text(json.dumps(manifest,indent=2))
    text='# Code changes and provenance\n\nAll writes are confined to the owned /opt/divl-trqam copy. Original /home/zwl/openpi was not edited. Public checkpoint/dependency references are read-only.\n\n'
    text+='Inherited AIRBOT YAML edit was copied as-is and is not invoked by this LIBERO trainer.\n\n| File | Reason |\n|---|---|\n'
    text+=''.join('| '+row['path']+' | '+row['reason']+' |\n' for row in manifest['changes'])
    (ROOT/'reports/code_changes.md').write_text(text)
    for args,name in [(['git','diff'],'current_code_diff.patch'),(['git','status','--short'],'current_code_status.txt')]:
        result=subprocess.run(args,cwd=ROOT,capture_output=True,text=True,check=True)
        (ROOT/'reports'/name).write_text(result.stdout)

def main():
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu')
    last_progress=None
    for _ in range(720):
        diagnostic=read(ROOT/'artifacts/action_diagnostic_complete.json')
        profiles=(ROOT/'reports/critic_batch_profile.md').exists()
        progress=tuple((read(ROOT/f'artifacts/critic_{m}_progress.json') or {}).get('step',0)//100 for m in ['scalar','divl'])
        if progress!=last_progress or diagnostic:
            subprocess.run(['python3',str(ROOT/'scripts/update_stage_summary.py')],cwd=ROOT,env=env,check=True)
            last_progress=progress
        if diagnostic and profiles:break
        for dispatch,complete,log in [
            ('critic_pilot_orchestrator_dispatch.json',(ROOT/'artifacts/critic_pilot_training_complete.json').exists(),'run_critic_pilot_after_collection.log'),
            ('action_diagnostic_orchestrator_dispatch.json',bool(diagnostic),'run_action_diagnostic_after_critics.log'),
            ('critic_profile_orchestrator_dispatch.json',profiles,'run_critic_profiles.log')]:
            data=read(ROOT/'artifacts'/dispatch)
            if data and not complete and not alive(data['pid']):
                error={'stage_dispatch':dispatch,'pid':data['pid'],'log':str(ROOT/'logs/stage_v1'/log),
                    'log_tail':(ROOT/'logs/stage_v1'/log).read_text(errors='replace')[-5000:],
                    'formal_actor_training_started':False,'utc_epoch':time.time()}
                (ROOT/'artifacts/stage_runtime_error.json').write_text(json.dumps(error,indent=2))
                subprocess.run(['python3',str(ROOT/'scripts/update_stage_summary.py')],cwd=ROOT,env=env,check=True)
                with (ROOT/'reports/current_stage_summary.md').open('a') as stream:
                    stream.write('\n## Execution stopped\n\n'+dispatch+' stopped before its completion gate. Inspect '+error['log']+'. No actor training is permitted. See artifacts/stage_runtime_error.json.\n')
                raise RuntimeError(error)
        time.sleep(60)
    else:raise TimeoutError('Staged jobs did not complete within12h; no actor training permitted')
    refresh_code_record()
    subprocess.run(['python3',str(ROOT/'scripts/update_stage_summary.py')],cwd=ROOT,env=env,check=True)
    subprocess.run(['python3',str(ROOT/'scripts/finalize_stage_artifacts.py')],cwd=ROOT,env=env,check=True)
    server=read(ROOT/'artifacts/pilot_server_dispatch.json')
    if server and alive(server['pid']):
        cmd=Path(f'/proc/{server["pid"]}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        if str(ROOT/'scripts/serve_pilot_policy.py') not in cmd:raise RuntimeError('Refusing to stop an unrelated policy process')
        os.kill(server['pid'],signal.SIGTERM)
        (ROOT/'artifacts/final_policy_server_stop.json').write_text(json.dumps({'pid':server['pid'],'reason':'All inference-only diagnostics complete; free idle GPU memory','utc_epoch':time.time()}))
    (ROOT/'artifacts/stage_execution_complete.json').write_text(json.dumps({'passed_execution_and_delivery':True,
        'action_evidence_decision':diagnostic['decision'],'formal_actor_training_started':False,'utc_epoch':time.time()}))
    print('STAGE_EXECUTION_COMPLETE',diagnostic['decision'],flush=True)

if __name__=='__main__':main()
