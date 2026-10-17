# Public release: deployment paths anonymized; algorithm equations unchanged.
"""Bounded disposable batch benchmark, separate from the matched pilot."""
import json
import os
from pathlib import Path
import subprocess
import time
from run_critic_pilot_after_collection import common

ROOT=Path('/opt/divl-trqam')

def main():
    for _ in range(2880):
        if all((ROOT/'checkpoints/critic_pilot_v1'/m/'metrics_step0250.json').exists() for m in ['scalar','divl']):break
        if (ROOT/'artifacts/critic_pilot_training_complete.json').exists():break
        d=json.loads((ROOT/'artifacts/critic_pilot_orchestrator_dispatch.json').read_text());s=Path(f'/proc/{d["pid"]}/stat')
        if not s.exists() or s.read_text().split()[2]=='Z':raise RuntimeError('Critic correctness gate incomplete')
        time.sleep(30)
    else:raise TimeoutError('No matched correctness checkpoint')
    rows=[]
    for batch in [256,512,1024,2048]:
        for method in ['scalar','divl']:
            destination=ROOT/f'artifacts/critic_profile_{method}_b{batch}.json'
            if destination.exists():
                rows.append(json.loads(destination.read_text()));continue
            env=common();env['CUDA_VISIBLE_DEVICES']='6'
            log_path=ROOT/f'logs/stage_v1/critic_profile_{method}_b{batch}.log'
            log=log_path.open('a');started=time.monotonic();observations=[]
            p=subprocess.Popen([str(ROOT/'.venv_actor/bin/python'),'-u',str(ROOT/'scripts/profile_critic_batch.py'),
                                '--method',method,'--batch',str(batch)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            timed_out=False
            while p.poll() is None:
                observations.append({'elapsed':time.monotonic()-started,'gpu':subprocess.run(['nvidia-smi','-i','6',
                    '--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()})
                if time.monotonic()-started>1800:
                    timed_out=True;p.terminate()
                    try:p.wait(timeout=20)
                    except subprocess.TimeoutExpired:p.kill();p.wait()
                    break
                time.sleep(15)
            log.close()
            if destination.exists():row=json.loads(destination.read_text());row['status']='completed'
            else:
                text=log_path.read_text(errors='replace')
                row={'method':method,'batch':batch,'status':'timeout' if timed_out else 'oom' if 'RESOURCE_EXHAUSTED' in text or 'out of memory' in text.lower() else 'failed',
                     'exit_code':p.returncode,'samples_per_second':None,'log':str(log_path)}
            row.update(physical_gpu=6,elapsed_seconds=time.monotonic()-started,observations=observations)
            rows.append(row);(ROOT/'artifacts/critic_batch_benchmark.json').write_text(json.dumps(rows,indent=2))
            print('PROFILE_RESULT',json.dumps({k:v for k,v in row.items() if k!='observations'}),flush=True)
    table='| Method | Batch | Status | Samples/s | Compute seconds | Preprocess seconds |\n|---|---:|---|---:|---:|---:|\n'
    for row in rows:table+=f'| {row["method"]} | {row["batch"]} | {row.get("status","completed")} | {row.get("samples_per_second")} | {row.get("compute_seconds")} | {row.get("preprocessing_seconds")} |\n'
    (ROOT/'reports/critic_batch_profile.md').write_text('# Disposable critic batch profile\n\n'+table+
        '\nThe matched formal pilot remains batch256 for both critics. These profiles do not replace its checkpoints or sample budget. Failed/OOM attempts are retained, not interpreted as throughput measurements.\n')
    print('PROFILES_COMPLETE',flush=True)

if __name__=='__main__':main()
