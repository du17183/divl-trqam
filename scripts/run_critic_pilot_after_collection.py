# Public release: deployment paths anonymized; algorithm equations unchanged.
"""Ordered, fail-closed pilot audit and matched critic-only training jobs."""
import os
import json
from pathlib import Path
import subprocess
import time
import argparse
import hashlib

ROOT=Path('/opt/divl-trqam')

def common():
    env=os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',PYTHONPATH=str(ROOT)+':'+str(ROOT/'src')+':'+str(ROOT/'packages/openpi-client/src'),
        OPENPI_DATA_HOME=str(ROOT/'.cache/openpi'),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',
        XLA_PYTHON_CLIENT_PREALLOCATE='false',XLA_PYTHON_CLIENT_MEM_FRACTION='.90',TF_CPP_MIN_LOG_LEVEL='2',JAX_COMPILATION_CACHE_DIR=str(ROOT/'.cache/jax'))
    env['LD_LIBRARY_PATH']='/home/zwl/openpi/.cache/cube_bc_cudnn924/nvidia/cudnn/lib:/home/zwl/openpi/.venvs/openpi-b300/lib/python3.11/site-packages/nvidia/cu13/lib:/home/zwl/openpi/.venvs/openpi-b300/lib/python3.11/site-packages/nvidia/nccl/lib'
    return env

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reuse-audits',action='store_true');args=parser.parse_args()
    assert json.loads((ROOT/'artifacts/four_arm_smoke_test.json').read_text())['passed']
    collector=json.loads((ROOT/'artifacts/pilot_collector_dispatch.json').read_text())['pid']
    for _ in range(720):
        if all((ROOT/'data'/name/'manifest.json').exists() for name in ['pilot_libero_v1','dev_libero_v1']):break
        status=Path(f'/proc/{collector}/stat')
        if not status.exists() or status.read_text().split()[2]=='Z':raise RuntimeError('Collector stopped before complete manifests; no critic launched')
        time.sleep(10)
    else:raise TimeoutError('Pilot collection not complete')
    env=common();env['CUDA_VISIBLE_DEVICES']=''
    if args.reuse_audits:
        frozen=json.loads((ROOT/'artifacts/phase3_audited_manifest_hashes.json').read_text())
        for name,sha in frozen.items():
            assert hashlib.sha256((ROOT/'data'/name/'manifest.json').read_bytes()).hexdigest()==sha
        assert json.loads((ROOT/'artifacts/critic_remat_equivalence.json').read_text())['passed']
        mixed=json.loads((ROOT/'artifacts/critic_mixed_precision_audit.json').read_text())
        assert mixed['passed'] and mixed['source_sha256']==hashlib.sha256((ROOT/'src/openpi/models/divl_gemma.py').read_bytes()).hexdigest()
        assert json.loads((ROOT/'artifacts/selective_replay_audit.json').read_text())['passed']
    else:
        for script,extra in [('audit_pilot_replay.py',[]),('validate_phase0b.py',['--pilot','--dataset-dir',str(ROOT/'data/pilot_libero_v1/episodes')])]:
            subprocess.run([str(ROOT/'.venv_actor/bin/python'),'-u',str(ROOT/'scripts'/script),*extra],cwd=ROOT,env=env,check=True)
    assert json.loads((ROOT/'artifacts/pilot_replay_audit.json').read_text())['passed']
    assert json.loads((ROOT/'artifacts/beta_target_pilot_audit.json').read_text())['passed']
    jobs={};validated=set();monitored=[]
    try:
        for method,gpu in [('scalar',0),('divl',5)]:
            if (ROOT/'checkpoints/critic_pilot_v1'/method).exists():raise ValueError('Critic output already exists; explicit resume required')
            training_env=common();training_env['CUDA_VISIBLE_DEVICES']=str(gpu)
            log=(ROOT/'logs/stage_v1'/f'critic_{method}_pilot.log').open('a')
            process=subprocess.Popen([str(ROOT/'.venv_actor/bin/python'),'-u',str(ROOT/'scripts/train_libero_lwd.py'),
                '--value-type',method,'--reference-type','fixed','--steps','2000','--batch-size','256'],cwd=ROOT,env=training_env,stdout=log,stderr=subprocess.STDOUT)
            jobs[method]=process
            (ROOT/'artifacts'/f'critic_{method}_dispatch.json').write_text(json.dumps({'pid':process.pid,'physical_gpu':gpu,'actor_frozen':True,'beta_disabled':True,'batch':256,'steps':2000}))
        (ROOT/'reports/current_stage_summary.md').write_text('# Current stage\n\nPilot audit passed: 200 train episodes and 100 independent dev episodes, all 10 tasks, frozen IDs and SHA. 100 train-pilot beta label samples pass.\n\nMatched Scalar/DIVL critic-only jobs launched on physical GPUs 0 and 5, batch 256, same seed/replay/order/LR/EMA/Double-Q. Actor and beta optimizers are absent; no QAM actor training.\n\nNext: retain 250/500/1000/2000 checkpoints, profile reasonable batches, then held-out static and real counterfactual action diagnostics.\n')
        while any(p.poll() is None for p in jobs.values()):
            for method,p in jobs.items():
                if p.poll() is not None and p.returncode!=0:raise RuntimeError(f'{method} critic failed with exit {p.returncode}; other critic stopped')
            for step in [250,500,1000,2000]:
                paths=[ROOT/'checkpoints/critic_pilot_v1'/m/f'metrics_step{step:04d}.json' for m in jobs]
                if step not in validated and all(p.exists() for p in paths):
                    values=[json.loads(p.read_text()) for p in paths]
                    if len({v['sample_order_sha256'] for v in values})!=1:raise AssertionError('Scalar/DIVL sample order mismatch')
                    validated.add(step);print('MATCHED_CHECKPOINT',step,flush=True)
            sample={'utc':time.time(),'gpu':subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,power.draw','--format=csv,noheader'],capture_output=True,text=True).stdout}
            for method in jobs:
                path=ROOT/'artifacts'/f'critic_{method}_progress.json'
                if path.exists():sample[method]=json.loads(path.read_text())
            monitored.append(sample);(ROOT/'artifacts/critic_runtime.json').write_text(json.dumps(monitored,indent=2))
            time.sleep(30)
        for p in jobs.values():assert p.returncode==0
        for step in [250,500,1000,2000]:
            values=[json.loads((ROOT/'checkpoints/critic_pilot_v1'/m/f'metrics_step{step:04d}.json').read_text()) for m in jobs]
            assert len({v['sample_order_sha256'] for v in values})==1
        final=[json.loads((ROOT/'artifacts'/f'critic_{m}_fixed.json').read_text()) for m in jobs]
        assert len({v['q_initial_hash'] for v in final})==1 and len({v['encoder_initial_hash'] for v in final})==1
        assert all(not v['actor_changed'] and not v['beta_enabled'] for v in final)
        (ROOT/'artifacts/critic_pilot_training_complete.json').write_text(json.dumps({'passed':True,'methods':list(jobs),'checkpoints':[250,500,1000,2000],'initialization_and_sample_hashes_match':True,'actor_frozen':True}))
        print('CRITIC_PILOT_COMPLETE',flush=True)
    except BaseException:
        for process in jobs.values():
            if process.poll() is None:process.terminate()
        raise

if __name__=='__main__':main()
