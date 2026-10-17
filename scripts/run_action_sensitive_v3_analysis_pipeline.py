"""Data/control gates precede two matched critic mixes and fresh real gradients."""
import fcntl,json,os,shutil,subprocess,time
from action_sensitive_v3_common import R,D,O,atomic,sha
from run_critic_pilot_after_collection import common
from run_action_sensitive_v3_collection import sim_env,now


def cpu_env():
    e=common();e.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2');return e


def reports():
    subprocess.run([str(R/'.venv_actor/bin/python'),str(R/'scripts/finalize_action_sensitive_v3.py')],cwd=R,env=cpu_env(),check=True)


def stage(state,name,script,args=(),env=None,venv='.venv_actor'):
    log=R/'logs/stage_v1'/f'v3_{name}.log';assert not log.exists()
    with log.open('w') as stream:
        p=subprocess.Popen([str(R/venv/'bin/python'),'-u',str(R/'scripts'/script),*args],cwd=R,env=env or cpu_env(),stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        state.update(stage=name,current_pid=p.pid,current_log=str(log),updated_at=now());atomic(O/'analysis_pipeline_status.json',state);reports()
        while p.poll() is None:time.sleep(10)
        if p.returncode:raise RuntimeError(f'{name} exit{p.returncode}; {log}')
    state['completed'].append(name);atomic(O/'analysis_pipeline_status.json',state)


def paired(state,mix):
    assert shutil.disk_usage(R).free>145*1024**3,'Reserve128GB for both complete checkpoint series'
    memory=subprocess.run(['nvidia-smi','--query-gpu=memory.free,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout
    for row in memory.splitlines():
        free,total=[float(x) for x in row.split(',')];assert free>.55*total+4096
    jobs={};streams=[]
    try:
        for method,gpus in [('scalar','0,1,2,3'),('divl','4,5,6,7')]:
            log=R/'logs/stage_v1'/f'v3_critic_{mix}_{method}.log';assert not log.exists();stream=log.open('w');streams.append(stream)
            env=common();env.update(CUDA_VISIBLE_DEVICES=gpus,JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_MEM_FRACTION='.55')
            jobs[method]=subprocess.Popen([str(R/'.venv_actor/bin/python'),'-u',str(R/'scripts/train_critic_replay_v3.py'),'--method',method,'--mix',mix,'--device-count','4'],cwd=R,env=env,stdout=stream,stderr=subprocess.STDOUT)
        atomic(O/f'v3_training_{mix}_dispatch.json',{m:dict(pid=p.pid,gpus=list(range(4)) if m=='scalar' else list(range(4,8)),global_batch=256,microbatch=64) for m,p in jobs.items()})
        state.update(stage=f'{mix}_paired_critic_training',updated_at=now());atomic(O/'analysis_pipeline_status.json',state)
        while any(p.poll() is None for p in jobs.values()):
            for m,p in jobs.items():
                if p.poll() is not None and p.returncode:raise RuntimeError(m+' critic failed; stop only matched partner')
            with (O/'critic_runtime.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(utc=now(),mix=mix,gpu=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,power.draw','--format=csv,noheader'],capture_output=True,text=True).stdout))+'\n')
            reports();time.sleep(20)
        completed=[json.loads((O/f'critic_{mix}_{m}_v3_complete.json').read_text()) for m in jobs]
        assert all(x['passed'] and x['steps']==2000 for x in completed)
        for step in [250,500,1000,2000]:
            values=[json.loads((R/'checkpoints/critic_replay_v3'/mix/m/f'metrics_step{step:04d}.json').read_text()) for m in jobs]
            assert len({x['sample_order_hash'] for x in values})==1
            for field in ['q_initial_hash','encoder_initial_hash']:assert len({x['protocol'][field] for x in values})==1
        atomic(O/f'matched_v3_{mix}_training_complete.json',dict(passed=True,checkpoints=[250,500,1000,2000],same_initialization=True,same_sample_order=True,actor_frozen=True))
        state['completed'].append(f'{mix}_paired_critic_training');atomic(O/'analysis_pipeline_status.json',state)
    finally:
        for p in jobs.values():
            if p.poll() is None:p.terminate()
        for p in jobs.values():
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:pass
        for stream in streams:stream.close()


def main():
    lock=(O/'analysis_pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (O/'analysis_pipeline_status.json').exists()
    state=dict(stage='data_audit',started_at=now(),completed=[],actor_updates=False,beta_dynamic=False)
    atomic(O/'analysis_pipeline_status.json',state)
    try:
        assert (D/'manifest.json').exists()
        env=common();env.update(CUDA_VISIBLE_DEVICES='7',JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_MEM_FRACTION='.20')
        stage(state,'public_feature_extraction','extract_v3_frozen_features.py',env=env)
        stage(state,'supervision_and_positive_control','analyze_action_sensitive_replay_v3.py')
        gate=json.loads((O/'positive_control_gate.json').read_text())
        data=json.loads((D/'manifest.json').read_text())
        if not gate['passed'] or data['states']<200:
            state.update(stage='STOP_DATA_REDESIGN',reason='Preregistered v3 learnability gate failed or fewer200validstates; no Scalar/DIVL retraining, no additional data/updates',finished_at=now())
            atomic(O/'analysis_pipeline_status.json',state);reports();return
        original=json.loads((R/'data/pilot_libero_v1/manifest.json').read_text())
        for mix,ratio in [('R1',.25),('R2',.5)]:
            manifest=dict(version='libero_action_sensitive_replay_v3',mix=mix,
                deployment_ratio=1-ratio,branch_ratio=ratio,
                original_manifest_sha256=sha(R/'data/pilot_libero_v1/manifest.json'),branch_manifest_sha256=sha(D/'manifest.json'),
                deployment_records=original['records'],branch_records=data['records'],
                sampling='deploymentuniformchunks; branchuniformstate-actionanchors; all outcomes retained; ratios exact perglobalbatch256',
                actor_frozen=True,primary_budget=2000,checkpoints=[250,500,1000,2000])
            atomic(R/'data/replay_manifests'/f'libero_v3_{mix}.json',manifest)
            paired(state,mix)
            stage(state,f'{mix}_frozen50_evaluation','evaluate_critic_replay_v3_parallel.py',['--mix',mix])
        for mix in ['R1','R2']:
            e=json.loads((O/f'v3_counterfactual_{mix}.json').read_text())
            eligible=[m for m,v in e['prerequisites'].items() if v['ready_for_new_gradient']]
            if eligible:
                env=sim_env();env['V3_GRADIENT_MIX']=mix
                stage(state,f'{mix}_fresh_real_gradient','run_new_v3_gradient_stage.py',['--methods',','.join(eligible)],env=env,venv='.venv_sim')
        passed=[]
        for mix in ['R1','R2']:
            p=O/f'v3_new_gradient_{mix}.json'
            if p.exists():
                g=json.loads(p.read_text());passed.extend((mix,m) for m,v in g['methods'].items() if v['passed'])
        # The opening instruction explicitly says noActor thisround. Record
        # any actual eligibility, without silently launching an Actor update.
        state.update(stage='READY_FOR_FIXED_QAM_PILOT' if passed else 'STOP_ACTOR_QAM',qualified_methods=passed,
                     reason='All planned v3criticmixes evaluated; Actor remains frozen percurrentround instruction',finished_at=now())
        atomic(O/'analysis_pipeline_status.json',state)
        for path,h in json.loads((O/'frozen_input_hashes.json').read_text()).items():assert sha(path)==h
        reports();print('V3_ANALYSIS_PIPELINE_COMPLETE',json.dumps(state),flush=True)
    except BaseException as exc:
        state.update(stage='FAILED',error=repr(exc),failed_at=now());atomic(O/'analysis_pipeline_status.json',state)
        reports();raise


if __name__=='__main__':main()
