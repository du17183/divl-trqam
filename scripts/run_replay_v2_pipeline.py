# Public release: deployment paths anonymized; algorithm equations unchanged.
"""A bounded train-only replay repair, matched critics, then gated original QAM."""
from pathlib import Path
from datetime import datetime, timezone
import fcntl, json, os, signal, subprocess, time, hashlib, argparse
from run_critic_pilot_after_collection import common

R=Path('/opt/divl-trqam');O=R/'artifacts/replay_action_audit_v3'
STATUS=O/'pipeline_status.json'

def now():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(path.read_text())
def atomic(path,value):
    tmp=path.with_suffix('.partial');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
def alive(pid):
    p=Path(f'/proc/{pid}/stat');return p.exists() and p.read_text().split()[2]!='Z'
def safe_stop_server(pid,port):
    if not alive(pid):return
    proc=Path(f'/proc/{pid}')
    cmd=proc.joinpath('cmdline').read_bytes().split(b'\0')
    args=[a.decode() for a in cmd if a]
    assert str(R/'scripts/serve_pilot_policy.py') in args and '--port' in args and args[args.index('--port')+1]==str(port),args
    assert proc.joinpath('cwd').resolve()==R.resolve()
    os.kill(pid,signal.SIGTERM)
    for _ in range(20):
        if not alive(pid):return
        time.sleep(.5)
    raise RuntimeError('Our frozen policy server did not stop; no force-killing shared tasks')

def gpu_env(gpu,fraction='.55'):
    e=common();e.update(CUDA_VISIBLE_DEVICES=str(gpu),JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_MEM_FRACTION=fraction)
    return e
def cpu_env():
    e=common();e.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2');return e
def sim_env():
    e=cpu_env();e.update(PYTHONPATH=e['PYTHONPATH']+':'+str(R/'third_party/libero'),LIBERO_CONFIG_PATH=str(R/'configs/libero_runtime'),TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID='7');return e

def run_stage(state,name,script,env,args=(),venv='.venv_actor'):
    log=R/'logs/stage_v1'/f'replay_v2_{name}.log'
    assert not log.exists(),'No silent stage overwrite: '+str(log)
    with log.open('w') as f:
        p=subprocess.Popen([str(R/venv/'bin/python'),'-u',str(R/'scripts'/script),*args],cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT)
        state.update(stage=name,current_pid=p.pid,current_log=str(log),updated_at=now());atomic(STATUS,state)
        rc=p.wait()
    if rc:raise RuntimeError(f'{name} failed, exit {rc}; inspect {log}')
    state['completed'].append(name);state['updated_at']=now();atomic(STATUS,state)

def paired_critics(state,scalar_gpus='1',divl_gpus='7'):
    jobs={};logs={};validated=set();runtime=[]
    allocations={'scalar':scalar_gpus,'divl':divl_gpus}
    cards={m:[int(x) for x in g.split(',')] for m,g in allocations.items()}
    assert len(cards['scalar'])==len(cards['divl']) and not set(cards['scalar'])&set(cards['divl'])
    usage=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free,memory.total','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True).stdout
    free={int(v.split(',')[0]):[float(x) for x in v.split(',')[1:]] for v in usage.splitlines()}
    for gpu in cards['scalar']+cards['divl']:assert free[gpu][0]>.55*free[gpu][1]+4096,('Insufficient spare GPU memory',gpu,free[gpu])
    try:
        for method,gpu in allocations.items():
            assert not (R/'checkpoints/critic_replay_v2'/method).exists(),'Fresh checkpoint directory must be absent'
            log=R/'logs/stage_v1'/f'critic_{method}_v2.log';assert not log.exists()
            logs[method]=log.open('w')
            jobs[method]=subprocess.Popen([str(R/'.venv_actor/bin/python'),'-u',str(R/'scripts/train_critic_replay_v2.py'),'--method',method,'--device-count',str(len(cards[method]))],cwd=R,env=gpu_env(gpu),stdout=logs[method],stderr=subprocess.STDOUT)
        state.update(stage='paired_critic_training',jobs={m:dict(pid=p.pid,physical_gpus=cards[m],data_parallel_devices=len(cards[m]),global_batch=256,log=str(R/'logs/stage_v1'/f'critic_{m}_v2.log')) for m,p in jobs.items()},updated_at=now());atomic(STATUS,state)
        while any(p.poll() is None for p in jobs.values()):
            for m,p in jobs.items():
                if p.poll() is not None and p.returncode:raise RuntimeError(f'{m} critic failed, exit{p.returncode}; matched partner stopped')
            for step in [250,500,1000,2000]:
                files=[R/'checkpoints/critic_replay_v2'/m/f'metrics_step{step:04d}.json' for m in jobs]
                if step not in validated and all(p.exists() for p in files):
                    try:values=[read(p) for p in files]
                    except json.JSONDecodeError:continue
                    assert len({v['sample_order_hash'] for v in values})==1,'Matched sample order differs'
                    for field in ['q_initial_hash','encoder_initial_hash']:
                        assert len({v['protocol'][field] for v in values})==1,field
                    validated.add(step);print('MATCHED_CHECKPOINT',step,flush=True)
            observation=dict(utc=now(),gpu=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,power.draw','--format=csv,noheader'],capture_output=True,text=True).stdout)
            for m in jobs:
                try:observation[m]=read(O/f'critic_{m}_v2_progress.json')
                except (FileNotFoundError,json.JSONDecodeError):pass
            runtime.append(observation);atomic(O/'v2_training_runtime.json',runtime);time.sleep(20)
        values=[read(O/f'critic_{m}_v2_complete.json') for m in jobs]
        assert all(v['passed'] and v['steps']==2000 for v in values)
        assert len({v['sample_order_hash'] for v in values})==1
        for field in ['q_initial_hash','encoder_initial_hash']:
            assert len({v['protocol'][field] for v in values})==1
        atomic(O/'matched_v2_training_complete.json',dict(passed=True,checkpoints=[250,500,1000,2000],same_initial_Q_encoder_and_sample_order=True,actor_frozen=True))
        state['completed'].append('paired_critic_training');atomic(STATUS,state)
    finally:
        for p in jobs.values():
            if p.poll() is None:p.terminate()
        for p in jobs.values():
            try:p.wait(timeout=20)
            except subprocess.TimeoutExpired:pass
        for f in logs.values():f.close()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reuse-completed-data-stage',action='store_true')
    parser.add_argument('--scalar-gpus',default='1');parser.add_argument('--divl-gpus',default='7');args=parser.parse_args()
    lock=(O/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not STATUS.exists(),'No second coordinator without explicit recovery'
    state=dict(started_at=now(),stage='waiting_for_train_only_collection',completed=[],actor_updates=False,beta_dynamic=False)
    atomic(STATUS,state);collector=read(O/'full_collection_dispatch.json')['pid'];deadline=time.monotonic()+10800
    servers=[(2874063,8027),(2942112,8028),(2942115,8029)]
    try:
        if not args.reuse_completed_data_stage:
            while not (R/'data/action_sensitive_replay_v2/manifest.json').exists():
                if not alive(collector):raise RuntimeError('Collector stopped without its full manifest; training blocked')
                if time.monotonic()>deadline:raise TimeoutError('Collection deadline exceeded; no additional outcomes auto-created')
                time.sleep(20)
            while alive(collector):time.sleep(2)
        data=read(R/'data/action_sensitive_replay_v2/manifest.json');assert data['states']==1000 and data['outcomes']==5000 and not data['dev_states_used']
        if args.reuse_completed_data_stage:
            assert read(O/'replay_v2_data_gate.json')['passed']
            assert (O/'replay_v2_positive_control.json').exists()
            state['completed']=['train_only_collection','dataset_gate_and_learnability']
            state['execution_recovery']='Reuse immutable completed data/audits; matched fresh training using throughput-selected real data parallelism'
            atomic(STATUS,state)
        else:
            for pid,port in servers:safe_stop_server(pid,port)
            state['completed'].append('train_only_collection');state['stopped_own_policy_servers']=[pid for pid,_ in servers];atomic(STATUS,state)
            run_stage(state,'dataset_gate_and_learnability','evaluate_action_sensitive_replay_v2.py',cpu_env())
        if not read(O/'replay_v2_data_gate.json')['passed']:
            state.update(stage='STOP_DATA_REDESIGN',decision='STOP_DATA_REDESIGN',reason='Preregistered same-state data diversity/reproduction gate failed');atomic(STATUS,state)
            run_stage(state,'reports','finalize_replay_v2_stage.py',cpu_env())
            state.update(stage=state['decision'],finished_at=now());atomic(STATUS,state);return
        # The mean-pooled frozen-vision MLP is a diagnostic, not a substitute for
        # the trainable token-level encoder. Permit one fixed2000-update repair;
        # if both controls and critics fail, stop expansion instead of tuning DIVL.
        paired_critics(state,args.scalar_gpus,args.divl_gpus)
        run_stage(state,'frozen50_evaluation','evaluate_critic_replay_v2_parallel.py',cpu_env())
        evaluation=read(O/'v2_counterfactual_evaluation.json');eligible=[m for m,v in evaluation['prerequisites'].items() if v['ready_for_new_gradient']]
        if eligible:
            run_stage(state,'new_gradient_branches','run_new_v2_gradient_stage.py',sim_env(),['--methods',','.join(eligible)],venv='.venv_sim')
            gradient=read(O/'v2_new_gradient_metrics.json');allowed=[m for m,v in gradient['methods'].items() if v['passed']]
            for m in allowed:
                run_stage(state,f'{m}_fixed_qam_pilot','run_fixed_qam_v2_pilot.py',gpu_env(7),['--method',m])
            state.update(stage='FIXED_REFERENCE_PILOT_COMPLETE' if allowed else 'STOP_ACTOR_QAM',actor_updates=bool(allowed),gradient_methods=eligible,actor_methods=allowed)
        else:state.update(stage='STOP_ACTOR_QAM',reason='Fixed2000 held-out ranking/Spearman/Near-ID prerequisites not all met; no new gradient rollouts or actor updates')
        state['decision']=state['stage'];atomic(STATUS,state)
        run_stage(state,'reports','finalize_replay_v2_stage.py',cpu_env())
        state.update(stage=state['decision'],finished_at=now());atomic(STATUS,state);print('REPLAY_V2_PIPELINE_FINISHED',json.dumps(state),flush=True)
    except BaseException as e:
        state.update(stage='FAILED',error=repr(e),failed_at=now());atomic(STATUS,state)
        try:subprocess.run([str(R/'.venv_actor/bin/python'),str(R/'scripts/finalize_replay_v2_stage.py')],cwd=R,env=cpu_env(),check=True)
        except Exception:pass
        raise

if __name__=='__main__':main()
