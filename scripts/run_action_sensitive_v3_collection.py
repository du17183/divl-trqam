"""Eight frozen policy servers, independent sims, strict first-case preflight."""
import argparse, fcntl, json, os, shutil, socket, subprocess, time
from datetime import datetime, timezone
from pathlib import Path
from action_sensitive_v3_common import R, D, O, atomic, sha
from run_critic_pilot_after_collection import common
from run_replay_v2_pipeline import safe_stop_server


def now(): return datetime.now(timezone.utc).isoformat()


def sim_env():
    env=common();env.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',
        OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',
        PYTHONPATH=env['PYTHONPATH']+':'+str(R/'third_party/libero'),
        LIBERO_CONFIG_PATH=str(R/'configs/libero_runtime'), TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',
        MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl')
    return env


def run_stage(state,name,script,args=(),env=None,venv='.venv_sim'):
    log=R/'logs/stage_v1'/f'v3_{name}.log';assert not log.exists()
    with log.open('w') as output:
        p=subprocess.Popen([str(R/venv/'bin/python'),'-u',str(R/'scripts'/script),*args],cwd=R,
                           env=env or sim_env(),stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT)
        state.update(stage=name,current_pid=p.pid,current_log=str(log),updated_at=now());atomic(O/'pipeline_status.json',state)
        while p.poll() is None:
            observation=dict(utc=now(),stage=name,gpu=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,power.draw','--format=csv,noheader'],capture_output=True,text=True).stdout,
                load_average=os.getloadavg(),disk_free_bytes=shutil.disk_usage(R).free)
            with (O/'runtime.jsonl').open('a') as runtime:runtime.write(json.dumps(observation)+'\n')
            if shutil.disk_usage(R).free<30*1024**3:
                p.terminate();raise RuntimeError('Low disk reserve; stop only our current stage')
            time.sleep(10)
        if p.returncode:raise RuntimeError(f'{name} failed exit{p.returncode}; {log}')
    state['completed'].append(name);atomic(O/'pipeline_status.json',state)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args()
    O.mkdir(parents=True,exist_ok=True);lock=(O/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (O/'pipeline_status.json').exists()
    assert json.loads((R/'artifacts/replay_action_audit_v3/pipeline_status.json').read_text())['stage']=='STOP_ACTOR_QAM'
    state=dict(stage='setup',started_at=now(),completed=[],actor_updates=False,beta_dynamic=False,gpus=list(range(8)),workers=16)
    servers=[];handles=[]
    try:
        if not (D/'protocol.json').exists():
            run_stage(state,'protocol','prepare_action_sensitive_replay_v3.py',venv='.venv_actor',env=sim_env())
        ports=list(range(8030,8038));gpu_memory=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout
        assert min(float(v) for v in gpu_memory.splitlines())>60*1024
        for gpu,port in enumerate(ports):
            with socket.socket() as probe:assert probe.connect_ex(('127.0.0.1',port))!=0,'Port already used'
            env=common();env.update(CUDA_VISIBLE_DEVICES=str(gpu),JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_MEM_FRACTION='.20')
            log=R/'logs/stage_v1'/f'v3_frozen_policy_gpu{gpu}.log';assert not log.exists();stream=log.open('w');handles.append(stream)
            p=subprocess.Popen([str(R/'.venv_actor/bin/python'),'-u',str(R/'scripts/serve_pilot_policy.py'),'--port',str(port)],cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            servers.append(dict(gpu=gpu,port=port,pid=p.pid,process=p,log=str(log)))
        atomic(O/'frozen_servers_dispatch.json',[{k:v for k,v in x.items() if k!='process'} for x in servers])
        state.update(stage='loading_eight_frozen_policies',updated_at=now());atomic(O/'pipeline_status.json',state)
        deadline=time.monotonic()+900;ready=set()
        while len(ready)<8:
            for server in servers:
                if server['process'].poll() is not None:raise RuntimeError('Our frozen server exited: '+server['log'])
                if server['port'] in ready:continue
                try:
                    with socket.create_connection(('127.0.0.1',server['port']),timeout=.2):ready.add(server['port'])
                except OSError:pass
            if time.monotonic()>deadline:raise TimeoutError('Frozen policy load timeout; no collection started')
            state.update(ready_servers=len(ready));atomic(O/'pipeline_status.json',state);time.sleep(2)
        base=['--workers','16','--ports',','.join(map(str,ports)),'--gpus','0,1,2,3,4,5,6,7']
        # A complete native source+five-action first-case gate precedes full collection.
        run_stage(state,'source_preflight','collect_action_sensitive_replay_v3.py',base+['--stage','sources','--limit','1'])
        assert json.loads((O/'sources_preflight.json').read_text())['passed']
        pre=json.loads((O/'sources_preflight.json').read_text())['records']
        atomic(D/'state_selection_manifest.json',dict(records=pre,preflight=True,selection_before_branch_outcomes=True))
        run_stage(state,'branch_preflight','collect_action_sensitive_replay_v3.py',base+['--stage','branches','--limit','1'])
        assert json.loads((O/'branches_preflight.json').read_text())['passed']
        if args.preflight_only:
            state.update(stage='PREFLIGHT_COMPLETE',finished_at=now());atomic(O/'pipeline_status.json',state);return
        run_stage(state,'normal_train_rollouts','collect_action_sensitive_replay_v3.py',base+['--stage','sources'])
        run_stage(state,'same_state_branches','collect_action_sensitive_replay_v3.py',base+['--stage','branches'])
        for s in servers:safe_stop_server(s['pid'],s['port'])
        state.update(stage='V3_COLLECTION_COMPLETE',finished_at=now());atomic(O/'pipeline_status.json',state)
        for p,h in json.loads((O/'frozen_input_hashes.json').read_text()).items():assert sha(p)==h
        # The driver and analysis are separate processes: no inference allocator
        # competes with feature extraction or the matched critic experiments.
        if (R/'scripts/run_action_sensitive_v3_analysis_pipeline.py').exists():
            run_stage(state,'v3_analysis_and_gated_critics','run_action_sensitive_v3_analysis_pipeline.py',venv='.venv_actor',env=sim_env())
            state.update(stage=json.loads((O/'analysis_pipeline_status.json').read_text())['stage'],finished_at=now());atomic(O/'pipeline_status.json',state)
        else:
            state.update(stage='READY_FOR_V3_ANALYSIS',reason='Collection complete; analysis driver not yet installed');atomic(O/'pipeline_status.json',state)
    except BaseException as exc:
        state.update(stage='FAILED',error=repr(exc),failed_at=now());atomic(O/'pipeline_status.json',state);raise
    finally:
        for server in servers:safe_stop_server(server['pid'],server['port'])
        for stream in handles:stream.close()


if __name__=='__main__':main()
