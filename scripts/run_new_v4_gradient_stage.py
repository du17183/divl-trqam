# Public release: deployment paths anonymized; algorithm equations unchanged.
"""New critic directions on the original fixed50 snapshots; old truth is immutable."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,hashlib,json,multiprocessing as mp,time,subprocess,socket,os
import numpy as np
from libero.libero import benchmark
from examples.libero.main import _get_libero_env,LIBERO_ENV_RESOLUTION
from examples.libero.collect_divl import policy_observation
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from libero_snapshot_v2 import restore_snapshot
from run_replay_v2_pipeline import gpu_env,safe_stop_server,atomic

R=Path('/opt/divl-trqam');O=R/'artifacts/action_sensitive_replay_v4';MIX=os.environ.get('V4_GRADIENT_MIX','R1');D=R/'data/new_gradient_replay_v4'/MIX;EPS=[.01,.03,.05]
def arrays(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k] for k in f.files}
def future(env,obs,prompt,client,meta,action):
    steps=0;value=0.;success=False;done=False
    for a in action[:meta['k']]:
        obs,r,done,_=env.step(np.clip(a,-1,1).tolist());success=bool(env.check_success());value+=.9999**steps*float(r);steps+=1
        if success or done:break
    chunk=meta['index']+1
    while not success and not done and steps<meta['remaining_steps']:
        plan=np.asarray(client.infer({**policy_observation(obs,prompt),'_noise_seed':meta['seed']*1000+chunk})['actions'],np.float32);chunk+=1
        for a in plan[:min(5,meta['remaining_steps']-steps)]:
            obs,r,done,_=env.step(np.clip(a,-1,1).tolist());success=bool(env.check_success());value+=.9999**steps*float(r);steps+=1
            if success or done:break
    assert np.isfinite(value);return dict(future_return=value,success=success,steps=steps)

def branch(meta,methods,port):
    source=arrays(meta['episode_path']);snap=arrays(R/'data/reproduction_snapshots_v2'/f'{meta["state_id"]}.npz')
    inputs={m:arrays(R/'data/critic_replay_v4_diagnostic'/MIX/f'{meta["state_id"]}_{m}.npz') for m in methods}
    inputhash={m:hashlib.sha256((R/'data/critic_replay_v4_diagnostic'/MIX/f'{meta["state_id"]}_{m}.npz').read_bytes()).hexdigest() for m in methods}
    output=D/f'{meta["state_id"]}.json';ph=hashlib.sha256((D/'protocol.json').read_bytes()).hexdigest()
    if output.exists():
        saved=json.loads(output.read_text());assert saved['protocol_sha256']==ph and saved['input_hashes']==inputhash and len(saved['branches'])==len(methods)*6
        return saved
    env,prompt=_get_libero_env(benchmark.get_benchmark_dict()['libero_10']().get_task(meta['task_id']),LIBERO_ENV_RESOLUTION,meta['seed'])
    client=WebsocketClientPolicy('127.0.0.1',port);env.seed(meta['seed']);env.reset();rows=[];start=time.monotonic()
    try:
        def reset():
            obs=restore_snapshot(env,snap['flat'],snap['metadata']);assert np.array_equal(env.get_sim_state(),snap['flat'])
            po=policy_observation(obs,prompt)
            assert float(abs(po['observation/state']-source['state'][meta['index']]).max())<1e-5
            assert np.array_equal(po['observation/image'],source['base'][meta['index']]) and np.array_equal(po['observation/wrist_image'],source['wrist'][meta['index']])
            return obs
        baseline=future(env,reset(),prompt,client,meta,inputs[methods[0]]['candidate_actions'][0])
        error=abs(baseline['future_return']-float(source['mc_returns'][meta['index']]));assert error<1e-5
        for method in methods:
            for epsilon in EPS:
                for sign in [-1,1]:
                    action=inputs[method][f'gradient_{epsilon}_{sign}']
                    result=future(env,reset(),prompt,client,meta,action)
                    row=dict(method=method,epsilon=epsilon,direction=sign,state_id=meta['state_id'],task_id=meta['task_id'],action=action.tolist(),initial_state_error=0.,**result)
                    rows.append(row);print('NEW_V2_GRADIENT',json.dumps({k:v for k,v in row.items() if k!='action'}),flush=True)
        result=dict(metadata=meta,baseline=baseline,baseline_error=error,branches=rows,input_hashes=inputhash,protocol_sha256=ph,elapsed_seconds=time.monotonic()-start)
        atomic(output,result);return result
    except Exception as e:
        atomic(output.with_suffix('.invalid.json'),dict(metadata=meta,error=repr(e),completed_branches=rows));raise
    finally:env.close()

def statistics(groups,methods):
    groups.sort(key=lambda g:g['metadata']['state_id']);task=np.array([v['metadata']['task_id'] for v in groups]);rng=np.random.default_rng(20261007)
    boots=np.array([np.concatenate([rng.choice(np.flatnonzero(task==t),5,replace=True) for t in range(10)]) for _ in range(5000)])
    def describe(values):
        values=np.asarray(values);draw=values[boots].mean(axis=1)
        return dict(mean=float(values.mean()),median=float(np.median(values)),ci95=np.quantile(draw,[.025,.975]).tolist(),positive_ratio=float((values>0).mean()),n_states=len(values),state_values=values.tolist())
    results={}
    for m in methods:
        directions=[];improvements=[];by_eps={}
        for eps in EPS:
            plus=np.array([next(b['future_return'] for b in g['branches'] if b['method']==m and b['epsilon']==eps and b['direction']==1) for g in groups])
            minus=np.array([next(b['future_return'] for b in g['branches'] if b['method']==m and b['epsilon']==eps and b['direction']==-1) for g in groups])
            baseline=np.array([g['baseline']['future_return'] for g in groups]);directions.append(plus-minus);improvements.append(plus-baseline)
            by_eps[str(eps)]=dict(plus_minus=describe(plus-minus),plus_baseline=describe(plus-baseline))
        direction=np.mean(directions,axis=0);improve=np.mean(improvements,axis=0);d=describe(direction);im=describe(improve)
        positive=np.sort(np.maximum(direction,0))[::-1];share=float(positive[:2].sum()/max(float(positive.sum()),1e-15))
        loo=[np.delete(direction,i).mean() for i in range(50)]
        conditions=dict(direction_ci_positive=d['ci95'][0]>0,actual_improvement_ci_positive=im['ci95'][0]>0,broad_positive_states=d['positive_ratio']>=.5,top_two_not_dominant=share<.5,positive_without_any_single_state=min(loo)>0)
        results[m]=dict(passed=all(conditions.values()),conditions=conditions,across_epsilon_plus_minus=d,across_epsilon_plus_baseline=im,by_epsilon=by_eps,positive_contribution_share_top2=share,positive_contribution_share_top1=float(positive[:1].sum()/max(float(positive.sum()),1e-15)),positive_contribution_share_top5=float(positive[:5].sum()/max(float(positive.sum()),1e-15)),leave_one_state_out_mean_range=[float(min(loo)),float(max(loo))])
    result=dict(methods=results,states=50,new_direction_branches=50*len(methods)*6,baseline_checks=50,epsilon=EPS,bootstrap='5000 whole-state/task-stratified resamples; no state or epsilon selection',original_candidate_and_gradient_truth_unchanged=True)
    atomic(O/f'v4_new_gradient_{MIX}.json',result)
    (R/'reports'/f'v4_real_gradient_{MIX}_report.md').write_text('# V3 real gradient directions\n\n'+json.dumps(result,indent=2)+'\n\nOld600 outcomes test old directions only. These new independently simulated directions use the same original50 snapshots, native reward, epsilon space, horizon and frozen continuation noise. Plus-versus-baseline guards against an apparent gain caused solely by harming minus actions. All states/eps retained.\n')
    print('V2_REAL_GRADIENT_GATE',json.dumps({m:{k:v for k,v in d.items() if k not in ['by_epsilon','across_epsilon_plus_minus','across_epsilon_plus_baseline']} for m,d in results.items()}),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--methods',required=True);p.add_argument('--workers',type=int,default=6);args=p.parse_args();methods=args.methods.split(',');assert set(methods)<=set(['scalar','divl'])
    prerequisites=json.loads((O/f'v4_counterfactual_{MIX}.json').read_text())['prerequisites'];assert all(prerequisites[m]['ready_for_new_gradient'] for m in methods)
    D.mkdir(parents=True,exist_ok=True);assert not (D/'protocol.json').exists(),'Do not overwrite a new-gradient stage'
    oldtruth={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [R/'artifacts/counterfactual_final.parquet',R/'artifacts/counterfactual_results_v2.json']}
    atomic(D/'protocol.json',dict(methods=methods,primary_steps=2000,states=50,epsilon=EPS,actor_updated=False,baseline_check=True,oldtruth_hashes=oldtruth,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),broad_gradient_gate='state-mean plus-minus CIlo>0; actual plus-baseline CIlo>0; >=50%state means positive; top2 share<50%; any-one-state omitted mean>0'))
    serverlog=(R/'logs/stage_v1'/f'v4_{MIX}_gradient_frozen_server.log').open('w')
    server=subprocess.Popen([str(R/'.venv_actor/bin/python'),'-u',str(R/'scripts/serve_pilot_policy.py'),'--port','8027'],cwd=R,env=gpu_env(7,'.25'),stdout=serverlog,stderr=subprocess.STDOUT)
    groups=[]
    try:
        deadline=time.monotonic()+900
        while True:
            if server.poll() is not None:raise RuntimeError('Frozen gradient server failed')
            try:
                with socket.create_connection(('127.0.0.1',8027),timeout=1):break
            except OSError:
                if time.monotonic()>deadline:raise TimeoutError('Gradient frozen server startup timeout')
                time.sleep(2)
        dev=sorted([g['metadata'] for g in json.loads((R/'artifacts/counterfactual_results_v2.json').read_text())],key=lambda m:m['state_id']);assert len(dev)==50
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as pool:
            jobs=[pool.submit(branch,m,methods,8027) for m in dev]
            for j in as_completed(jobs):
                groups.append(j.result());atomic(O/f'new_gradient_v4_{MIX}_progress.json',dict(states=len(groups),target_states=50,branches=sum(len(g['branches']) for g in groups)))
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in oldtruth.items());assert len(groups)==50
        statistics(groups,methods)
    finally:safe_stop_server(server.pid,8027);serverlog.close()

if __name__=='__main__':main()
