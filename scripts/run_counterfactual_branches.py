# Public release: deployment paths anonymized; algorithm equations unchanged.
"""True same-snapshot return tests; frozen policy and common continuation noise."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import hashlib
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
from libero.libero import benchmark
from examples.libero.main import _get_libero_env,LIBERO_ENV_RESOLUTION
from examples.libero.collect_divl import policy_observation
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from libero_snapshot import restore_snapshot

ROOT=Path('/opt/divl-trqam')

def arrays(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k] for k in f.files}

def state_branches(item,port):
    scalar=arrays(item['scalar']);divl=arrays(item['divl']);meta=json.loads(str(scalar['metadata']))
    input_hashes={method:hashlib.sha256(Path(item[method]).read_bytes()).hexdigest() for method in ['scalar','divl']}
    episode=arrays(meta['episode_path']);i=meta['index'];k=meta['k']
    assert np.array_equal(scalar['candidate_actions'],divl['candidate_actions'])
    task=benchmark.get_benchmark_dict()['libero_10']().get_task(meta['task_id'])
    env,prompt=_get_libero_env(task,LIBERO_ENV_RESOLUTION,meta['seed'])
    client=WebsocketClientPolicy('127.0.0.1',port)
    choices=[('candidate',str(j),None,a) for j,a in enumerate(scalar['candidate_actions'])]
    for method,data in [('scalar',scalar),('divl',divl)]:
        for epsilon in [.01,.03,.05]:
            for sign in [-1,1]:choices.append(('gradient',str(sign),method,data[f'gradient_{epsilon}_{sign}']))
    target=ROOT/'data/counterfactual_libero_v1'/f'{meta["state_id"]}.json'
    if target.exists():
        old=json.loads(target.read_text())
        if old.get('input_hashes')!=input_hashes:raise ValueError('Existing branch data belongs to different diagnostic inputs')
        if len(old['branches'])==20:return old
        raise ValueError('Incomplete pre-existing branch file')
    results=[];started=time.monotonic()
    try:
        # Rebuild from the original official task XML, not MuJoCo's exported
        # get_xml(): its rounded/recomputed inertials changed model constants.
        env.reset()
        for order,(kind,label,method,action) in enumerate(choices):
            obs=restore_snapshot(env,episode['snapshot_states'][i],episode['snapshot_metadata'][i])
            error=float(np.max(np.abs(env.get_sim_state()-episode['snapshot_states'][i])))
            if error>1e-9:raise AssertionError('Snapshot reset mismatch')
            start_state=policy_observation(obs,prompt)['observation/state']
            proprio_error=float(np.max(np.abs(start_state-episode['state'][i])))
            if proprio_error>1e-5:raise AssertionError(('Observation/proprio mismatch',proprio_error))
            steps=0;value=0.;success=False;done=False
            for a in action[:k]:
                obs,r,done,_=env.step(np.clip(a,-1,1).tolist())
                success=bool(env.check_success());value+=.9999**steps*float(r);steps+=1
                if success or done:break
            next_state_error=None
            if kind=='candidate' and label=='0' and i+1<len(episode['snapshot_states']):
                next_state_error=float(np.max(np.abs(env.get_sim_state()-episode['snapshot_states'][i+1])))
                if next_state_error>1e-7:raise AssertionError(('Original behavior prefix does not reproduce next snapshot',next_state_error))
            continuation_chunk=i+1
            while not success and not done and steps<meta['remaining_steps']:
                request=policy_observation(obs,prompt)
                request['_noise_seed']=meta['seed']*1000+continuation_chunk
                plan=np.asarray(client.infer(request)['actions'],np.float32);continuation_chunk+=1
                for a in plan[:min(5,meta['remaining_steps']-steps)]:
                    obs,r,done,_=env.step(np.clip(a,-1,1).tolist())
                    success=bool(env.check_success());value+=.9999**steps*float(r);steps+=1
                    if success or done:break
            epsilon=[.01,.03,.05][(order-8)%6//2] if kind=='gradient' else None
            row={'state_id':meta['state_id'],'task_id':meta['task_id'],'initial_state_id':meta['initial_state_id'],'episode_id':str(episode['episode_id']),
                 'kind':kind,'candidate_index':int(label) if kind=='candidate' else None,'method':method,
                 'epsilon':epsilon,'direction':int(label) if kind=='gradient' else None,'future_return':value,'success':success,'steps':steps,
                 'initial_state_error':error,'proprio_error':proprio_error,'baseline_next_state_error':next_state_error,'action':action.tolist(),
                 'q_scalar':float(scalar['candidate_q'][int(label)]) if kind=='candidate' else None,
                 'q_divl':float(divl['candidate_q'][int(label)]) if kind=='candidate' else None}
            if not np.isfinite(value):raise FloatingPointError('Nonfinite native future return')
            if kind=='candidate' and label=='0':
                row['baseline_future_error']=abs(value-float(episode['mc_returns'][i]))
                if row['baseline_future_error']>1e-5:raise AssertionError(('Frozen behavior continuation not reproduced',row['baseline_future_error']))
            results.append(row)
            print('BRANCH '+json.dumps({k:v for k,v in row.items() if k!='action'}),flush=True)
    finally:env.close()
    result={'metadata':meta,'input_hashes':input_hashes,'branches':results,'elapsed_seconds':time.monotonic()-started}
    target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_suffix('.partial');tmp.write_text(json.dumps(result));tmp.replace(target)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8027);p.add_argument('--workers',type=int,default=3);p.add_argument('--steps',type=int,default=2000);args=p.parse_args()
    assert json.loads((ROOT/'artifacts/pilot_snapshot_preflight.json').read_text())['passed']
    scalar=json.loads((ROOT/f'artifacts/static_action_scalar_step{args.steps}.json').read_text())
    divl=json.loads((ROOT/f'artifacts/static_action_divl_step{args.steps}.json').read_text())
    methods={m['state_id']:m for m in divl['records']}
    items=[{'scalar':m['path'],'divl':methods[m['state_id']]['path']} for m in scalar['records'] if m['selected_for_branches']]
    assert len(items)==50
    collected=[];start=time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as pool:
        futures=[pool.submit(state_branches,item,args.port) for item in items]
        for future in as_completed(futures):
            collected.append(future.result())
            (ROOT/'artifacts/counterfactual_progress.json').write_text(json.dumps({'states':len(collected),'target_states':50,'branches':len(collected)*20,'elapsed_seconds':time.monotonic()-start}))
    # 400 candidate branches plus 600 gradient branches; no actor optimizer is instantiated.
    (ROOT/'artifacts/counterfactual_results.json').write_text(json.dumps(collected))
    print('COUNTERFACTUAL_COMPLETE '+json.dumps({'states':50,'candidate_branches':400,'gradient_branches':600,'elapsed_seconds':time.monotonic()-start}),flush=True)

if __name__=='__main__':main()
