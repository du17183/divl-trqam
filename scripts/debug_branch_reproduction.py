# Public release: deployment paths anonymized; algorithm equations unchanged.
"""Independent original-action-only diagnostic; never generates candidates."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
from libero.libero import benchmark
from examples.libero.main import _get_libero_env, LIBERO_ENV_RESOLUTION
from examples.libero.collect_divl import policy_observation
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from libero_snapshot import restore_snapshot, numeric_state

ROOT = Path('/opt/divl-trqam')
SIM_FIELDS = ['qpos', 'qvel', 'act', 'ctrl', 'time', 'mocap_pos', 'mocap_quat',
              'qacc', 'qacc_warmstart', 'qfrc_applied', 'xfrc_applied', 'qfrc_constraint']

def arrays(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}

def maxerr(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)), initial=0))

def rng_record():
    state = np.random.get_state()
    out = {'numpy': [state[0], state[1].tolist(), state[2], state[3], state[4]],
           'python': random.getstate()}
    try:
        import torch
        out['torch_cpu'] = torch.get_rng_state().tolist()
        out['torch_cuda'] = [s.tolist() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else []
        out['torch_cuda_initialized'] = torch.cuda.is_initialized()
    except ImportError:
        out['torch'] = 'not imported/available in simulator process'
    out['jax'] = 'Remote frozen policy uses explicit normal flow noise from recorded noise seed; no simulator JAX key'
    return out

def observe_record(env, obs, prompt):
    p = policy_observation(obs, prompt)
    return {'proprio': p['observation/state'], 'base': p['observation/image'], 'wrist': p['observation/wrist_image'],
            'camera_agentview': np.asarray(obs['agentview_image']),
            'camera_wrist': np.asarray(obs['robot0_eye_in_hand_image'])}

def controller_record(env):
    rows = []
    for robot in env.robots:
        row = {'controller': numeric_state(robot.controller), 'gripper': numeric_state(robot.gripper)}
        for k in ['interpolator_pos', 'interpolator_ori']:
            obj = getattr(robot.controller, k, None)
            if obj is not None:
                row[k] = numeric_state(obj)
        rows.append(row)
    return rows

def trace(meta, client, *, run_id, save=True, creation='legacy', restore_version=1):
    ep = arrays(meta['episode_path'])
    i = int(meta['index']); k = int(meta['k'])
    task = benchmark.get_benchmark_dict()['libero_10']().get_task(int(meta['task_id']))
    env_seed = int(meta['seed']) if creation == 'legacy' else 20261007 + int(meta['task_id'])
    env, prompt = _get_libero_env(task, LIBERO_ENV_RESOLUTION, env_seed)
    path = ROOT/'data/reproduction_debug_v1'/meta['state_id']
    path.mkdir(parents=True, exist_ok=True)
    vectors = {key: [] for key in SIM_FIELDS}
    for key in ['proprio', 'base', 'wrist', 'camera_agentview', 'camera_wrist', 'action', 'reward', 'done', 'success', 'controller', 'environment']:
        vectors[key] = []
    checks=[]; plans=[]; first=None; value=0.; steps=0; success=False; done=False
    def addcheck(row):
        nonlocal first
        checks.append(row)
        if first is None:
            if row.get('state_error', 0)>1e-7: first={**row, 'classification':'A/D: simulator state divergence'}
            elif row.get('proprio_error', 0)>1e-5 or row.get('pixel_max', 0)>0: first={**row, 'classification':'B: observable differs at recorded chunk anchor'}
            elif row.get('plan_error', 0)>1e-6: first={**row, 'classification':'C: action differs; inspect input/noise first'}
            elif row.get('executed_action_error',0)>1e-6: first={**row,'classification':'C/D: executed action differs'}
            elif row.get('reward_error', 0)>1e-7: first={**row, 'classification':'D/E: native reward divergence'}
    try:
        if creation != 'legacy': env.seed(int(meta['seed']))
        env.reset()
        if restore_version==2:
            from libero_snapshot_v2 import migrate_v1_for_branch, restore_snapshot as restore_v2
            migrated=migrate_v1_for_branch(env,ep['snapshot_states'][i],ep['snapshot_metadata'][i],ep['model_xml'])
            obs=restore_v2(env,ep['snapshot_states'][i],migrated)
        else:
            obs=restore_snapshot(env, ep['snapshot_states'][i], ep['snapshot_metadata'][i])
        offset=int(ep['lengths'][:i].sum())
        chunk=i
        while not success and not done and steps<int(meta['remaining_steps']):
            p=policy_observation(obs,prompt)
            if chunk<len(ep['snapshot_states']):
                pixels=np.concatenate([(p['observation/image'].astype(float)-ep['base'][chunk]).reshape(-1),
                                       (p['observation/wrist_image'].astype(float)-ep['wrist'][chunk]).reshape(-1)])
                addcheck({'step':steps,'chunk':chunk,'phase':'before_inference',
                          'state_error':maxerr(env.get_sim_state(),ep['snapshot_states'][chunk]),
                          'proprio_error':maxerr(p['observation/state'],ep['state'][chunk]),
                          'pixel_max':float(np.max(np.abs(pixels))), 'pixel_mean':float(np.mean(np.abs(pixels))),
                          'rng':rng_record()})
            noise=int(ep['policy_noise_seeds'][chunk]) if chunk<len(ep['policy_noise_seeds']) else int(meta['seed'])*1000+chunk
            plan=np.asarray(client.infer({**p,'_noise_seed':noise})['actions'],np.float32)
            plans.append(plan)
            if chunk<len(ep['policy_action_plan']):
                addcheck({'step':steps,'chunk':chunk,'phase':'inference','plan_error':maxerr(plan,ep['policy_action_plan'][chunk]),'noise_seed':noise})
            action=ep['actions'][i][:k] if chunk==i else plan[:min(5,int(meta['remaining_steps'])-steps)]
            for a in action:
                a=np.clip(a,-1,1).astype(np.float32)
                before=observe_record(env,obs,prompt)
                if save:
                    for key in SIM_FIELDS: vectors[key].append(np.asarray(getattr(env.sim.data,key)).copy())
                    for key,v in before.items(): vectors[key].append(v.copy())
                    vectors['controller'].append(json.dumps(controller_record(env),separators=(',',':')))
                    numeric=numeric_state(env.env)
                    # Task numeric state is retained; large sensor images are already stored above.
                    vectors['environment'].append(json.dumps(numeric,separators=(',',':')))
                obs,reward,done,_=env.step(a.tolist());success=bool(env.check_success())
                reward=float(reward)
                if offset+steps<len(ep['raw_env_actions']):
                    addcheck({'step':steps,'chunk':chunk,'phase':'after_step',
                              'executed_action_error':maxerr(a,ep['raw_env_actions'][offset+steps]),
                              'reward_error':abs(reward-float(ep['env_step_rewards'][offset+steps]))})
                value += .9999**steps*reward;steps+=1
                if save:
                    vectors['action'].append(a.copy());vectors['reward'].append(reward)
                    vectors['done'].append(bool(done));vectors['success'].append(success)
                if success or done: break
            chunk+=1
        error=abs(value-float(ep['mc_returns'][i]))
        result={'state_id':meta['state_id'],'run_id':run_id,'creation_mode':creation,'snapshot_version':restore_version,'steps':steps,'success':success,
                'future_return':value,'recorded_return':float(ep['mc_returns'][i]),'future_return_error':error,
                'passed_return':error<=1e-5,'first_divergence':first,'checks':checks,
                'original_per_step_limitation':'Original dataset saves simulator/controller snapshots and RGB at inference boundaries, plus all executed actions/rewards. Per-step simulator values in this package are measured restore runs; unavailable historical per-step states are not fabricated.'}
        if save:
            np.savez_compressed(path/f'{run_id}.npz',**{key:np.asarray(v) for key,v in vectors.items()},policy_plans=np.asarray(plans))
            result['trace_path']=str(path/f'{run_id}.npz')
        (path/f'{run_id}.json').write_text(json.dumps(result,indent=2))
        compact={k:v for k,v in result.items() if k not in ['checks']}
        if compact['first_divergence'] is not None:
            compact['first_divergence']={k:v for k,v in compact['first_divergence'].items() if k!='rng'}
        print('REPRODUCTION_RUN',json.dumps(compact),flush=True)
        return result
    finally: env.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--state-id');p.add_argument('--screen',action='store_true')
    p.add_argument('--repeats',type=int,default=20);p.add_argument('--creation',choices=['legacy','collector'],default='legacy')
    p.add_argument('--port',type=int,default=8027)
    p.add_argument('--restore-version',type=int,choices=[1,2],default=1);args=p.parse_args()
    records=json.loads((ROOT/'artifacts/static_action_scalar_step2000.json').read_text())['records']
    selected=[d for d in records if d['selected_for_branches']]
    client=WebsocketClientPolicy('127.0.0.1',args.port)
    if args.screen:
        selected=[d for d in selected if not (ROOT/'data/counterfactual_libero_v1'/f"{d['state_id']}.json").exists()]
        results=[]
        for meta in selected:
            out=trace(meta,client,run_id=f'screen_v{args.restore_version}_'+args.creation,save=False,creation=args.creation,restore_version=args.restore_version)
            results.append(out)
            (ROOT/'artifacts/reproduction_screen.json').write_text(json.dumps(results,indent=2))
            if not out['passed_return']:
                (ROOT/'artifacts/reproduction_failed_state.json').write_text(json.dumps(meta,indent=2))
                print('FAILED_STATE_IDENTIFIED',meta['state_id'],flush=True)
                return
        return
    meta=next(d for d in selected if d['state_id']==args.state_id) if args.state_id else json.loads((ROOT/'artifacts/reproduction_failed_state.json').read_text())
    ep=arrays(meta['episode_path']);i=int(meta['index'])
    fixed={'observation/image':ep['base'][i],'observation/wrist_image':ep['wrist'][i],
           'observation/state':ep['state'][i],'prompt':str(ep['prompt']),'_noise_seed':int(ep['policy_noise_seeds'][i])}
    actions=np.stack([np.asarray(client.infer(fixed)['actions'],np.float32) for _ in range(20)])
    directory=ROOT/'data/reproduction_debug_v1'/meta['state_id'];directory.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(directory/f'fixed_input_policy_20_v{args.restore_version}.npz',actions=actions,executed_prefix=actions[:,:5],noise_seed=np.asarray(fixed['_noise_seed']))
    fixed_test={'state_id':meta['state_id'],'repeats':20,'max_action_difference':maxerr(actions,actions[0]),
                'recorded_plan_error':maxerr(actions[0],ep['policy_action_plan'][i]),'explicit_flow_noise_seed':fixed['_noise_seed']}
    (ROOT/'artifacts/fixed_input_policy_reproduction.json').write_text(json.dumps(fixed_test,indent=2))
    print('FIXED_INPUT_POLICY',json.dumps(fixed_test),flush=True)
    results=[]
    for n in range(args.repeats):
        results.append(trace(meta,client,run_id=f'v{args.restore_version}_{args.creation}_run{n:02d}',save=True,creation=args.creation,restore_version=args.restore_version))
        compact=[{k:v for k,v in d.items() if k!='checks'} for d in results]
        (ROOT/'artifacts'/f'reproduction_debug_progress_v{args.restore_version}.json').write_text(json.dumps({'completed':len(results),'target':args.repeats,'runs':compact},indent=2))
    result={'state_id':meta['state_id'],'creation_mode':args.creation,'policy_fixed_input':fixed_test,'runs':results,
            'max_future_return_error':max(d['future_return_error'] for d in results),'repeats':args.repeats}
    (ROOT/'artifacts'/(f'reproduction_debug_v{args.restore_version}_'+args.creation+'.json')).write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
