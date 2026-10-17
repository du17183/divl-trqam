"""New train-only phase-selected normal rollouts and five same-state branches."""
import argparse, hashlib, json, multiprocessing as mp, os, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from libero.libero import benchmark
from examples.libero.main import _get_libero_env, LIBERO_ENV_RESOLUTION, LIBERO_DUMMY_ACTION
from examples.libero.collect_divl import collect_episode, policy_observation
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from openpi.training.libero_replay_contract import behavior_sequences, validate_episode
from libero_snapshot_v2 import take_snapshot, restore_snapshot
from collect_action_sensitive_replay_v2 import BranchPolicy, NativeAudit
from action_sensitive_v3_common import R, D, O, atomic, sha, action_bounds, normalized, candidates, select_phase

WORKER_PORT = None


def initialize_worker(ports, gpus):
    global WORKER_PORT
    slot = (mp.current_process()._identity[0] - 1) % len(ports)
    WORKER_PORT = ports[slot]
    os.environ['MUJOCO_EGL_DEVICE_ID'] = str(gpus[slot])
    os.environ['OMP_NUM_THREADS'] = '2'


class EnvAudit:
    def __init__(self, env): self.env = env; self.native = []
    def step(self, action):
        result = self.env.step(action); self.native.append(float(result[1])); return result
    def check_success(self): return self.env.check_success()


def phase_features(env, obs, plan, previous):
    core = env.env; robot = core.robots[0]; gripper = robot.gripper
    objects = getattr(core, 'objects_dict', {})
    geometry = set(); positions = []; grasped = False
    for obj in objects.values():
        positions.append(np.asarray(env.sim.data.body_xpos[env.sim.model.body_name2id(obj.root_body)]))
        names = getattr(obj, 'contact_geoms', [])
        geometry.update(env.sim.model.geom_name2id(name) for name in names)
        grasped = grasped or bool(core._check_grasp(gripper=gripper, object_geoms=names))
    grip_names = gripper.contact_geoms
    if isinstance(grip_names, dict): grip_names = [g for v in grip_names.values() for g in v]
    grip_geometry = {env.sim.model.geom_name2id(name) for name in grip_names}
    grip_contacts = 0; other_contacts = 0
    for contact in env.sim.data.contact[:env.sim.data.ncon]:
        a, b = int(contact.geom1), int(contact.geom2)
        grip_contacts += int((a in geometry and b in grip_geometry) or (b in geometry and a in grip_geometry))
        other_contacts += int((a in geometry and b not in geometry and b not in grip_geometry)
                              or (b in geometry and a not in geometry and a not in grip_geometry))
    eef = np.asarray(obs['observation/state'][:3])
    distance = min((np.linalg.norm(eef - p) for p in positions), default=1.)
    return dict(nearest_object_distance=float(distance), robot_object_contacts=grip_contacts,
                object_other_contacts=other_contacts, grasped=grasped,
                grasp_seen_before=bool(previous and (previous['grasped'] or previous['grasp_seen_before'])),
                gripper_command=float(plan[0, 6]),
                previous_gripper_command=float(previous['gripper_command']) if previous else -1.,
                eef_motion=float(np.linalg.norm(eef-np.asarray(previous['eef_position']))) if previous else 0.,
                eef_position=eef.tolist(), object_count=len(objects),
                note='Generic observed-contact phase proxy; object-other contact includes support surfaces, not a task-specific target oracle')


class SourcePolicy:
    def __init__(self, client, env, seed):
        self.client=client; self.env=env; self.seed=seed
        self.plans=[]; self.snapshots=[]; self.hidden=[]; self.phases=[]
    def infer(self, obs):
        flat, hidden = take_snapshot(self.env)
        plan = np.asarray(self.client.infer({**obs, '_noise_seed': self.seed*1000+len(self.plans)})['actions'], np.float32)
        assert plan.shape == (10, 7) and np.isfinite(plan).all()
        # Physical state is for state selection/debug only, never actor inputs.
        phase = phase_features(self.env, obs, plan, self.phases[-1] if self.phases else None)
        self.snapshots.append(flat); self.hidden.append(hidden); self.plans.append(plan); self.phases.append(phase)
        return dict(actions=plan)


def add_contract(episode, native, low, high, case, episode_id, source):
    full, mask = behavior_sequences(episode)
    executed = np.concatenate([a[m] for a,m in zip(episode['actions'],episode['action_mask'],strict=True)])
    episode.update(actor_bc_executed_sequence=full, actor_bc_mask=mask,
        critic_executed_actions=episode['actions'].copy(), critic_action_mask=episode['action_mask'].copy(),
        critic_k=episode['lengths'].copy(), raw_env_actions=executed,
        normalized_actor_actions=normalized(executed, low, high), env_step_rewards=np.asarray(native,np.float32),
        task_id=np.asarray(case['task_id']), task_name=np.asarray(case['task_name']),
        init_id=np.asarray(case['initial_state_id']), seed=np.asarray(case['seed']),
        source=np.asarray(source), split=np.asarray('train'), episode_id=np.asarray(episode_id),
        dataset_version=np.asarray('action_sensitive_replay_v3'),
        split_hash=np.asarray(json.loads((D/'protocol.json').read_text())['split_hash']))
    validate_episode(episode)
    assert len(native) == int(episode['lengths'].sum())
    offset=0
    for reward, k in zip(episode['rewards'],episode['lengths'],strict=True):
        expected=sum(.9999**j*float(native[offset+j]) for j in range(int(k)))
        assert abs(expected-float(reward))<1e-6, 'Native reward differs from collector'
        offset+=int(k)
    return episode


def save_npz(path, **arrays):
    tmp=path.with_suffix('.partial')
    with tmp.open('wb') as stream: np.savez_compressed(stream,**arrays)
    tmp.replace(path)


def source_case(case):
    suite=benchmark.get_benchmark_dict()['libero_10'](); task=suite.get_task(case['task_id'])
    split=json.loads((R/'configs/splits/libero_factorial_v1.json').read_text())
    assert case['initial_state_id'] in split['tasks'][task.name]['train']
    path=D/'sources'/f'{case["trajectory_id"]}.npz'; meta=path.with_suffix('.json')
    if meta.exists():
        saved=json.loads(meta.read_text()); assert sha(path)==saved['sha256']; return saved
    low,high=action_bounds(); env,prompt=_get_libero_env(task,LIBERO_ENV_RESOLUTION,case['seed'])
    client=WebsocketClientPolicy('127.0.0.1',WORKER_PORT)
    try:
        env.seed(case['seed']); env.reset(); obs=env.set_init_state(suite.get_task_init_states(case['task_id'])[case['initial_state_id']])
        for _ in range(10): obs,_,_,_=env.step(LIBERO_DUMMY_ACTION)
        audited=EnvAudit(env); policy=SourcePolicy(client,env,case['seed'])
        episode=collect_episode(audited,policy,obs,prompt,520,replan_steps=5,action_horizon=10,gamma=.9999)
        add_contract(episode,audited.native,low,high,case,case['trajectory_id'],'v3_source_rollout')
        episode.update(snapshot_states=np.stack(policy.snapshots), snapshot_metadata=np.asarray(policy.hidden),
                       policy_action_plan=np.stack(policy.plans), phase_features_json=np.asarray(json.dumps(policy.phases)),
                       model_xml=np.asarray(env.sim.model.get_xml()))
        index,selection=select_phase(policy.phases,episode['lengths'],case['phase_index'],case['selection_seed'])
        save_npz(path,**episode)
        row={**case,**selection,'path':str(path),'sha256':sha(path),'source_index':index,
             'state_id':case['trajectory_id']+f'_chunk{index:03d}',
             'source_chunks':len(episode['rewards']),'source_steps':int(episode['lengths'].sum()),
             'remaining_steps':520-int(episode['lengths'][:index].sum()),
             'phase_selection_frozen_before_branch':True}
        atomic(meta,row); print('V3_SOURCE_DONE',json.dumps(row),flush=True); return row
    finally: env.close()


def branch_case(row):
    sid=row['state_id']; target=D/'states'/f'{sid}.json'; protocol_sha=sha(D/'protocol.json')
    if target.exists():
        saved=json.loads(target.read_text()); assert saved['protocol_sha256']==protocol_sha
        return saved
    assert sha(row['path'])==row['sha256']
    with np.load(row['path'],allow_pickle=False) as f: source={k:f[k] for k in f.files}
    low,high=action_bounds(); stats=json.loads((D/'action_distribution.json').read_text())
    sigma=np.asarray(stats[str(row['task_id'])]['normalized']['std'],np.float32)
    i=row['source_index']; k=int(source['lengths'][i])
    plans,norms,padded,specs=candidates(source['actions'][i],k,sigma,row['branch_seed'],low,high)
    flat=source['snapshot_states'][i]; hidden=source['snapshot_metadata'][i]
    env,prompt=_get_libero_env(benchmark.get_benchmark_dict()['libero_10']().get_task(row['task_id']),LIBERO_ENV_RESOLUTION,row['seed'])
    client=WebsocketClientPolicy('127.0.0.1',WORKER_PORT); env.seed(row['seed']);env.reset();records=[]
    try:
        for j,(name,scale,direction) in enumerate(specs):
            obs=restore_snapshot(env,flat,hidden)
            state_error=float(abs(env.get_sim_state()-flat).max()); assert state_error==0
            po=policy_observation(obs,prompt)
            assert np.array_equal(po['observation/image'],source['base'][i])
            assert np.array_equal(po['observation/wrist_image'],source['wrist'][i])
            assert np.array_equal(po['observation/state'],source['state'][i])
            policy=BranchPolicy(client,plans[j],row['seed'],i,env,source,j==0); audit=NativeAudit(env,k)
            episode=collect_episode(audit,policy,obs,prompt,row['remaining_steps'],replan_steps=5,action_horizon=10,gamma=.9999)
            episode_id=sid+f'_cf{j}'; add_contract(episode,audit.native,low,high,row,episode_id,'counterfactual_train_v3')
            error=abs(float(episode['mc_returns'][0])-float(source['mc_returns'][i])) if j==0 else None
            if j==0:
                if error>1e-5 or policy.max_state_error>1e-7 or policy.max_pixel_error!=0 or policy.max_plan_error>1e-6:
                    debug=dict(state_id=sid,source=row,baseline_reproduction_error=error,
                               max_state_error=policy.max_state_error,max_pixel_error=policy.max_pixel_error,
                               max_plan_error=policy.max_plan_error,reason='Strict original behavior continuation restore gate failed')
                    save_npz(D/'invalid'/f'{sid}_baseline.npz',**episode)
                    atomic(D/'invalid'/f'{sid}.json',debug)
                    return dict(state_id=sid,valid=False,records=[],debug=debug)
            episode.update(state_id=np.asarray(sid),candidate_index=np.asarray(j),branch_seed=np.asarray(row['branch_seed']),
                           phase=np.asarray(row['achieved_phase']), initial_candidate_raw=plans[j],
                           initial_candidate_normalized=norms[j], initial_candidate_padded32=padded[j],
                           initial_planned_mask=np.arange(10)<k,
                           perturbation_direction=np.asarray(direction,np.float32),perturbation_scale=np.asarray(scale))
            path=D/'episodes'/f'{episode_id}.npz';save_npz(path,**episode)
            records.append(dict(episode_id=episode_id,state_id=sid,task_id=row['task_id'],task_name=row['task_name'],
                initial_state_id=row['initial_state_id'],source_episode_id=row['trajectory_id'],source_index=i,source_path=row['path'],
                phase=row['achieved_phase'],assigned_phase=row['assigned_phase'],phase_fallback=row['phase_fallback'],
                candidate_index=j,candidate_name=name,perturbation_scale=scale,perturbation_direction=direction.tolist(),
                normalized_sigma=sigma.tolist(),behavior_rms=float(np.sqrt(np.mean((norms[j,:k]-norms[0,:k])**2))),
                raw_behavior_rms=float(np.sqrt(np.mean((plans[j,:k]-plans[0,:k])**2))),
                path=str(path),sha256=sha(path),future_return=float(episode['mc_returns'][0]),success=bool(episode['success']),
                terminal=bool(episode['terminal'][-1]),chunks=len(episode['rewards']),steps=int(episode['lengths'].sum()),
                initial_state_error=state_error,baseline_reproduction_error=error,k=k,remaining_steps=row['remaining_steps'],
                snapshot_id=sid,continuation_noise_seed_rule='source_seed*1000+source_index+continuation_chunk',
                branch_seed=row['branch_seed']))
        save_npz(D/'states'/f'{sid}.npz',flat=flat,metadata=np.asarray(hidden),base=source['base'][i],wrist=source['wrist'][i],
                 proprio=source['state'][i],candidate_raw=plans,candidate_normalized=norms,candidate_padded32=padded,
                 valid_action_mask=np.arange(10)<k,source_path=np.asarray(row['path']),source_index=np.asarray(i),
                 instruction=source['prompt'],snapshot_id=np.asarray(sid))
        result=dict(state_id=sid,valid=True,source=row,records=records,protocol_sha256=protocol_sha)
        atomic(target,result);print('V3_STATE_DONE',sid,'returns',[x['future_return'] for x in records],flush=True);return result
    except Exception as exc:
        atomic(D/'invalid'/f'{sid}.exception.json',dict(source=row,error=repr(exc),completed_records=records))
        raise
    finally:env.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['sources','branches'],required=True)
    p.add_argument('--ports',default='8030,8031,8032,8033,8034,8035,8036,8037');p.add_argument('--gpus',default='0,1,2,3,4,5,6,7')
    p.add_argument('--workers',type=int,default=16);p.add_argument('--limit',type=int);args=p.parse_args()
    ports=[int(v) for v in args.ports.split(',')];gpus=[int(v) for v in args.gpus.split(',')]
    assert len(ports)==len(gpus)
    protocol=json.loads((D/'protocol.json').read_text())
    assert protocol['code_hashes']['collect_action_sensitive_replay_v3.py']==sha(Path(__file__))
    records=json.loads((D/'source_cases.json').read_text()) if args.stage=='sources' else json.loads((D/'state_selection_manifest.json').read_text())['records']
    if args.limit:records=records[:args.limit]
    started=time.monotonic();collected=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),initializer=initialize_worker,initargs=(ports,gpus)) as pool:
        jobs=[pool.submit(source_case if args.stage=='sources' else branch_case,row) for row in records]
        for job in as_completed(jobs):
            collected.append(job.result());elapsed=time.monotonic()-started
            atomic(D/f'{args.stage}_progress.json',dict(stage=args.stage,status='collecting',completed=len(collected),target=len(records),
                elapsed_seconds=elapsed,items_per_minute=len(collected)*60/elapsed,workers=args.workers,inference_gpus=gpus))
    if args.limit:
        atomic(O/f'{args.stage}_preflight.json',dict(passed=all(x.get('valid',True) for x in collected),records=collected))
        return
    if args.stage=='sources':
        assert len(collected)==300 and len({r['state_id'] for r in collected})==300
        atomic(D/'state_selection_manifest.json',dict(records=sorted(collected,key=lambda x:x['state_id']),states=300,
            selection_before_branch_outcomes=True,dev_states_used=False,protocol_sha256=sha(D/'protocol.json')))
    else:
        valid=[x for x in collected if x['valid']]; rows=[r for group in valid for r in group['records']]
        assert len(rows)==len(valid)*5
        atomic(D/'manifest.json',dict(version='action_sensitive_replay_v3',states=len(valid),attempted_states=len(collected),
            invalid_states=len(collected)-len(valid),outcomes=len(rows),records=sorted(rows,key=lambda x:x['episode_id']),
            split_hash=protocol['split_hash'],protocol_sha256=sha(D/'protocol.json'),
            state_selection_sha256=sha(D/'state_selection_manifest.json'),all_outcomes_retained=True,
            dev_states_used=False,final_states_used=False,actor_updated=False,elapsed_seconds=time.monotonic()-started))
    atomic(D/f'{args.stage}_progress.json',dict(stage=args.stage,status='complete',completed=len(collected),target=len(records),elapsed_seconds=time.monotonic()-started))
    print('V3_COLLECTION_STAGE_COMPLETE',args.stage,len(collected),flush=True)


if __name__=='__main__':main()
