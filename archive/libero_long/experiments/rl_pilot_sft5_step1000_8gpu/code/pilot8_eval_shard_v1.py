# Public release: deployment paths anonymized; algorithm equations unchanged.
"""Additional native closed-loop standard50 task; no policy or critic learning."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import time
from multiprocessing.connection import Client
import h5py
import numpy as np
import torch
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from standard50_reset_factory_v1 import NativeFreshResetEnv
from native_action_audit_v1 import native_action_contract, passive_action_record
from robosuite.utils.transform_utils import quat2axisangle

ROOT = Path('/data/divl-trqam/benchmark')
EXP=ROOT/'experiments/rl_pilot_sft5_step1000_8gpu'
AUTHKEY = __import__('os').environ.get('DIVL_TRQAM_IPC_KEY', 'local-development-only').encode()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, data):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def reset_source(env, initial, episode, protocol):
    env.seed(episode['environment_seed'])
    env.reset()
    obs = env.set_init_state(initial)
    error = float(np.max(np.abs(np.asarray(env.get_sim_state()) - initial)))
    assert error < 1e-12, ('official_initial_state_not_restored', error)
    for _ in range(protocol['settle_steps']):
        obs, _, _, _ = env.step(protocol['dummy_action'])
    return obs, error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',type=int,choices=range(8),required=True)
    parser.add_argument('--task-id', type=int, required=True)
    parser.add_argument('--shard',type=int,required=True)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--protocol-sha', required=True)
    parser.add_argument('--socket', required=True)
    args = parser.parse_args()
    import os
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(args.gpu) and os.environ['MUJOCO_EGL_DEVICE_ID']==str(args.gpu)
    assert sha(args.protocol) == args.protocol_sha
    protocol = json.loads(Path(args.protocol).read_text())
    tid = args.task_id
    task_info = protocol['tasks'][tid]
    assert 0<=args.shard<5
    selected_episodes=task_info['episodes'][10*args.shard:10*(args.shard+1)]
    assert len(selected_episodes)==10
    assert sha(task_info['init_file']) == task_info['init_file_sha256']
    assert sha(task_info['bddl_path']) == task_info['bddl_sha256']
    initial_states = np.asarray(torch.load(task_info['init_file'], map_location='cpu', weights_only=False))
    suite = benchmark.get_benchmark_dict()['libero_10'](task_order_index=0)
    task = suite.get_task(tid)
    assert task.name == task_info['task_name']
    constructor = NativeFreshResetEnv if tid in (3, 8, 9) else OffScreenRenderEnv
    env = constructor(bddl_file_name=task_info['bddl_path'], camera_heights=128, camera_widths=128, render_gpu_device_id=args.gpu)
    directory = EXP / 'evaluation' / protocol['evaluation_id']
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ('task_%02d_shard_%02d.json' % (tid,args.shard))
    assert not path.exists()
    traces = EXP / 'data/evaluation' / protocol['evaluation_id']
    traces.mkdir(parents=True, exist_ok=True)
    trace_path = traces / ('task_%02d_shard_%02d.hdf5' % (tid,args.shard))
    rows = []
    started = time.monotonic()
    result = {'task_id': tid, 'shard':args.shard, 'task_name': task.name, 'evaluation_id': protocol['evaluation_id'],
              'checkpoint_manifest_sha256': protocol['checkpoint_manifest_sha256'], 'protocol_sha256': args.protocol_sha,
              'partition': 'official_standard50_development_reference', 'records': rows, 'complete': False,
              'settle_steps': 10, 'policy_step_limit': 520, 'executed_prefix': 5, 'horizon': 10,
              'not_used_for_sft_selection': False, 'source_overlap': 'unknown',
              'reset_engineering_policy': 'fresh unchanged native environment per reset' if tid in (3, 8, 9) else 'original native reused environment',
              'reset_repair_evidence': str(ROOT / 'provenance/sft_standard50_001/reset_diagnostic001.json') if tid in (3, 8, 9) else None}
    try:
        first = selected_episodes[0]
        initial = initial_states[first['official_index']]
        obs_a, _ = reset_source(env, initial, first, protocol)
        state_a = env.get_sim_state().copy()
        rgb_a = obs_a['agentview_image'].copy()
        obs_b, _ = reset_source(env, initial, first, protocol)
        reset_error = float(np.max(np.abs(env.get_sim_state() - state_a)))
        rgb_error = int(np.max(np.abs(obs_b['agentview_image'].astype(np.int16) - rgb_a.astype(np.int16))))
        print("STRICT_RESET_GATE", tid, reset_error, rgb_error, flush=True)
        assert reset_error < 1e-12 and rgb_error == 0
        result['first_official_source_reset_gate'] = {'passed': True, 'state_max_error': reset_error,
                                                     'rgb_max_error': rgb_error, 'scope': 'two native resets of first official source with registered ten settle steps'}
        with h5py.File(trace_path, 'x') as out, Client(args.socket, family='AF_UNIX', authkey=AUTHKEY) as client:
            for episode in selected_episodes:
                before_episode = time.monotonic()
                initial = initial_states[episode['official_index']]
                assert hashlib.sha256(initial.tobytes()).hexdigest() == episode['initial_vector_sha256']
                obs, initial_error = reset_source(env, initial, episode, protocol)
                action_contract = native_action_contract(env)
                success = bool(env.check_success())
                initial_success = success
                native_done = False
                steps = chunk_index = 0
                group = out.create_group('%02d' % episode['official_index'])
                group.attrs['state_source_id'] = episode['state_source_id']
                group.attrs['initial_vector_sha256'] = episode['initial_vector_sha256']
                group.create_dataset('official_initial_state', data=initial)
                group.create_dataset('post_settle_state', data=env.get_sim_state().copy())
                rewards, dones, states, proprios, executed, normalized, predictions, ks, server_wall = ([] for _ in range(9))
                clipped_arm, scaled_arm, gripper_sign = [], [], []
                while steps < 520 and not success and not native_done:
                    words = np.frombuffer(bytes.fromhex(episode['state_source_id'])[:16], dtype='<u4')
                    noise = np.random.default_rng(np.random.SeedSequence([20261009, *map(int, words), chunk_index])).standard_normal((10, 32)).astype(np.float32)
                    proprio = np.concatenate([obs['robot0_eef_pos'], quat2axisangle(np.asarray(obs['robot0_eef_quat']).copy()), obs['robot0_gripper_qpos']]).astype(np.float32)
                    client.send({'rgb': {'agentview_rgb': obs['agentview_image'], 'eye_in_hand_rgb': obs['robot0_eye_in_hand_image']},
                                 'proprio': proprio, 'language': task.language, 'noise': noise})
                    prediction = client.recv()
                    assert prediction['ok'] and prediction['precision'] == 'FP32 highest'
                    actions = np.asarray(prediction['raw_h10'], np.float32)
                    assert actions.shape == (10, 7) and np.isfinite(actions).all()
                    predictions.append(actions)
                    normalized.append(prediction['normalized_h10'])
                    server_wall.append(prediction['native_sample_wall_s'])
                    k = 0
                    for action in actions[:min(5, 520 - steps)]:
                        states.append(env.get_sim_state().copy())
                        proprios.append(np.concatenate([obs['robot0_eef_pos'], quat2axisangle(np.asarray(obs['robot0_eef_quat']).copy()), obs['robot0_gripper_qpos']]))
                        executed.append(action.copy())
                        audit = passive_action_record(action_contract, action)
                        low, high = action_contract['input_low'], action_contract['input_high']
                        clipped_arm.append(audit['native_arm_clipped_input'])
                        scaled_arm.append(audit['native_arm_scaled_delta'])
                        gripper_sign.append(audit['native_gripper_sign_command'])
                        result['native_arm_clipped_components'] = result.get('native_arm_clipped_components', 0) + audit['arm_clipped_components']
                        result['native_arm_components'] = result.get('native_arm_components', 0) + 6
                        result.setdefault('controller_input_out_of_bounds_components',0)
                        result.setdefault('controller_input_components',0)
                        result['controller_input_out_of_bounds_components'] += int(np.sum((action < low) | (action > high)))
                        result['controller_input_components'] += int(action.size)
                        obs, reward, native_done, info = env.step(action.tolist())
                        native_done = bool(native_done)
                        rewards.append(float(reward)); dones.append(native_done)
                        steps += 1; k += 1; success = bool(env.check_success())
                        if native_done or success:
                            break
                    ks.append(k); chunk_index += 1
                reason = 'native_goal_success' if success else 'native_terminal_without_success' if native_done else '520_policy_steps_censored'
                for key, value in [('raw_h10', predictions), ('normalized_h10', normalized), ('actual_k', ks), ('executed_actions', executed),
                                   ('native_rewards', rewards), ('native_dones', dones), ('pre_action_states', states), ('pre_action_proprio', proprios),
                                   ('native_arm_clipped_input', clipped_arm), ('native_arm_scaled_delta', scaled_arm), ('native_gripper_sign_command', gripper_sign)]:
                    group.create_dataset(key, data=np.asarray(value))
                group.attrs['success'] = success
                group.attrs['failure_class'] = reason
                out.flush()
                row = {'task_id': tid, **episode, 'policy_seed':20261009, 'inference_checkpoint_manifest_sha256':protocol['checkpoint_manifest_sha256'], 'success': success, 'return': sum(rewards), 'episode_length': steps,
                       'initial_success_after_settle': initial_success, 'initial_restore_error': initial_error,
                       'native_done': native_done, 'censored': not native_done and not success, 'failure_class': reason,
                       'wall_s': time.monotonic() - before_episode, 'native_inference_total_s': sum(server_wall), 'chunks': chunk_index,
                       'fine_failure_classification': 'contact/grasp not instrumented; not inferred'}
                rows.append(row)
                atomic(path, result)
                print('STANDARD50_NATIVE_EPISODE', tid, episode['official_index'], steps, success, flush=True)
    finally:
        env.close()
    assert len(rows) == 10 and {row['official_index'] for row in rows} == set(range(args.shard*10,(args.shard+1)*10))
    result.update(complete=True, successes=sum(row['success'] for row in rows), episodes=10,
                  success_rate=float(np.mean([row['success'] for row in rows])), mean_return=float(np.mean([row['return'] for row in rows])),
                  mean_episode_length=float(np.mean([row['episode_length'] for row in rows])),
                  trace_hdf5=str(trace_path), trace_sha256=sha(trace_path), wall_s=time.monotonic() - started,
                  completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  controller_input_bound_violation_ratio=result.get('controller_input_out_of_bounds_components',0)/max(result.get('controller_input_components',0),1),
                  native_arm_clip_ratio=result.get('native_arm_clipped_components',0)/max(result.get('native_arm_components',0),1),
                  clipping_scope='Actual passed actions and native OSC scale_action input limits; passive exact clipping/scaling formula, unchanged native execution. Gripper uses sign and cumulative actuator saturation; torque clipping is not recorded.')
    atomic(path, result)


if __name__ == '__main__':
    main()
