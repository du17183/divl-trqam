"""Executed-prefix TD and real BC labels; CPU only, no model or environment import."""
import numpy as np


def normalize_real(actions, norm):
    actions = np.asarray(actions, np.float32)
    assert actions.ndim == 2 and actions.shape[1] == 7 and np.isfinite(actions).all()
    statistics = norm['statistics']['actions']
    low = np.asarray(statistics['q01'], np.float32)
    high = np.asarray(statistics['q99'], np.float32)
    return (actions - low) / (high - low + np.float32(1e-6)) * 2 - 1


def make_chunk_arrays(actions, rewards, dones, successes, plans_raw, plans_normalized, ks, norm, gamma):
    """Plans describe the chosen H5; BC targets describe only actions actually sent to env.step.

    A real BC H10 can span two replanning chunks. Unexecuted predicted tails never fill it.
    Native goal or native done ends bootstrapping. A 520-step collection cap is censoring.
    """
    actions = np.asarray(actions, np.float32).reshape((-1, 7))
    rewards = np.asarray(rewards, np.float64)
    dones = np.asarray(dones, bool)
    successes = np.asarray(successes, bool)
    plans_raw = np.asarray(plans_raw, np.float32).reshape((-1, 10, 7))
    plans_normalized = np.asarray(plans_normalized, np.float32).reshape((-1, 10, 32))
    ks = np.asarray(ks, np.int32)
    length, count = len(actions), len(ks)
    assert 0 < gamma <= 1 and np.isfinite(gamma)
    assert rewards.shape == dones.shape == successes.shape == (length,)
    assert len(plans_raw) == len(plans_normalized) == count
    assert np.all((ks >= 1) & (ks <= 5)) and int(ks.sum()) == length
    assert all(np.isfinite(x).all() for x in (actions, rewards, plans_raw, plans_normalized))
    # No transition after a terminal event, even if the original file had a fabricated tail.
    assert not np.any(dones[:-1] | successes[:-1])
    starts = np.concatenate(([0], np.cumsum(ks)[:-1])).astype(np.int32) if count else np.empty(0, np.int32)
    ends = starts + ks
    result = {
        'start_observation_index': starts, 'next_observation_index': ends, 'actual_k': ks,
        'planned_h5_normalized': plans_normalized[:, :5, :7].copy(),
        'planned_h5_fixed_mask': np.ones((count, 5, 7), np.float32),
        'executed_prefix_raw': np.zeros((count, 5, 7), np.float32),
        'executed_prefix_normalized': np.zeros((count, 5, 7), np.float32),
        'executed_prefix_mask': np.zeros((count, 5, 7), np.float32),
        'chunk_reward_raw_sum': np.zeros(count, np.float64),
        'chunk_reward_discounted': np.zeros(count, np.float64),
        'bootstrap_discount': np.zeros(count, np.float64),
        'native_terminal': np.zeros(count, bool), 'goal_success': np.zeros(count, bool),
        'bc_actual_h10_raw': np.zeros((count, 10, 7), np.float32),
        'bc_actual_h10_normalized': np.zeros((count, 10, 32), np.float32),
        'bc_actual_h10_mask': np.zeros((count, 10, 32), np.float32),
        'bc_full_h10_eligible': np.zeros(count, bool),
        'bc_actual_steps_available': np.zeros(count, np.int32),
    }
    normalized_actual = normalize_real(actions, norm)
    for index, (start, end, k) in enumerate(zip(starts, ends, ks)):
        assert np.array_equal(actions[start:end], plans_raw[index, :k])
        result['executed_prefix_raw'][index, :k] = actions[start:end]
        result['executed_prefix_normalized'][index, :k] = normalized_actual[start:end]
        result['executed_prefix_mask'][index, :k] = 1
        result['chunk_reward_raw_sum'][index] = rewards[start:end].sum()
        result['chunk_reward_discounted'][index] = np.dot(gamma ** np.arange(k), rewards[start:end])
        terminal = bool(dones[end - 1] or successes[end - 1])
        result['native_terminal'][index] = bool(dones[end - 1])
        result['goal_success'][index] = bool(successes[end - 1])
        result['bootstrap_discount'][index] = 0 if terminal else gamma ** int(k)
        available = min(10, length - int(start))
        result['bc_actual_steps_available'][index] = available
        result['bc_actual_h10_raw'][index, :available] = actions[start:start + available]
        result['bc_actual_h10_normalized'][index, :available, :7] = normalized_actual[start:start + available]
        result['bc_actual_h10_mask'][index, :available, :7] = 1
        result['bc_full_h10_eligible'][index] = available == 10
    return result


def validate_episode_group(group, norm, gamma):
    """Independently reconstruct arrays and check explicit s_t/s_(t+k) image/state alignment."""
    arrays = make_chunk_arrays(group['environment_actions_raw'][:], group['native_rewards'][:],
                               group['native_dones'][:], group['native_successes'][:],
                               group['predicted_raw_h10'][:], group['predicted_normalized_h10'][:],
                               group['chunks/actual_k'][:], norm, gamma)
    length = len(group['environment_actions_raw'])
    for name in ['agentview_rgb', 'eye_in_hand_rgb', 'proprio', 'sim_state']:
        assert len(group['observations/' + name]) == length + 1
    for name in ['agentview_rgb', 'eye_in_hand_rgb']:
        image = group['observations/' + name]
        assert image.dtype == np.dtype('uint8') and image.shape[1:] == (128, 128, 3)
    assert group['observations/proprio'].shape == (length + 1, 8)
    assert np.isfinite(group['observations/proprio'][:]).all()
    assert np.isfinite(group['observations/sim_state'][:]).all()
    assert np.array_equal(group['environment_actions_normalized'][:], normalize_real(group['environment_actions_raw'][:], norm))
    for name, expected in arrays.items():
        observed = group['chunks/' + name][:]
        assert np.array_equal(observed, expected), name
    assert int(group.attrs['actual_environment_steps']) == length
    assert bool(group.attrs['success']) == bool(np.any(group['native_successes'][:]) or group.attrs['initial_goal_success'])
    if length:
        censored = not bool(group['native_dones'][-1] or group['native_successes'][-1])
        assert bool(group.attrs['collection_censored']) == censored
        assert bool(group.attrs['collection_censored']) or arrays['bootstrap_discount'][-1] == 0
    return {'environment_steps': length, 'transitions': len(arrays['actual_k']),
            'bc_full_h10_transitions': int(arrays['bc_full_h10_eligible'].sum()),
            'partial_h10_tail_transitions': int((~arrays['bc_full_h10_eligible']).sum())}
