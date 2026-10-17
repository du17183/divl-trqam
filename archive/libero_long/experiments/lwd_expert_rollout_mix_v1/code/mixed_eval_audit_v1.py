"""Audit completed native shards; reuse their bytes and execute missing shards only."""
from pilot8_common_v1 import *
from mixed_common_v2 import EXP,CODE,OLD
import numpy as np
import h5py


def completed_shards(method, protocol_path, inspect_trace=True):
    protocol = read(protocol_path)
    directory = EXP / 'evaluation' / protocol['evaluation_id']
    trace_directory = EXP / 'data/evaluation' / protocol['evaluation_id']
    records = []
    completed = {}
    for path in sorted(directory.glob('task_*_shard_*.json')):
        data = read(path)
        tid, shard = data['task_id'], data['shard']
        assert 0 <= tid < 10 and 0 <= shard < 5
        assert path.name == f'task_{tid:02d}_shard_{shard:02d}.json'
        assert data['complete'] and data['episodes'] == 10, ('Incomplete shard requires explicit archival', path)
        assert data['protocol_sha256'] == sha(protocol_path)
        assert data['evaluation_id'] == protocol['evaluation_id']
        assert data['checkpoint_manifest_sha256'] == protocol['checkpoint_manifest_sha256']
        assert (data['settle_steps'], data['policy_step_limit'], data['executed_prefix'], data['horizon']) == (10, 520, 5, 10)
        assert data['first_official_source_reset_gate']['passed']
        assert data['first_official_source_reset_gate']['state_max_error'] == 0
        assert data['first_official_source_reset_gate']['rgb_max_error'] == 0
        task = protocol['tasks'][tid]
        assert data['task_name'] == task['task_name']
        expected = task['episodes'][shard * 10:(shard + 1) * 10]
        assert len(data['records']) == len(expected) == 10
        trace = trace_directory / f'task_{tid:02d}_shard_{shard:02d}.hdf5'
        assert str(trace) == data['trace_hdf5'] and sha(trace) == data['trace_sha256']
        with h5py.File(trace, 'r') as handle:
            assert set(handle.keys()) == {f"{e['official_index']:02d}" for e in expected}
            for row, episode in zip(data['records'], expected):
                assert all(row[k] == v for k, v in episode.items())
                assert row['task_id'] == tid and row['policy_seed'] == 20261009
                assert row['inference_checkpoint_manifest_sha256'] == protocol['checkpoint_manifest_sha256']
                # Same reset threshold as the unchanged native shard worker.
                assert 0 <= row['initial_restore_error'] < 1e-12
                group = handle[f"{row['official_index']:02d}"]
                assert group.attrs['state_source_id'] == row['state_source_id']
                assert group.attrs['initial_vector_sha256'] == row['initial_vector_sha256']
                assert hashlib.sha256(np.asarray(group['official_initial_state']).tobytes()).hexdigest() == row['initial_vector_sha256']
                assert bool(group.attrs['success']) == row['success']
                if inspect_trace:
                    k = np.asarray(group['actual_k'])
                    raw = np.asarray(group['raw_h10'])
                    normalized = np.asarray(group['normalized_h10'])
                    actions = np.asarray(group['executed_actions'])
                    rewards = np.asarray(group['native_rewards'])
                    dones = np.asarray(group['native_dones'])
                    assert raw.shape == (len(k), 10, 7) and normalized.shape == (len(k), 10, 32)
                    assert np.isfinite(raw).all() and np.isfinite(normalized).all() and np.isfinite(rewards).all()
                    assert np.all((k >= 1) & (k <= 5))
                    assert int(k.sum()) == row['episode_length'] == len(actions) == len(rewards)
                    assert np.array_equal(np.concatenate([a[:int(n)] for a, n in zip(raw, k)]), actions)
                    assert float(rewards.sum()) == row['return']
                    assert (bool(dones[-1]) if len(dones) else False) == row['native_done']
                    assert group['pre_action_states'].shape[0] == group['pre_action_proprio'].shape[0] == len(actions)
                records.append((tid, row['state_source_id']))
        key = (tid, shard)
        assert key not in completed
        completed[key] = {'path': str(path), 'sha256': sha(path), 'trace_path': str(trace),
                          'trace_sha256': data['trace_sha256'], 'episodes': 10}
    assert len(set(records)) == len(records), 'Duplicate physical source'
    return completed

