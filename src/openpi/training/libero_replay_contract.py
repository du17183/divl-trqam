"""Separate executed critic prefixes from complete, masked actor BC sequences."""
import numpy as np


def behavior_sequences(episode, horizon=10):
    actions = np.asarray(episode['actions'], dtype=np.float32)
    masks = np.asarray(episode['action_mask'], dtype=bool)
    lengths = np.asarray(episode['lengths'], dtype=np.int32)
    if actions.ndim != 3 or actions.shape[-1] != 7 or masks.shape != actions.shape[:2]:
        raise ValueError('LIBERO action/mask shapes are invalid')
    if not np.array_equal(masks, np.arange(actions.shape[1])[None, :] < lengths[:, None]):
        raise ValueError('Critic action mask must describe the real executed prefix')
    if (lengths < 1).any() or (lengths > 5).any():
        raise ValueError('Actual LIBERO chunk length must be in [1,5]')
    terminals = np.asarray(episode['terminal'], dtype=bool)
    if terminals.shape != lengths.shape or terminals[:-1].any() or not terminals[-1]:
        raise ValueError('An episode must end at its first terminal; no cross-episode labels')
    sequence = np.concatenate([a[m] for a, m in zip(actions, masks, strict=True)])
    offsets = np.concatenate(([0], np.cumsum(lengths[:-1])))
    targets = np.zeros((len(lengths), horizon, 7), dtype=np.float32)
    valid = np.zeros((len(lengths), horizon), dtype=bool)
    for i, start in enumerate(offsets):
        n = min(horizon, len(sequence) - int(start))
        targets[i, :n] = sequence[int(start):int(start) + n]
        # Padding is a storage convention only; its loss mask is false.
        targets[i, n:] = targets[i, n - 1]
        valid[i, :n] = True
    return targets, valid


def validate_episode(episode, gamma=0.9999):
    targets, mask = behavior_sequences(episode)
    k = np.asarray(episode['lengths'])
    expected_discount = np.where(episode['terminal'], 0, gamma ** k)
    np.testing.assert_allclose(episode['discounts'], expected_discount, atol=1e-7)
    n = len(k)
    for name in ['base', 'wrist', 'state']:
        if len(episode[name]) != n + 1:
            raise ValueError(f'{name}: obs/next_obs chunk continuity failed')
    if np.asarray(episode['state']).shape != (n + 1, 8):
        raise ValueError('LIBERO proprio must have dimension 8')
    if 'actor_bc_executed_sequence' in episode:
        np.testing.assert_allclose(episode['actor_bc_executed_sequence'], targets, atol=1e-6)
    if 'actor_bc_mask' in episode:
        np.testing.assert_array_equal(episode['actor_bc_mask'], mask)
    return targets, mask
