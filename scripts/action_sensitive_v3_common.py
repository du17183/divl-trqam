# Public release: deployment paths anonymized; algorithm equations unchanged.
"""Immutable interfaces and preregistered v3 state/action rules."""
from pathlib import Path
import hashlib, json, numpy as np

R = Path('/opt/divl-trqam')
D = R / 'data/action_sensitive_replay_v3'
O = R / 'artifacts/action_sensitive_replay_v3'
SEED = 2026100723
PHASES = ['approach_object', 'pregrasp_contact', 'postgrasp_transport',
          'approach_target_or_release', 'placement_contact', 'contact_alignment']


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.partial')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def action_bounds():
    x = json.loads((R / 'checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json').read_text())['norm_stats']['actions']
    return np.asarray(x['q01'], np.float32)[:7], np.asarray(x['q99'], np.float32)[:7]


def normalized(raw, low, high):
    return (2 * (raw - low) / (high - low + 1e-6) - 1).astype(np.float32)


def candidates(behavior, k, sigma, seed, low, high):
    assert behavior.shape == (10, 7) and 1 <= k <= 5
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(3, 7)).astype(np.float32)
    specs = [('behavior', 0., np.zeros(7, np.float32)),
             ('small_minus', .25, -directions[0]),
             ('small_plus', .25, directions[0]),
             ('medium', .50, directions[1]),
             ('large', 1., directions[2])]
    base = normalized(behavior, low, high)
    plans, norms = [], []
    for name, size, direction in specs:
        if size == 0:
            raw = behavior.copy()
        else:
            proposed = base + size * np.asarray(sigma, np.float32) * direction
            raw = np.clip((proposed + 1) / 2 * (high - low + 1e-6) + low, -1, 1).astype(np.float32)
        raw[k:] = 0
        plans.append(raw)
        norms.append(np.where(np.arange(10)[:, None] < k, normalized(raw, low, high), 0))
    plans, norms = np.stack(plans), np.stack(norms).astype(np.float32)
    padded = np.zeros((5, 10, 32), np.float32); padded[:, :, :7] = norms
    assert np.count_nonzero(padded[:, :, 7:]) == 0
    return plans, norms, padded, specs


def select_phase(features, lengths, assigned, seed):
    """Select using current physical/observable state only, never future labels."""
    valid = np.flatnonzero(np.asarray(lengths) == 5)
    if not len(valid):
        raise ValueError('No full five-step source prefix; keep source as invalid selection')
    candidates_by_phase = []
    for i in valid:
        f = features[int(i)]
        near = f['nearest_object_distance'] < .12
        contact = f['robot_object_contacts'] > 0
        grasp = bool(f['grasped'])
        seen = bool(f['grasp_seen_before'])
        other = f['object_other_contacts'] > 0
        closing = f['gripper_command'] > 0 and f['previous_gripper_command'] <= 0
        opening = f['gripper_command'] <= 0 and f['previous_gripper_command'] > 0
        tests = [near and not grasp,
                 (contact or closing) and not grasp,
                 grasp and f['eef_motion'] > .001,
                 seen and (opening or grasp),
                 seen and other,
                 contact and (grasp or f['nearest_object_distance'] < .08)]
        if tests[assigned]:
            candidates_by_phase.append(int(i))
    # Fixed fallbacks are reported, not hidden. No replacement based on branch G.
    fallback = not candidates_by_phase
    if fallback:
        scores = np.array([features[int(i)]['robot_object_contacts']
                           + float(features[int(i)]['grasped'])
                           - features[int(i)]['nearest_object_distance'] for i in valid])
        candidates_by_phase = valid[np.argsort(scores)[-max(1, len(valid) // 4):]].tolist()
    index = int(np.random.default_rng(seed).choice(candidates_by_phase))
    return index, dict(assigned_phase=PHASES[assigned], phase_fallback=fallback,
                      achieved_phase=PHASES[assigned] if not fallback else 'fallback_near_interaction',
                      eligible_phase_chunks=len(candidates_by_phase),
                      selector_inputs=features[index], future_return_used=False,
                      success_label_used=False, branch_results_used=False)
