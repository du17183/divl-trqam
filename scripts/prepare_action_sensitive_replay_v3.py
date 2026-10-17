"""Freeze v3 protocol, all30 train initial IDs/task, and empirical action scales."""
import json, numpy as np
from action_sensitive_v3_common import R, D, O, SEED, PHASES, atomic, sha, action_bounds, normalized


def main():
    assert not (D / 'protocol.json').exists(), 'Do not overwrite a v3 protocol'
    split_path = R / 'configs/splits/libero_factorial_v1.json'
    split = json.loads(split_path.read_text())
    original = json.loads((R / 'data/pilot_libero_v1/manifest.json').read_text())
    assert original['episodes'] == 200 and original['split_hash'] == split['sha256']
    low, high = action_bounds(); values = {t: [] for t in range(10)}
    for row in original['records']:
        assert row['initial_state_id'] in split['tasks'][row['task_name']]['train']
        assert sha(row['path']) == row['sha256']
        with np.load(row['path'], allow_pickle=False) as e:
            values[row['task_id']].append(e['actions'][e['action_mask']])
    distribution = {}
    for t, arrays in values.items():
        raw = np.concatenate(arrays); norm = normalized(raw, low, high)
        distribution[str(t)] = dict(samples=len(raw),
            raw=dict(mean=raw.mean(0).tolist(), std=raw.std(0).tolist(), quantiles=np.quantile(raw, [.1,.5,.9], axis=0).tolist()),
            normalized=dict(mean=norm.mean(0).tolist(), std=norm.std(0).tolist(), quantiles=np.quantile(norm, [.1,.5,.9], axis=0).tolist()))
    atomic(D / 'action_distribution.json', distribution)
    cases = []
    for task_name in split['task_order']:
        task = split['tasks'][task_name]; ids = task['train']; assert len(ids) == 30
        for rank, init in enumerate(ids):
            cases.append(dict(task_id=task['task_id'], task_name=task_name, initial_state_id=int(init),
                              trajectory_id=f'v3_task{task["task_id"]:02d}_init{init:03d}',
                              seed=2026100800 + task['task_id'] * 10000 + int(init) * 10,
                              phase_index=rank % len(PHASES),
                              selection_seed=SEED + task['task_id'] * 1000 + int(init),
                              branch_seed=SEED + 1000000 + task['task_id'] * 1000 + int(init)))
    assert len(cases) == 300 and len({c['seed'] for c in cases}) == 300
    for name in ['sources', 'episodes', 'states', 'invalid']:
        (D / name).mkdir(parents=True, exist_ok=True)
    atomic(D / 'source_cases.json', cases)
    frozen_paths = [R / 'data/pilot_libero_v1/manifest.json', R / 'data/action_sensitive_replay_v2/manifest.json',
                    R / 'artifacts/counterfactual_final.parquet', R / 'artifacts/counterfactual_results_v2.json',
                    R / 'artifacts/replay_action_audit_v3/v2_counterfactual_evaluation.json', split_path]
    files = {str(p): sha(p) for p in frozen_paths}
    protocol = dict(dataset_version='action_sensitive_replay_v3', seed=SEED,
        target_states=300, actions_per_state=5, target_outcomes=1500, tasks=10,
        split_hash=split['sha256'], source_manifest_sha256=sha(R / 'data/pilot_libero_v1/manifest.json'),
        cases_sha256=sha(D / 'source_cases.json'), action_distribution_sha256=sha(D / 'action_distribution.json'),
        phases=PHASES, phase_selector='One assigned phase per unique train initial ID; 5 IDs/phase/task; predicates use current physical contacts/grasp/EEF-object proximity/gripper events; fixed fallback near interaction; no future labels or branch labels',
        candidate_mode='A: coherent7D perturbation over actual five-step executed prefix; H10 masked suffix unchanged',
        scales=dict(small=.25, medium=.5, large=1.), candidates=['behavior','small_minus','small_plus','medium','large'],
        sigma='Per-task per-real-dimension STD of ALL executed normalized actions in immutable original200 train episodes; no fixed action-unit amplitude or outcome tuning',
        gamma=.9999, action_dim=7, padded_dim=32, horizon=10, replan_steps=5, max_episode_steps=520,
        restore_gate=1e-5, snapshot_version=2, invalid_state_policy='Retain debug and baseline trajectory; exclude invalid state from train manifest, no outcome-based replacement',
        continuation='Frozen original pi05_libero; seed=source_seed*1000+source_chunk+continuation_chunk; first native success or native terminal/horizon',
        informative_thresholds=[1e-6,1e-4,1e-3,1e-2], positive_control=dict(folds=5, grouping='Whole(task,initial_state_id); all candidate actions stay together',
        encoder='Public frozen two-camera SigLIP, not trained critic; actor-visible8Dproprio only', pca=32,
        hidden=[128,64], updates=1000, batch_states=64, lr=.001, loss='MSE +0.1 pairwise softplus /0.1',
        engineering_gate='mean held-out rank>=0.60 and state/task bootstrap CIlo>0.50; fixed before any outcomes'),
        replay_mixes=dict(R1=dict(deployment=.75,branch=.25),R2=dict(deployment=.5,branch=.5)),
        formal_budget=2000, formal_checkpoints=[250,500,1000,2000], global_batch=256, critic_lr=.0002,
        td_chunks=10, actor_updates=False, beta_dynamic=False, dev_states_used=False, final_states_used=False,
        frozen_input_hashes=files,
        code_hashes={name:sha(R/'scripts'/name) for name in ['action_sensitive_v3_common.py','collect_action_sensitive_replay_v3.py']})
    atomic(D / 'protocol.json', protocol)
    atomic(O / 'frozen_input_hashes.json', files)
    report = '# V3 action distribution\n\nAll executed real7D actions from original200train episodes. Official quantile normalization retained. Quantiles below are P10/P50/P90; STD is populationSTD. Zero-variance channels remain unperturbed; no outcome-dependent sigma floor.\n\n' + json.dumps(distribution, indent=2)
    (R / 'reports/v3_action_distribution_report.md').write_text(report)
    (R / 'reports/v3_state_selection_report.md').write_text('# V3 preregistered state selection\n\n300 attempted states:30unique train initial IDs/task,1 normal frozen rollout and1phase-selected anchor/ID. All ten officialLIBERO_Long tasks. No dev/final IDs.\n\n' + json.dumps(protocol, indent=2))
    print('V3_PROTOCOL_FROZEN', sha(D / 'protocol.json'), flush=True)


if __name__ == '__main__': main()
