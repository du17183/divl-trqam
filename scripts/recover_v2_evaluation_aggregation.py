"""Recover only CPU aggregation of the eight completed frozen evaluations."""
from pathlib import Path
import fcntl, hashlib, json, subprocess
from run_replay_v2_pipeline import O, STATUS, atomic, cpu_env, now, read, run_stage, R


def main():
    lock = (O / 'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = read(STATUS)
    assert state['stage'] == 'FAILED'
    assert 'frozen50_evaluation failed' in state['error']
    assert read(O / 'matched_v2_training_complete.json')['passed']
    assert not state['actor_updates'] and not state['beta_dynamic']
    files = [O / f'v2_eval_{method}_step{step:04d}.json'
             for method in ['scalar', 'divl'] for step in [250, 500, 1000, 2000]]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    for p in files:
        result = read(p)
        assert all(len(result[b]['states']) == 50
                   for b in ['all', 'Near-ID', 'Mid', 'Far-OOD'])
    backup = R / 'backups/evaluation_aggregation_20261007'
    backup.mkdir(parents=True, exist_ok=True)
    prior = backup / 'failed_pipeline_status.json'
    assert not prior.exists()
    prior.write_bytes(STATUS.read_bytes())
    state.setdefault('execution_recoveries', []).append(dict(
        at=now(), prior_error=state.pop('error'), failed_at=state.pop('failed_at'),
        reason='Undefined per-state rank serialized as null; preserve missing estimates and all fixed states during paired aggregation',
        raw_worker_result_hashes=hashes, gpu_reevaluation=False, retraining=False))
    env = cpu_env()
    env['V2_EVAL_AGGREGATE'] = '1'
    try:
        run_stage(state, 'frozen50_evaluation_recovery', 'evaluate_critic_replay_v2.py', env)
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
                   for p, h in hashes.items())
        evaluation = read(O / 'v2_counterfactual_evaluation.json')
        eligible = [m for m, v in evaluation['prerequisites'].items()
                    if v['ready_for_new_gradient']]
        # Current actual fixed2000 worker results already show both ranking
        # gates fail. This recovery never launches extra training or rollout.
        assert not eligible, 'Unexpected eligible method: use original gated continuation'
        if 'frozen50_evaluation' not in state['completed']:
            state['completed'].append('frozen50_evaluation')
        state.update(stage='STOP_ACTOR_QAM', decision='STOP_ACTOR_QAM',
                     reason='Fixed2000 held-out ranking/Spearman/Near-ID prerequisites not all met; no new gradient rollouts or actor updates')
        atomic(STATUS, state)
        run_stage(state, 'reports_after_evaluation_recovery', 'finalize_replay_v2_stage.py', cpu_env())
        state.update(stage=state['decision'], finished_at=now())
        atomic(STATUS, state)
        print('RECOVERY_COMPLETE', json.dumps(state), flush=True)
    except BaseException as exc:
        state.update(stage='FAILED', error=repr(exc), failed_at=now())
        atomic(STATUS, state)
        raise


if __name__ == '__main__':
    main()
