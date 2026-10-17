# Public release: deployment paths anonymized; algorithm equations unchanged.
"""Regression and real saved-result checks for undefined fixed-state ranks."""
from pathlib import Path
import json, math
from critic_eval_pairing import paired_state_differences


def row(sid, rank, rho, pairs=1):
    return dict(state_id=sid, task_id=0, rank=rank, spearman=rho,
                pairs=pairs, candidates=8)


def main():
    scalar = [row('flat', None, None, 0), row('defined', .25, -.1),
              row('constant_prediction', .5, None)]
    divl = [row('flat', None, None, 0), row('defined', .75, .2),
            row('constant_prediction', .75, .3)]
    result = paired_state_differences(scalar, divl)
    assert len(result) == 3 and math.isnan(result[0]['rank'])
    assert result[1]['rank'] == .5 and math.isclose(result[1]['spearman'], .3)
    assert result[2]['rank'] == .25 and math.isnan(result[2]['spearman'])
    malformed = [dict(x) for x in divl]
    malformed[0]['state_id'] = 'different_state'
    try:
        paired_state_differences(scalar, malformed)
    except AssertionError:
        pass
    else:
        raise AssertionError('Mismatched state identity must not be paired')
    root = Path('/opt/divl-trqam/artifacts/replay_action_audit_v3')
    actual = []
    if root.exists():
        for step in [250, 500, 1000, 2000]:
            s = json.loads((root / f'v2_eval_scalar_step{step:04d}.json').read_text())
            d = json.loads((root / f'v2_eval_divl_step{step:04d}.json').read_text())
            for bucket in ['all', 'Near-ID', 'Mid', 'Far-OOD']:
                paired = paired_state_differences(s[bucket]['states'], d[bucket]['states'])
                assert len(paired) == 50
                count = {}
                for metric in ['rank', 'spearman']:
                    count[metric] = sum(math.isfinite(x[metric]) for x in paired)
                    independently_defined = sum(
                        l[metric] is not None and r[metric] is not None
                        for l, r in zip(s[bucket]['states'], d[bucket]['states'], strict=True))
                    assert count[metric] == independently_defined
                actual.append(dict(step=step, bucket=bucket, total_states=50,
                                   informative_counts=count))
    print(json.dumps(dict(passed=True, regression_cases=4, saved_results=actual), indent=2))


if __name__ == '__main__':
    main()
