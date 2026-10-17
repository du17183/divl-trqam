"""Pair all fixed states while preserving undefined ranking estimates."""
import math


def paired_state_differences(scalar, divl):
    rows = []
    for left, right in zip(scalar, divl, strict=True):
        assert left['state_id'] == right['state_id']
        assert left['task_id'] == right['task_id']
        assert left['pairs'] == right['pairs']
        assert left['candidates'] == right['candidates']
        row = dict(state_id=left['state_id'], task_id=left['task_id'])
        for metric in ['rank', 'spearman']:
            before, after = left[metric], right[metric]
            defined = (before is not None and after is not None
                       and math.isfinite(before) and math.isfinite(after))
            # No distinct-return pair (or undefined correlation) is not a zero
            # algorithm difference. Keep this state, with a missing estimate.
            row[metric] = after - before if defined else float('nan')
        rows.append(row)
    return rows
