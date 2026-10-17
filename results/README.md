# Aggregate experimental results

These are aggregate CSV tables from the accepted final experiment snapshot.
They do not contain raw episodes, images, trajectories or model parameters.
Source snapshot SHA256: `ee738cb74c835185a8ea8ba60073bfafbd70e87fa236d3bae93d29da9f5445c6`.

All checkpoint comparisons refer to the independent Cosine500 trajectory.
Earlier Cosine100 runs use a different schedule and must not be equated to Cosine500 at step100.
See the root README and docs for single-seed/development-set and confidence-interval limitations.

## Per-task checkpoint metrics

[`per_task_metrics.csv`](per_task_metrics.csv) contains 90 aggregate task rows:
H-R3 DIVL+TRQAM, BC parameters and BC EMA × 100/250/500 updates × 10 tasks.
Each row summarizes 50 episodes. `mean_return` uses the undiscounted binary terminal success reward;
`mean_episode_length` averages all episodes, including failures. No raw episodes or trajectories are included.
Percentages, returns and lengths are formatted to 1, 3 and 2 decimal places, respectively.
See the root README for settings, paired confidence intervals and interpretation limits.

## Diagnostic evidence for the README analysis

- [`critic_validation_summary.csv`](critic_validation_summary.csv): four Scalar/DIVL × R0/R3 aggregate rows,
  including same-state/near-ID ranking, Spearman and positive-gradient-minus-random observed-return intervals.
  Ranking columns are fractions; return differences are return units, not percentage points.
- [`trqam_optimizer_summary.csv`](trqam_optimizer_summary.csv): six G/H run summaries derived from accepted
  update records, including applied/clipped update counts and post-update effective-lambda floor counts.
  The logged post-update lambda is distinct from the previous lambda used to compute that step.

These summaries contain no raw episodes or per-update logs. Proposed experiments in the README have not been run.
