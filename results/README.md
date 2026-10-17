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
