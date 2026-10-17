# Method and experiment map

| Method | Critic | Actor update |
| --- | --- | --- |
| B | none | BC; EMA is parameter averaging on the same trajectory |
| C / D | Scalar / DIVL | QAM |
| E / F | Scalar / DIVL | BC + QAM |
| G / H | Scalar / DIVL | TRQAM with independently optimized slow BC β and EMA reference |

R0 denotes the original Rollout main pool. R3 denotes expert + Rollout main replay.
The BC supervision pool is distinct from the main RL pool and contains only expert/successful actions.
The main sampler is task-uniform, then trajectory-uniform within task, then chunk-uniform within trajectory,
with replacement. There is no main-pool expert source quota.

DIVL learns a state-conditioned distribution over replay action values. A quantile supplies the bootstrap statistic
for scalar Q learning. QAM/TRQAM then use Q action gradients to form flow-model supervision.
A good value fit does not automatically establish accurate local action derivatives.

Native Critic transitions use the actually executed H5 prefix; BC uses real cross-chunk H10 action labels.
Padding dimensions and plan lengths must not be silently substituted for executed actions.
Complete MC and censored observed return are tracked separately. Timeout is not automatically a terminal failure
for TD bootstrapping; follow the preserved transition contract.

## Known findings and limits

- Mixed DIVL same-state ranking: 54.97%; Near-ID ranking 49.52% versus original 80.12%.
- Mixed DIVL gradient-minus-random observed-return gain: +0.00819, nominal CI [-0.00294, +0.02111].
- Mixed H-Cosine500 maximum raw path KL: 0.405117, budget 1.0, no over-budget updates.
- Matched BC and TRQAM results do not yet establish a stable RL advantage.
- β alone and its EMA reference were not separately evaluated in closed loop.
- Changing both Critic and main Actor replay prevents causal attribution to one component.

These are measured limitations of this experiment, not a proof that either method is ineffective.
