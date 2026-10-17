"""Clarify report interpretation using completed measurements; no model/rollout work."""
import json
import re
from horizon_common import R, O, REP, report, atomic

sens = json.loads((R / 'artifacts/action_horizon_metrics.json').read_text())
rec = json.loads((R / 'artifacts/recovery_curve_metrics.json').read_text())
probe = json.loads((R / 'artifacts/horizon_positive_control.json').read_text())

interpretation = (
    '**平均恢复曲线不支持“短扰动在下一次 replan 被迅速拉回”的解释。** '
    'H1/H3/H5 的 EEF 平均位置偏差在第 1 次 replan 后均高于干预结束时，'
    '前 3 次 replan 的平均偏差继续增加。达到恢复阈值的比例较高，'
    '只能说明许多分支偏差仍小；一些分支在干预结束时就已接近原行为，'
    '不能将 recovered-only 的平均时间理解为主动纠错速度。'
    '后续观测受终止删失及可比较样本组成变化影响；没有 open-loop 对照，'
    '不能排除个别轨迹的纠错，也不能把弱 return 差异归因于已证实的强策略纠错能力。'
)
next_steps = (
    '## 下一步建议（本轮不执行）\n\n'
    '1. 保持 STOP_ACTOR_QAM，不扩大同类 v5/v6 局部扰动采集。\n'
    '2. 用一个小规模、预注册实验检验已有不同质量 behavior 或更长连续行为偏移，'
    '保持 reward/controller/normalization，整条候选结果全部保留；先检验是否增加可学习的动作回报差异。\n'
    '3. 对少量同状态候选重复 frozen continuation，分别估计动作效应和续跑随机性；'
    '没有该对照前不宣称哪个机制已被证实。\n'
    '4. 明确后续 critic 的估计目标：冻结策略完整 MC return 与 Scalar IQL '
    'expectile/DIVL adaptive-quantile bootstrap 不同。按算法定义诊断目标，'
    '不能按结果最好看的 target 换奖励或推进 QAM。\n'
    '5. 只有新的同状态 held-out 动作监督与真实 critic 梯度验证通过后，'
    '再决定正式 Scalar/DIVL 训练；本轮小 MLP 不构成 QAM 的启动许可。'
)

# Preserve all 34 answers; make the recovery answer explicit.
summary = (REP / 'current_stage_summary.md').read_text()
summary, count = re.subn(
    r'(## 23\. [^\n]+\n\n).*?(?=\n## 24\.)',
    lambda m: m.group(1) + interpretation + '\n',
    summary, flags=re.S,
)
assert count == 1
summary += '\n' + next_steps + '\n'
for path in [REP / 'current_stage_summary.md', R / 'reports/current_stage_summary.md']:
    path.write_text(summary)

lines = [interpretation, '\n## 平均偏差与恢复阈值\n',
    '| Horizon | EEF：干预结束 / replan1 / replan2 / replan3（mm） | 三次 replan 内达阈值 | 达阈值样本平均 replan 数 | 全部样本截断平均数 | 未恢复或删失 |',
    '|---|---|---:|---:|---:|---:|']
for h in ['1', '3', '5']:
    curve = rec['curves'][h]
    vals = [curve[k]['EEF_position_L2']['mean'] * 1000
            for k in ['intervention_end', 'replan1', 'replan2', 'replan3']]
    t = rec['recovery_times'][h]
    lines.append(f"| H{h} | {' / '.join(f'{v:.3f}' for v in vals)} | "
                 f"{100*t['recovered_by_replan3_fraction']:.2f}% | "
                 f"{t['mean_replans_among_recovered']:.3f} | "
                 f"{t['restricted_mean_replans3']:.3f} | "
                 f"{t['unresolved_or_censored']}/400 |")
lines += [
    '| H-QAM | 同 H5 | 同 H5 | 同 H5 | 同 H5 | 同 H5 |',
    '\n恢复定义：连续两次 replan 时所有物理偏差均在预注册阈值内。'
    '阈值为 proprio L2≤0.02，EEF/对象平移≤5 mm，旋转≤2°。'
    '“全部样本截断平均”将未恢复样本计为 3，并将更迟恢复截在 3；不是无删失的真实恢复时间。',
    '\n比较使用相同 elapsed env step，不在任一轨迹终止后外推。'
    'terminal 是每条轨迹的终止比较点，不是等长的固定时间间隔。'
    '各点可用样本数、五项偏差、state bootstrap 95% CI 全部保存在 '
    '`artifacts/recovery_curve_metrics.json` 和 `recovery_samples.parquet`；'
    '视觉 embedding 曲线未计算。',
    '\n![Recovery curve](policy_recovery_curve.png)',
]
body = '\n'.join(lines)
report('policy_recovery_curve.md', 'Policy recovery curves', body)
root = R / 'reports/policy_recovery_curve.md'
root.write_text(root.read_text().replace('(policy_recovery_curve.png)',
    '(action_horizon_alignment_20261008/policy_recovery_curve.png)'))

lines = [
    '每个 horizon 使用自己的 behavior 基线。100 个配对 train states，每档 500 条分支，'
    '其中 400 条为扰动分支。以下 ΔG 指扰动相对同档 behavior 的完整 native discounted return 差值。',
    '\n| H | Mean abs(ΔG) | Median abs(ΔG) | P>1e−4 | P>1e−3 | P>1e−2 | Success flip |',
    '|---|---:|---:|---:|---:|---:|---:|',
]
for h in ['1', '3', '5']:
    s = sens['horizons'][h]
    p = s['probabilities']
    lines.append(f"| H{h} | {s['mean_absolute_delta_G']:.8f} | "
        f"{s['median_absolute_delta_G']:.8f} | {100*p['0.0001']:.2f}% | "
        f"{100*p['0.001']:.2f}% | {100*p['0.01']:.2f}% | "
        f"{100*s['success_flip_rate']:.2f}% |")
lines += ['| H-QAM | 同 H5 | 同 H5 | 同 H5 | 同 H5 | 同 H5 | 同 H5 |',
    '\n## 配对统计\n',
    '| 比较 | Mean abs(ΔG) 差 / 95% CI | P(abs(ΔG)>0.01) 差 / 95% CI |',
    '|---|---|---|']
for name, v in sens['paired'].items():
    a = v['mean_abs_delta']; b = v['probability_delta_gt_01']
    lines.append(f"| {name} | {a['mean']:.6f} [{a['ci95'][0]:.6f}, {a['ci95'][1]:.6f}] | "
        f"{100*b['mean']:.2f} pp [{100*b['ci95'][0]:.2f}, {100*b['ci95'][1]:.2f}] pp |")
lines += [
    '\n均值主要由少数 success flip 驱动，中位差约 1e−4。'
    '干预更长使极小差异出现比例增加，但较大回报差异和排序能力没有系统提高。',
    '\nH1/H3 因更早 replan，零扰动行为本身也不同于原 H5；'
    '所以 source-H5 return reproduction 对 H1/H3 是 N/A。'
    '所有档位独立 behavior 重复的最大 restore 误差为 0；H5 对原记录的最大误差也为 0。',
    '\n每个 horizon 采用相同 snapshot、方向、逐步幅度和 continuation call ordinal 随机数。'
    '总扰动预算随 H 增加；改变首次 replan 时点是干预的一部分。'
    '所有候选/相同回报状态保留，5-fold 按整状态及 initial ID 分组。',
    '\n当前 0.75% 是 H5 的 400 条扰动分支相对 behavior 的比例；'
    '旧 v4 3.18% 是 Near-ID 内候选 pair 的比例，分母不同，不能直接比较。'
    '本轮全部候选 pair 的 P(abs(ΔG)>0.01)：H1 1.2%，H3 3.2%，H5 1.3%。',
    '\n完整逐状态结果、restore 项和 CI 见 `artifacts/action_horizon_metrics.json`；'
    '逐分支轨迹索引见 `artifacts/action_horizon_branches.parquet`。',
]
report('action_horizon_return_sensitivity.md', 'Future return sensitivity', '\n'.join(lines))

for path in [REP / 'action_horizon_conclusion.md', R / 'reports/action_horizon_conclusion.md']:
    path.write_text(path.read_text() + '\n\n## 恢复机制的证据边界\n\n' + interpretation + '\n\n' + next_steps + '\n')

atomic(O / 'report_interpretation.json', dict(
    recovery_contraction_supported_by_mean_curve=False,
    high_threshold_recovery_is_not_active_correction_evidence=True,
    original_H5_equivalence_for_H1_H3='not applicable; independent same-H behavior repeats used',
    scalar_DIVL_target_association_is_not_current_Q_action_ranking=True,
    next_steps_executed=False,
    data_measurements_unchanged=True,
))
print('REPORT_INTERPRETATION_CLARIFIED; NO_COLLECTION_OR_TRAINING')
