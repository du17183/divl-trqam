"""Interpret completed v3 gates, retain uncertainty, and record actual throughput."""
import collections,json,pathlib,numpy as np
from action_sensitive_v3_common import R,D,O,atomic,sha


def main():
    manifest=json.loads((D/'manifest.json').read_text());audit=json.loads((O/'v3_action_supervision.json').read_text())
    control=json.loads((R/'artifacts/v3_positive_control.json').read_text());protocol=json.loads((D/'protocol.json').read_text())
    selected=json.loads((D/'state_selection_manifest.json').read_text())['records']
    sources=json.loads((D/'sources_progress.json').read_text());branches=json.loads((D/'branches_progress.json').read_text())
    state_report=R/'reports/v3_state_selection_report.md'
    appendix='\n\n## Actual selection audit\n\n'+json.dumps(dict(
        states=len(selected),unique_initial_ids_per_task={str(t):len({r['initial_state_id'] for r in selected if r['task_id']==t}) for t in range(10)},
        assigned_phases=dict(collections.Counter(r['assigned_phase'] for r in selected)),achieved_phases=dict(collections.Counter(r['achieved_phase'] for r in selected)),
        fallbacks=sum(r['phase_fallback'] for r in selected),
        no_future_or_branch_label_used=all(not r['future_return_used'] and not r['success_label_used'] and not r['branch_results_used'] for r in selected),
        selection_manifest_sha256=sha(D/'state_selection_manifest.json')),indent=2)
    assert '## Actual selection audit' not in state_report.read_text()
    state_report.write_text(state_report.read_text()+appendix)
    prior=json.loads((R/'artifacts/replay_action_audit_v3/replay_v2_statistics.json').read_text())['summary']
    prior_control=json.loads((R/'artifacts/replay_action_audit_v3/replay_v2_positive_control.json').read_text())['metrics']
    summary=audit['summary'];p=audit['perturbations'];total=manifest['outcomes']
    evidence=dict(v2=dict(states=prior['states'],mixed_outcome_ratio=prior['mixed_success_failure_states']/prior['states'],
        rank=prior_control['rank'],spearman=prior_control['spearman']),
        v3=dict(states=summary['states'],mixed_outcome_ratio=summary['mixed_success_failure_states']/summary['states'],
        rank=control['metrics']['all']['rank'],spearman=control['metrics']['all']['spearman']),
        note='Positive controls use different train datasets/statecounts; this difference is descriptive, not a paired v3-v2 effect estimate')
    source_pre=json.loads((O/'sources_preflight.json').read_text())['records'][0]
    branch_pre=json.loads((O/'branches_preflight.json').read_text())['records'][0]['records']
    new_source_steps=sum(r['source_steps'] for r in selected)-source_pre['source_steps']
    preids={r['episode_id'] for r in branch_pre}
    new_branch_rows=[r for r in manifest['records'] if r['episode_id'] not in preids]
    runtime=dict(inference_gpu_count=8,independent_sim_workers=16,
        normal_rollout_seconds=sources['elapsed_seconds'],branch_seconds=branches['elapsed_seconds'],
        normal_rollouts_per_minute=(len(selected)-1)*60/sources['elapsed_seconds'],
        branch_outcomes_per_minute=len(new_branch_rows)*60/branches['elapsed_seconds'],
        branch_states_per_minute=(manifest['states']-1)*60/branches['elapsed_seconds'],
        normal_env_steps_per_second=new_source_steps/sources['elapsed_seconds'],
        branch_env_steps_per_second=sum(r['steps'] for r in new_branch_rows)/branches['elapsed_seconds'],
        branch_file_bytes=sum(pathlib.Path(r['path']).stat().st_size for r in new_branch_rows),
        branch_output_bytes_per_second=sum(pathlib.Path(r['path']).stat().st_size for r in new_branch_rows)/branches['elapsed_seconds'],
        preflight_cache_excluded_from_execution_rates=True,
        gpu_utilization_samples=str(O/'runtime.jsonl'),
        attribution='nvidia-smi utilization/memory is GPU-wide and includes unrelated jobs; do not claim all GPU utilization belongs to our collection')
    atomic(O/'v3_runtime_summary.json',runtime)
    table='| 指标 | v3结果 |\n|---|---:|\n'
    table+=f'| 有效state/action outcomes | {summary["states"]}/{total} |\n'
    table+=f'| 恢复最大误差 | {summary["baseline_gate_max_error"]} |\n'
    table+=f'| 可排序state (>1e-6) | {summary["informative"]["1e-06"]["proportion"]:.2%} |\n'
    table+=f'| ΔG>1e-3 / >1e-2 state | {summary["informative"]["0.001"]["proportion"]:.2%} / {summary["informative"]["0.01"]["proportion"]:.2%} |\n'
    table+=f'| success/failure混合state | {summary["mixed_success_failure_states"]}/{summary["states"]} |\n'
    table+=f'| Near-ID / Mid / Far候选 | {summary["support_counts"]} |\n'
    allrank=control['metrics']['all']['rank'];near=control['metrics']['Near-ID']['rank']
    table+=f'| 正对照整体ranking | {allrank["mean"]:.2%} [{allrank["ci95"][0]:.2%},{allrank["ci95"][1]:.2%}] |\n'
    table+=f'| 正对照Near-ID ranking | {near["mean"]:.2%} [{near["ci95"][0]:.2%},{near["ci95"][1]:.2%}]；{near["states"]}states |\n'
    conclusion='# V3 round conclusion\n\n'+table+'\n## Decision\n\nSTOP_DATA_REDESIGN. 已有弱的、统计上高于随机的动作可学习信号；没有达到预注册整体ranking>=60%工程门槛。不能说完全随机，也不把Near-ID子集67%替代整体门槛。未启动任何v3正式Scalar/DIVL训练、QAM、Actor或dynamic beta。所有旧数据/新分支均保留。\n\n'
    conclusion+='## Evidence and next design\n\n'
    conclusion+=f'1. Near-ID仅{summary["support_counts"]["Near-ID"]/total:.1%}。small的Near+Mid为{(p["small"]["support_counts"]["Near-ID"]+p["small"]["support_counts"]["Mid"])/p["small"]["outcomes"]:.1%}；medium为{(p["medium"]["support_counts"]["Near-ID"]+p["medium"]["support_counts"]["Mid"])/p["medium"]["outcomes"]:.1%}；large的Far为{p["large"]["support_counts"]["Far-OOD"]/p["large"]["outcomes"]:.1%}。下一轮优先预先定义phase/local replay sigma，保留同样.25/.5/1倍水平和所有结果，不按G筛state。这里是固定公共feature近邻的经验support代理，不是已证明的物理state-action OOD。\n\n'
    conclusion+=f'2. small/medium的|G-G_behavior|中位数分别{p["small"]["absolute_delta_G"]["p50"]:.6g}/{p["medium"]["absolute_delta_G"]["p50"]:.6g}；不少监督仍为成功时间差。large立即失败比例为{p["large"]["immediate_failure_ratio"]:.1%}，不能说large全部立即失败。需按phase与真实动作channel设计下一轮对称干预，维持native reward/gamma/controller不变。\n\n'
    conclusion+='3. 冻结mean-pooled SigLIP+8Dproprio只是一个正对照表示；其失败不证明完整π0.5token表示或正式critic不可学习。可在独立预注册诊断中比较冻结policy token embedding；不能事后挑最好的模型结果冒充本轮primary。当前v3正式critic没有训练，architecture根因尚未确定。\n\n'
    conclusion+='4. 训练initialID覆盖增加到30/task，但5fold整体结果不稳定，只有1fold达到60%。不增加critic预算、不只调DIVL、不自动采几万条。下一轮先封存新protocol，沿用train-only与现有frozen50/final隔离。\n\n'
    conclusion+='## V2 versus V3 positive controls\n\n'+json.dumps(evidence,indent=2)+'\n\n## Actual runtime\n\n'+json.dumps(runtime,indent=2)
    (R/'reports/v3_round_conclusion.md').write_text(conclusion,encoding='utf-8')
    atomic(O/'v3_round_complete.json',dict(complete=True,decision='STOP_DATA_REDESIGN',v3_states=300,v3_outcomes=1500,
        actor_updated=False,formal_critics_trained=False,original_input_hashes_unchanged=all(sha(p)==h for p,h in protocol['frozen_input_hashes'].items())))
    print('V3_ROUND_REPORT_COMPLETE',json.dumps(runtime),flush=True)


if __name__=='__main__':main()
