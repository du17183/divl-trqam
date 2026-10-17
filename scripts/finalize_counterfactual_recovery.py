# Public release: deployment paths anonymized; algorithm equations unchanged.
"""Publish the requested twenty answers from completed, gated real branches."""
from datetime import datetime,timezone,timedelta
import hashlib
import json
from pathlib import Path

ROOT=Path('/opt/divl-trqam')

def load(name):return json.loads((ROOT/'artifacts'/name).read_text())
def metric(d):return f"mean={d['mean']}, median={d['median']}, 95% state-bootstrap CI={d['ci95']}, n_states={d['n_states']}"
def epsilon_summary(method):
    return '\n\n'+ '\n'.join(f"- ε={eps}: "+metric(d)+f"; P(ΔG>0)={d['positive_ratio']}; ties={d['zero_ratio']}" for eps,d in method['by_epsilon'].items())

def main():
    gate=load('reproduction_audit.json');rank=load('action_ranking_metrics.json')
    grad=load('gradient_direction_metrics.json');align=load('q_alignment_audit.json')
    neighbors=load('frozen_state_future_neighbors.json')
    backup=load('counterfactual_pre_fix_manifest.json')
    assert gate['passed'] and align['alignment_passed'] and backup['backup_complete']
    assert rank['states']==50 and rank['candidate_branches']==400 and grad['gradient_branches']==600
    methods=rank['methods'];directions=grad['methods'];paired=rank['paired_comparison']
    delta=paired['ranking_divl_minus_scalar']
    if delta['ci95'][0]>0:stronger='DIVL has stronger paired action-ranking evidence'
    elif delta['ci95'][1]<0:stronger='Scalar has stronger paired action-ranking evidence'
    else:stronger='No reliable ordering between Scalar and DIVL; paired ranking CI crosses zero'
    eligible=[m for m,g in rank['engineering_gate'].items() if g['passed']]
    pairs=[
      ('反事实 restore 不一致的根因',gate['root_cause']),
      ('修复的具体状态/字段','task01_init018_rep0_chunk009: preserve original free-joint quaternion values after forward; task03_init009_rep0_chunk017: recover exact cabinet/wine-rack model poses with full physical fingerprint proof. Version2 applies primary/integration/controller/cache/RNG and fixture restoration to the complete fixed50 inventory.'),
      ('20次重复 restore 最大误差',json.dumps(gate['maximum_errors'])+'; each of five diverse states repeated20 times; every repeated measured trace must match repeat0 exactly.'),
      ('原35个 state 是否重跑、原因','YES. All50 states and all1000 branches rerun under V2 because primary/model-state restoration changed. Old700 branches remain untouched and provisional; backup='+str(ROOT/'backups')),
      ('最终有效 counterfactual states',str(rank['states'])+' /50 fixed; no hard/invalid state silently removed'),
      ('candidate branches',str(rank['candidate_branches'])+'; plus600 gradient branches'),
      ('Scalar pairwise ranking',metric(methods['scalar']['state_macro_ranking'])),
      ('DIVL pairwise ranking',metric(methods['divl']['state_macro_ranking'])),
      ('paired DIVL−Scalar 排序差值',metric(delta)+'; median95%CI='+str(delta['median_ci95'])),
      ('Scalar state-wise Spearman',metric(methods['scalar']['state_wise_spearman'])),
      ('DIVL state-wise Spearman',metric(methods['divl']['state_wise_spearman'])),
      ('Scalar 梯度方向真实 ΔG',metric(directions['scalar']['across_epsilon_state_mean'])+epsilon_summary(directions['scalar'])+'\n\nScalar ε=.01 has a small positive interval, but the fixed primary aggregate across all three eps crosses0 and ranking/Spearman also fail. No epsilon was selected after results.'),
      ('DIVL 梯度方向真实 ΔG',metric(directions['divl']['across_epsilon_state_mean'])+epsilon_summary(directions['divl'])+'\n\nInfluence: '+json.dumps(directions['divl']['influence_diagnostic'])+'. Its positive mean must not be treated as broad reliable improvement.'),
      ('P(ΔG>0)','Across-epsilon state mean: '+str({m:d['across_epsilon_state_mean']['positive_ratio'] for m,d in directions.items()})+'; individual eps positive/zero ratios retained in gradient_direction_metrics.json'),
      ('负跨状态 Pearson 是否存在 alignment bug','No target/state/action alignment bug observed in the fixed random audit of10 success+10 failure episodes,1570 chunks. '+json.dumps(align['max_errors'])+'. This is a sampled implementation audit, not proof of every possible input.'),
      ('若未发现 bug，如何解释负 Pearson','Original100-state diagnostic contains96 successful and4 failed states. Success-only Pearson is positive (.410828 Scalar/.275622 DIVL), while failures are overestimated; task05_init035_rep0_chunk062 has G=0 but Q=.750994/.600730. Group composition and rare failed-state miscalibration drive the negative overall correlation. No points were removed from primary analysis.'),
      ('learned critic 是否依赖 action','Action shuffle relative to totalQ std: '+str({m:d['shuffle_relative_q_std'] for m,d in methods.items()})+'. Nonzero dependence alone does not prove correct ranking or causal gradient direction; use the true branch metrics above.'),
      ('谁的动作排序证据更强',stronger),
      ('是否满足 QAM Actor 准入','Eligible critics='+str(eligible)+'; decision='+rank['decision']+'. '+json.dumps(rank['engineering_gate'])+'; engineering60% reference is not a paper significance threshold.'),
      ('最终继续 QAM 或停止诊断','Diagnostic evidence permits considering a separately authorized next actor stage' if eligible else 'STOP_ACTOR_QAM: retain frozen actor and continue critic/replay diagnosis; no formal Scalar/DIVL QAM, dynamic beta, PPO or factorial was launched.')
    ]
    assert len(pairs)==20
    text='# Current stage summary / 当前阶段20项结论\n\nUpdated Asia/Shanghai: '+datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=8))).isoformat()+'\n\n'
    text+='## Decision\n\n'+rank['decision']+'\n\nNo formal actor update was performed during this recovery/action-evidence stage.\n\n'
    text+='## 核心结果（固定2000-step critic）\n\n| Method | 同状态排序均值 | 95% CI | State-wise Spearman | 真实ΔG均值 | P(ΔG>0) |\n|---|---:|---|---:|---:|---:|\n'
    for m,d in methods.items():
        rv=d['state_macro_ranking'];gv=directions[m]['across_epsilon_state_mean']
        text+=f"| {m} | {100*rv['mean']:.2f}% | [{100*rv['ci95'][0]:.2f}%, {100*rv['ci95'][1]:.2f}%] | {d['state_wise_spearman']['mean']:.5f} | {gv['mean']:.6f} | {100*gv['positive_ratio']:.1f}% |\n"
    text+='\n50/50状态通过恢复与分叉；48状态有可排序的非平局动作，共1151有效pair。其余2状态全候选失败，仍保留在数据和梯度统计中。Bootstrap采样单位为state，非1151个pair。\n\n'
    text+='DIVL−Scalar排序差值：'+f"{100*delta['mean']:.2f}个百分点，95% paired CI=[{100*delta['ci95'][0]:.2f}, {100*delta['ci95'][1]:.2f}]个百分点。当前无可靠方法优劣证据。\n\n"
    for n,(title,answer) in enumerate(pairs,1):text+=f'## {n}. {title}\n\n{answer}\n\n'
    text+='## Protocol and limits\n\nFixed50 dev diagnostic states, five per task; eight frozen candidates/state plus Scalar/DIVL ± directions at epsilon .01/.03/.05.5000 paired whole-state bootstrap resamples stratified bytask. Returns with differences≤1e-6 count as ties and are excluded from ranking. All250/500/1000/2000 critic budgets reported;2000 is the fixed primary budget. No final-split checkpoint selection.\n\n'
    text+='Original per-step simulator states were not recorded historically: original action/reward streams and chunk state/controller/observation anchors are compared; new per-step repetitions are retained and compared exactly. Original snapshots did not record Torch RNG; simulator has no initialized CUDA RNG and frozen policy uses an explicit saved flow-noise seed, with20 identical-input inference outputs verified identical. No missing historical fields are fabricated.\n\n'
    text+='Official500 experts remain excluded because initial-state provenance is unresolved. Train/dev/final IDs remain30/10/10 per task. Earlier disposable feasibility ID7 overlaps5train/2dev/3final IDs, preserved in provenance. Officialpi05 pretraining provenance is not claimed disjoint. Neither source/home/zwl/openpi nor actor/controller/reward/normalization was changed.\n\n'
    neighbor_summary={k:v for k,v in neighbors.items() if k not in ['predictions','examples','unconditional_nearest5']}
    neighbor_summary['unconditional_nearest5']={k:v for k,v in neighbors['unconditional_nearest5'].items() if k!='records'}
    text+='## Frozen-encoder future diversity\n\n'+json.dumps(neighbor_summary,indent=2)+'\n\nObservational neighbors are distinct states and cannot establish causal P(G|s,a); same-snapshot branch results are the action evidence.\n'
    text+='\n## 下一步建议\n\n保持actor冻结。分别检查成功/失败动作反差与成功时机排序，不以整体return相关或较低TD loss替代动作证据；评估现有train replay对反差动作的覆盖、动作输入在critic中的作用和稀有失败状态的校准。若后续构造critic训练用反事实数据，必须来自train split并保留独立dev评估，本轮dev分叉不能直接回灌训练。任何新增训练属于下一阶段。\n'
    (ROOT/'reports/current_stage_summary.md').write_text(text)
    required=['reproduction_debug_report.md','reproduction_gate_report.md','counterfactual_final_report.md',
      'critic_action_ranking_report.md','qam_gradient_direction_report.md','q_return_alignment_audit.md',
      'nearest_future_diversity_report.md','current_stage_summary.md']
    files=[ROOT/'reports'/name for name in required]+[ROOT/'artifacts'/name for name in ['reproduction_audit.json','counterfactual_final.parquet','action_ranking_metrics.json','gradient_direction_metrics.json','q_alignment_audit.json']]
    inventory=[]
    for p in files:
        assert p.is_file() and p.stat().st_size>0,str(p)
        inventory.append({'path':str(p),'size':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    output={'completed_at_utc':datetime.now(timezone.utc).isoformat(),'decision':rank['decision'],'actor_updated':False,
            'states':50,'candidate_branches':400,'gradient_branches':600,'files':inventory}
    (ROOT/'artifacts/counterfactual_recovery_delivery.json').write_text(json.dumps(output,indent=2))
    print('RECOVERY_REPORTS_COMPLETE',json.dumps(output),flush=True)

if __name__=='__main__':main()
