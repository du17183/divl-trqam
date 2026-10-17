"""Thirty requested answers, actual completed artifacts only."""
import json,datetime
from action_sensitive_v3_common import R,D,O,atomic


def load(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None


def estimate(value,percent=False):
    if not value or value.get('mean') is None:return '未有可用估计'
    f=100 if percent else 1;unit='%' if percent else '';s=f'{value["mean"]*f:.4f}{unit}'
    if value.get('ci95'):s+=f'，95%CI [{value["ci95"][0]*f:.4f},{value["ci95"][1]*f:.4f}]{unit}'
    return s


def main():
    collection=load(O/'pipeline_status.json') or {};analysis=load(O/'analysis_pipeline_status.json') or {}
    manifest=load(D/'manifest.json');audit=load(O/'v3_action_supervision.json');control=load(R/'artifacts/v3_positive_control.json')
    summary=audit['summary'] if audit else {};perturb=audit['perturbations'] if audit else {}
    sources=load(D/'sources_progress.json') or {};branches=load(D/'branches_progress.json') or {}
    answers={i:'尚未完成；不得用旧结果填充。' for i in range(1,31)}
    valid=(manifest or {}).get('states');answers[1]=f'有效分叉状态：{valid if valid is not None else "待restore审计"}；正常rollout进度 {sources.get("completed",0)}/300；分叉进度 {branches.get("completed",0)}/300。'
    if manifest:
        counts={str(t):len({r['state_id'] for r in manifest['records'] if r['task_id']==t}) for t in range(10)}
        answers[2]=json.dumps(counts);answers[3]=str(manifest['outcomes'])
    if audit:
        answers[4]=json.dumps(summary['candidate_counts']);answers[5]=f'{summary["support_counts"].get("Near-ID",0)/summary["outcomes"]:.2%}；support完整计数 {summary["support_counts"]}'
        answers[6]=json.dumps(summary['informative']);answers[7]=f'{summary["mixed_success_failure_states"]}/{summary["states"]} = {summary["mixed_success_failure_states"]/summary["states"]:.2%}'
        answers[8]=json.dumps(summary['informative']);answers[9]=json.dumps({t:v['average_effective_pairs'] for t,v in summary['informative'].items()})
        answers[10]=json.dumps(perturb['small']);answers[11]=json.dumps(perturb['medium']);answers[12]=json.dumps(perturb['large'])+'。只报告分布；不按branch结果删除或调尺度。'
    if control:
        answers[13]=estimate(control['metrics']['all']['rank'],True);answers[14]=estimate(control['metrics']['all']['spearman'])
        answers[15]=f'工程正对照门槛通过={control["passed"]}；规则：{control["engineering_gate"]}。这不证明完整P(G|s,a)已被识别，也不代表最终LWD有效。'
        answers[16]='R1(75%原deployment+25%branch anchors)及R2(50%+50%)均预先固定；R1为主、R2为消融。'+('正对照未通过，两种正式critic均不训练。' if not control['passed'] else '两种mix分别运行同初始化/同顺序Scalar、DIVL；不通过held-out结果选择最好checkpoint。')
    evaluations={mix:load(O/f'v3_counterfactual_{mix}.json') for mix in ['R1','R2']}
    gradients={mix:load(O/f'v3_new_gradient_{mix}.json') for mix in ['R1','R2']}
    for method,n in [('scalar',17),('divl',18)]:
        reports=[]
        for mix,e in evaluations.items():
            if e:
                reports.append(mix+': '+', '.join(f'{step}={estimate(e["all_budgets"][method][str(step)]["all"]["rank"],True)}' for step in [250,500,1000,2000]))
            else:
                progress=load(O/f'critic_{mix}_{method}_v3_progress.json')
                reports.append(mix+': '+(f'{progress["step"]}/2000；尚未held-out评估' if progress else '未训练'))
        answers[n]='\n\n'.join(reports)
    prior=load(R/'artifacts/replay_action_audit_v3/v2_counterfactual_evaluation.json')
    for n,m in [(19,'scalar'),(20,'divl')]:
        answers[n]='\n\n'.join(mix+': '+estimate(e['all_budgets'][m]['2000']['all']['rank'],True) if e else mix+': 未评估' for mix,e in evaluations.items())
    answers[21]='\n\n'.join(mix+': '+estimate(e['paired_divl_minus_scalar']['2000']['all']['rank'],True)+'（差值单位为百分点）' if e else mix+': 未评估' for mix,e in evaluations.items())
    answers[22]='\n\n'.join(mix+': '+', '.join(m+'='+estimate(e['all_budgets'][m]['2000']['all']['spearman']) for m in ['scalar','divl']) if e else mix+': 未评估' for mix,e in evaluations.items())
    answers[23]='\n\n'.join(mix+': '+', '.join(m+'='+estimate(e['all_budgets'][m]['2000']['Near-ID']['rank'],True) for m in ['scalar','divl']) if e else mix+': 未评估' for mix,e in evaluations.items())
    actual=[(mix,g) for mix,g in gradients.items() if g]
    answers[24]='\n\n'.join(mix+': '+', '.join(m+'='+estimate(v['across_epsilon_plus_minus']) for m,v in g['methods'].items()) for mix,g in actual) if actual else '尚未模拟v3新critic方向；旧600branches只验证旧gradient，不重用其结果。'
    answers[25]='\n\n'.join(mix+': '+json.dumps({m:{k:v.get(k) for k in ['positive_contribution_share_top1','positive_contribution_share_top2','positive_contribution_share_top5','passed']} for m,v in g['methods'].items()}) for mix,g in actual) if actual else '尚无v3真实gradient贡献证据。'
    if prior:
        answers[26]='\n\n'.join(mix+': '+', '.join(f'{m}={(e["all_budgets"][m]["2000"]["all"]["rank"]["mean"]-prior["all_budgets"][m]["2000"]["all"]["rank"]["mean"])*100:+.3f}pp' for m in ['scalar','divl']) if e else mix+': 待v3评估' for mix,e in evaluations.items())
    for n,m in [(27,'scalar'),(28,'divl')]:
        answers[n]='\n\n'.join(mix+': '+json.dumps(dict(ranking_prerequisites=e['prerequisites'][m] if e else None,real_gradient=(g or {}).get('methods',{}).get(m)),ensure_ascii=False) for mix,e in evaluations.items() for g in [gradients[mix]])
    allowed=[(mix,m) for mix,g in actual for m,v in g['methods'].items() if v['passed']]
    answers[29]='完整gradient gate通过：'+str(allowed)+'。本轮尚未启动Actor；只有通过方法才具备后续fixed-reference pilot资格。' if allowed else '否；Actor/QAM、dynamic beta仍暂停。'
    answers[30]=('整体正对照未达到预设60%工程门槛；若CI下界>50%及Spearman为正，应表述为弱可学习信号，不能说完全随机或理论不可学习。优先检查Near-ID覆盖、phase/local扰动尺度及冻结embedding对空间/接触的表达；当前尚未训练v3正式critic，不能据此断言其architecture是根因。不增加正式updates或修改DIVL。' if control and not control['passed'] else
        '正对照通过而正式critic未达标时，检查共享action injection、7D/padding/mask、state dominance与optimization；不盲目扩大数据。' if control else '待同状态数据审计和正对照；不能提前归因。')
    titles=['实际branch states','每任务state数量','state-action outcomes','扰动数量','Near-ID比例','非零future差异比例','成功/失败混合比例','可排序状态比例','每state有效pairs','small的future差异','medium的future差异','large的OOD','正对照held-out ranking','正对照Spearman','是否可学习','replay mix','Scalar全部预算ranking','DIVL全部预算ranking','Scalar2000ranking/CI','DIVL2000ranking/CI','pairedDIVL-Scalar/CI','state-wiseSpearman','Near-IDranking','真实gradientΔG','gradient贡献集中度','相比v2','Scalar准入','DIVL准入','fixed-referenceActor资格','STOP后根因方向']
    status=analysis.get('stage') or collection.get('stage','准备中')
    text='# 当前阶段：Action-Sensitive Replay v3\n\n'+datetime.datetime.now().astimezone().isoformat()+f'；b300-1 `{R}`。\n\n状态：**{status}**；Actor冻结，dynamic beta关闭。\n\n'
    for i,title in enumerate(titles,1):text+=f'## {i}. {title}\n\n{answers[i]}\n\n'
    text+='## 解释边界\n\n保留原200replay、v2全部数据/checkpoints、50devstates/400候选/600旧方向；final IDs未使用。v3仅train30初态/task。真实reward/controller/normalization/actor均未改变；phase使用当前接触/抓取代理，不使用branch未来结果。全部候选保留，restore失败独立debug。最终LIBERO_Long策略success结果尚未获得，不能把critic排序当作策略成功率。\n'
    for filename,content in [('current_stage_summary.md',text),('v3_replay_mix_report.md', '# V3 fixed replay mixes\n\n'+answers[16]+'\n\nFormal branch samples are uniform among all v3 state-action anchors; full continuation is retained for nativeTD10targets. GlobalB256: R1=192deployment+64branch; R2=128+128. No extra ranking loss added to either formalcritic.\n')]:
        (R/'reports'/filename).write_text(content,encoding='utf-8')
    for method in ['scalar','divl']:
        report='# '+method+' v3 critic\n\n'+answers[17 if method=='scalar' else 18]+'\n\n'
        for mix in ['R1','R2']:
            path=R/'reports'/f'{method}_v3_{mix}_critic_report.md'
            if path.exists():report+='\n'+path.read_text()
        (R/'reports'/f'{method}_v3_critic_report.md').write_text(report)
    heldout='# V3 fixed held-out50 comparison\n\nOriginalnative400actions andtruefutureG, unchangedtrain-derivedsupport bins. Primary2000, all budgets retained. Actorstillfrozen.\n\n'+answers[21]+'\n\n'+answers[26]+'\n\n'
    old=load(R/'artifacts/replay_action_audit_v3/ood_analysis.json')
    heldout+='## Primary2000 versions side by side\n\n| Version | Method | All ranking/CI | Spearman/CI | Near-ID ranking/CI |\n|---|---|---|---|---|\n'
    for method in ['scalar','divl']:
        if old:
            v=old['ranking'];heldout+=f'| v1 | {method} | {estimate(v["all"][method]["rank"],True)} | {estimate(v["all"][method]["spearman"])} | {estimate(v["Near-ID"][method]["rank"],True)} |\n'
        if prior:
            v=prior['all_budgets'][method]['2000'];heldout+=f'| v2 | {method} | {estimate(v["all"]["rank"],True)} | {estimate(v["all"]["spearman"])} | {estimate(v["Near-ID"]["rank"],True)} |\n'
        for mix,e in evaluations.items():
            if e:
                v=e['all_budgets'][method]['2000'];heldout+=f'| v3 {mix} | {method} | {estimate(v["all"]["rank"],True)} | {estimate(v["all"]["spearman"])} | {estimate(v["Near-ID"]["rank"],True)} |\n'
            else:heldout+=f'| v3 {mix} | {method} | 未评估 | 未评估 | 未评估 |\n'
    heldout+='\nComparisons are descriptive: v2 changed replay andTD horizon together, and moved from1GPUto4GPU BF16; v3 retains the v2 objective/precision/device count and changes traincoverage/action supervision/replaymix. SparseNear-ID states limit inference. No finalsplit or policy-success result is used here.\n\n'
    for mix in ['R1','R2']:
        path=R/'reports'/f'v3_heldout_{mix}.md'
        if path.exists():heldout+='\n'+path.read_text()
    (R/'reports/v3_heldout_counterfactual_report.md').write_text(heldout)
    atomic(R/'artifacts/v3_critic_metrics.json',dict(stage=status,mixes=evaluations,gradient=gradients,actor_updates=False))
    print('V3_STAGE_REPORT_UPDATED',status,flush=True)


if __name__=='__main__':main()
