# Public release: deployment paths anonymized; algorithm equations unchanged.
"""Truthful running/final20-answer summary; never invent unfinished gates."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,re
R=Path('/opt/divl-trqam');O=R/'artifacts/replay_action_audit_v3';D=R/'data/action_sensitive_replay_v2';P=R/'reports'
def load(p):
    try:return json.loads(p.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None
def estimate(v,percent=False):
    if not v or v.get('mean') is None:return '尚无可用估计'
    factor=100 if percent else 1;unit='%' if percent else '';ci=v.get('ci95',[None,None])
    text=f'{v["mean"]*factor:.4f}{unit}'
    if ci[0] is not None:text+=f'，95% CI [{ci[0]*factor:.4f}, {ci[1]*factor:.4f}]{unit}'
    return text
def main():
    old=(P/'current_stage_summary.md').read_text();sections=re.split(r'(?m)^## (\d+)\. ',old);answers={}
    for i in range(1,len(sections),2):answers[int(sections[i])]=sections[i+1].strip()
    progress=load(D/'progress.json') or {};pipeline=load(O/'pipeline_status.json') or {};data=load(O/'replay_v2_data_gate.json');control=load(O/'replay_v2_positive_control.json');evaluation=load(O/'v2_counterfactual_evaluation.json');gradient=load(O/'v2_new_gradient_metrics.json')
    collection=f'{progress.get("states",0)}/1000 states，{progress.get("outcomes",0)}/5000 outcomes，{progress.get("status","尚未启动")}'
    if progress.get('outcomes_per_minute'):collection+=f'，{progress["outcomes_per_minute"]:.2f} outcomes/min'
    state=pipeline.get('decision',pipeline.get('stage','采集新 replay'));actorallowed=bool(gradient and any(v['passed'] for v in gradient['methods'].values()))
    stamp=datetime.now(timezone(timedelta(hours=8))).isoformat();text=f'# 当前阶段完整进度\n\n更新时间：{stamp}；b300-1 `/opt/divl-trqam`。\n\n状态：**{state}**。\n\n新 replay：{collection}。\n\n'
    for n in range(1,14):text+=f'## {n}. '+answers[n]+'\n\n'
    text+='## 14. 是否构造 action-sensitive replay v2？\n\n是，train-only 固定1000个时间点×5个真实执行候选。没有使用50-state dev diagnostic训练，没有改原数据。10个唯一train初态/task×2rep，每轨迹5时间点；不是20个唯一train初态/task。\n\n'
    text+='## 15. v2 state-action outcomes 数量\n\n'+collection+'。以 `data/action_sensitive_replay_v2/manifest.json` 完整冻结值为准。\n\n'
    text+='## 16. v2 是否真的增加局部 action/future diversity？\n\n'
    if data:
        s=data['summary'];text+=f'{s["different_future_states"]}/1000状态有非平局真实G；{s["mixed_success_failure_states"]}/1000状态混合成功/失败；{s["successes"]}/5000候选成功；support={s["support_counts"]}。baseline最大误差={s["baseline_gate_max_error"]}；同组actor-visible RGB/proprio完全一致。data gate={data["passed"]}。\n\n'
        text+='future variance / return range：\n\n'+json.dumps(dict(future_variance=s['future_variance'],return_range=s['return_range']),indent=2)+'\n\n'
        if control:text+='新数据简单MLP（整个initial-ID分组的5-fold CV） held-out rank：'+estimate(control['metrics']['rank'],True)+'；Spearman：'+estimate(control['metrics']['spearman'])+'。\n\n'
        text+='必须区分完成时间差与success/failure分叉；1次固定续跑noise只得到一个真实未来样本，不等于完整估计P(G|s,a)。公共mean-pooled视觉MLP只是诊断，不能代替可训练token encoder。若同状态数据门槛通过，执行一次固定2000-update共享修复；若正对照及v2仍失败，停止扩大数据。\n\n'
    else:text+='完整统计待采集结束。10-state pilot中7个不同G、多为时间差、无新增混合success/failure；不能提前证明完整v2已产生足够监督。\n\n'
    for number,method in [(17,'scalar'),(18,'divl')]:
        text+=f'## {number}. {method.upper()} v2 ranking\n\n'
        training=load(O/f'critic_{method}_v2_progress.json');complete=load(O/f'critic_{method}_v2_complete.json')
        if evaluation:
            value=evaluation['all_budgets'][method]['2000'];text+='固定2000更新，全candidate：'+estimate(value['all']['rank'],True)+'；Spearman：'+estimate(value['all']['spearman'])+'；Near-ID：'+estimate(value['Near-ID']['rank'],True)+'。\n\n'
            text+='所有预算250/500/1000/2000都保留；没有挑最好checkpoint。新actor准入前置条件：'+json.dumps(evaluation['prerequisites'][method],ensure_ascii=False)+'。\n\n'
        elif training:
            text+=f'已更新 {training["step"]}/2000；held-out动作排序尚未计算。不能把Q loss或TD error当作排序通过。\n\n'
            execution=training.get('execution',{})
            if execution:text+='实际训练资源：'+json.dumps(execution,ensure_ascii=False)+f'；最近一批吞吐 {training["latest"].get("samples_per_second",0):.2f} samples/s。\n\n'
        else:text+='尚未训练 / 尚无v2排序结果。\n\n'
        if complete:text+=f'训练耗时 {complete["elapsed_seconds"]/60:.1f}分钟；最终metric：'+json.dumps(complete['final'],ensure_ascii=False)+'。\n\n'
        if training and not (P/f'{method}_v2_critic_report.md').exists():
            (P/f'{method}_v2_critic_report.md').write_text(f'# {method} v2 critic — running\n\n'+json.dumps(training,indent=2)+'\n\nMatched2000 budget, shared Replay and originalLIBERO10-chunk TD; actor frozen. Held-out quality not yet evaluated.\n')
    text+='## 19. gradient 真实 ΔG 是否由广泛 state 贡献？\n\n'
    if gradient:
        for m,v in gradient['methods'].items():text+=f'{m}：plus-minus '+estimate(v['across_epsilon_plus_minus'])+'；plus-baseline '+estimate(v['across_epsilon_plus_baseline'])+f'；positive states={v["across_epsilon_plus_minus"]["positive_ratio"]:.2%}；top2正贡献share={v["positive_contribution_share_top2"]:.2%}；gate={v["passed"]}。\n\n'
    else:text+='本轮尚无新critic真实梯度收益；旧DIVL的99%正贡献由2状态提供，旧总体CI跨0。旧600方向与结果全部保留，不能冒充新方向。若v2 ranking/Spearman/Near-ID前置条件不通过，不启动额外梯度分叉。\n\n'
    text+='## 20. 当前是否允许 QAM？\n\n'
    text+=('有critic通过真实梯度gate，已允许它执行原QAM smoke→50更新fixed-reference小pilot；不允许未通过方法或dynamic beta。' if actorallowed else '否。STOP_ACTOR_QAM；只有固定held-out排序及真实广泛gradient门槛通过才允许actor更新。')+'\n\n'
    for m in ['scalar','divl']:
        act=load(O/f'fixed_qam_{m}_complete.json');pilot=load(O/f'fixed_qam_{m}_dev_pilot.json')
        if act:text+=f'{m} fixed actor实际完成{act["steps"]}updates；reference/critic不变；action drift={act["action_drift"]}。\n\n'
        if pilot:text+=f'{m} 20-case paired dev pilot：baseline{pilot["baseline_success"]:.1%}→updated{pilot["updated_success"]:.1%}；Δ={pilot["success_delta"]:.1%}，CI={pilot["paired_success_delta_ci95"]}。小工程pilot，非最终论文成绩；final split未使用。\n\n'
    allocation={m:load(O/f'critic_{m}_v2_sharding.json') for m in ['scalar','divl']}
    text+='## 执行及解释边界\n\nScalar/DIVL只共享修复 replay分布及LIBERO原默认td10，原loss/QAM/encoder/Q结构不变；同步变更两项不能用于单变量因果归因。新采集无结果筛选，使用9个独立sim、3个冻结推理服务；多卡按实际 batch shard 与参数设备记录确认，保持全局batch256及同样2000预算。其他用户进程未停止。正式最终10任务success比较未完成，不能宣称DIVL有效。\n\n'
    if any(allocation.values()):text+='实际多卡证据：\n\n'+json.dumps(allocation,ensure_ascii=False,indent=2)+'\n\n'
    if evaluation and not actorallowed:
        text+='## V2结果对应的下一步\n\n'
        strong_control=bool(control and control['metrics']['rank']['mean']>=.6 and control['metrics']['rank']['ci95'][0]>.5)
        any_rank=any(v['conditions']['ranking'] for v in evaluation['prerequisites'].values())
        if not any_rank and strong_control:text+='简单正对照已有明确可学习性，正式critic动作排序未通过：停止扩大数据，优先检查共享critic架构/训练实现；不能继续增加DIVL预算或只调DIVL。\n\n'
        elif not any_rank:text+='本轮尚未建立可靠的held-out动作可学习性：停止扩大数据和盲目重训；需要重新设计训练初态覆盖/局部动作干预或state representation，不能把结果解释成DIVL失效。\n\n'
        elif not gradient:text+='部分整体动作排序已改善，但其Spearman或Near-ID证据未同时达到原门槛。保留所有结果，明确具体缺项；不能降低标准直接进行actor。\n\n'
        else:text+='动作排序的前置门槛通过，但新的真实梯度收益仍未广泛成立。停止actor，不把提高预测Q当作提高真实return。\n\n'
    if pipeline.get('error'):text+='本轮执行失败详情：'+pipeline['error']+'。日志与已有结果全部保留，禁止把中断视为完成。\n'
    tmp=P/'current_stage_summary.partial';tmp.write_text(text,encoding='utf-8');tmp.replace(P/'current_stage_summary.md')
    print('STAGE_REPORT_UPDATED',state,collection,flush=True)

if __name__=='__main__':main()
