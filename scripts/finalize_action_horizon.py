"""Final bounded conclusion, all34answers, frozen target audit and recovery figures."""
import json,time,numpy as np,pandas as pd
from scipy.stats import pearsonr,spearmanr
from horizon_common import R,D,O,REP,atomic,sha,report,safe
from analyze_replay_action_v3 import metrics,clustered
def fmt(x,n=6):return 'N/A' if x is None else f'{x:.{n}f}'
def ci(m,scale=1):
 return 'N/A' if m['mean'] is None else f"{m['mean']*scale:.2f} [{m['ci95'][0]*scale:.2f}, {m['ci95'][1]*scale:.2f}]"
def main():
 p=json.loads((D/'protocol.json').read_text());sens=json.loads((R/'artifacts/action_horizon_metrics.json').read_text())
 recovery=json.loads((R/'artifacts/recovery_curve_metrics.json').read_text());probe=json.loads((R/'artifacts/horizon_positive_control.json').read_text())
 # H1/H3 are changed schedules: originalH5 source equivalence is not applicable.
 for h in ['1','3']:sens['horizons'][h]['source_H5_gate_max']=None
 atomic(R/'artifacts/action_horizon_metrics.json',sens)
 trace=json.loads((R/'artifacts/action_object_shape_trace.json').read_text());arith=json.loads((O/'target_arithmetic.json').read_text())
 selected=json.loads((D/'state_selection_manifest.json').read_text())['records']
 trace['replay_provenance']=dict(path=selected[0]['path'],source_index=selected[0]['source_index'],sha256=selected[0]['sha256'],batch_note='Two copies of the same real selected replay state for the B=2 dimension/mask audit; no synthetic observation.')
 atomic(R/'artifacts/action_object_shape_trace.json',trace);atomic(O/'action_object_shape_trace.json',trace)
 questions='''\n\n## 十二项逐项回答\n\n| 问题 | 代码事实 / 运行时确认 |\n|---|---|\n| 1. Q输入A的shape | [B,10,7]，真实batch B=2 |\n| 2. A包含多少时间位置 | 存储10；前5有效，后5时间padding |\n| 3. 单步7D吗 | 不是；每步7D、有效5步 |\n| 4. 5-step prefix吗 | 是，预测10、执行5、重观察 |\n| 5. 完整10-step prediction吗 | Actor预测10×32，Q只看首5×7 |\n| 6. 32维padding如何进入Q | 后25通道在qam.py:59切掉，不进入Q head |\n| 7. QAM对哪个tensor求梯度 | 对flow终点[B,10,32]的−sum(minQ)/temperature |\n| 8. gradient对应整个chunk还是首动作 | gradient shape整个10×32，但直接终端非零仅首5×7，不只是首动作 |\n| 9. terminal adjoint作用时间步 | prefix位置0..4、真实通道0..6 |\n| 10. backward传播多少action steps | 10个flow积分interval，每次VJP覆盖10×32；不是10envsteps |\n| 11. velocity matching哪些位置 | qam.py:364对全部10×32求平方和，网络Jacobian允许间接跨位置耦合 |\n| 12. target k与action horizon关系 | 首chunk k<=5；训练TD10chunks的累计K通常50；从s_t起算到s_{t+K}，乘gamma^K或terminal0 |\n\n实际TwinQHead(7,10)为libero_lwd_training.py:56；Scalar TD为:107；奖励/discount累计为divl_libero_adapter.py:75-77。\n'''
 for ap in [REP/'action_object_code_audit.md',R/'reports/action_object_code_audit.md']:
  with ap.open('a') as f:f.write(questions)
 manifest=json.loads((D/'manifest.json').read_text());taskmap={r['state_id']:r['task_id'] for r in manifest['records']}
 values=[];targetsummary={}
 for method in ['scalar','divl']:
  frame=pd.read_parquet(O/f'target_{method}.parquet');values.append(frame);targetsummary[method]={}
  assert len(frame)==1500 and frame.target_formula_error.max()<1e-6
  for h in [1,3,5]:
   part=frame[frame.horizon==h];state=[]
   for sid,rows in part.groupby('state_id'):
    rows=rows.sort_values('candidate_index');m=metrics(rows.Target_C.to_numpy(),rows.Target_A.to_numpy());m.update(state_id=sid,task_id=int(taskmap[sid]));state.append(m)
   targetsummary[method][h]=dict(samples=len(part),Target_A_mean=float(part.Target_A.mean()),Target_B_mean=float(part.Target_B.mean()),Target_C_mean=float(part.Target_C.mean()),
    td_environment_steps_min=int(part.td_environment_steps.min()),td_environment_steps_max=int(part.td_environment_steps.max()),td_environment_steps_mean=float(part.td_environment_steps.mean()),
    nonterminal_bootstrap_fraction=float((~part.terminal_window).mean()),
    Pearson_A_C=float(pearsonr(part.Target_A,part.Target_C).statistic) if part.Target_A.std()>0 and part.Target_C.std()>0 else None,
    Spearman_A_C=float(spearmanr(part.Target_A,part.Target_C).statistic) if part.Target_A.std()>0 and part.Target_C.std()>0 else None,
    within_state_C_vs_A_ranking=clustered(state,'rank'),within_state_C_vs_A_spearman=clustered(state,'spearman'))
 pd.concat(values).to_parquet(O/'target_alignment_values.parquet',index=False)
 targetobj={**arith,'bootstrap_values_pending':False,'frozen_v2_target_evaluations':targetsummary,
  'estimand_boundary':'MC frozen-continuation return versus scalar expectile/adaptive-quantile bootstrapped value are different estimands. Numeric disagreement alone is NOT an implementation bug. Same origin/native reward/gamma/terminal definitions verified.',
  'checkpoints':{m:json.loads((O/f'target_{m}_complete.json').read_text()) for m in ['scalar','divl']}}
 atomic(R/'artifacts/target_alignment.json',safe(targetobj))
 # No best target/model/checkpoint selection. Fixed gate from the pre-outcome protocol.
 diff=sens['paired']['H5_minus_H1'];rank=probe['horizons']['5']['metrics']['all']['rank']
 conditions=dict(mean_sensitivity=diff['mean_abs_delta']['ci95'][0]>0,
  large_return_contrast=diff['probability_delta_gt_01']['ci95'][0]>0,
  H5_positive_control=rank['mean'] is not None and rank['mean']>=.60 and rank['ci95'][0]>.50)
 decision='TARGET_ALIGNMENT_BUG' if targetobj['implementation_bug'] else 'ACTION_OBJECT_MISMATCH' if p['action_object_mismatch'] else 'ACTION_HORIZON_CONFIRMED' if all(conditions.values()) else 'ACTION_HORIZON_REJECTED'
 lines=['| Horizon | Branches | Mean / median abs(ΔG) | P>1e−4 | P>1e−3 | P>1e−2 | Success flip | OOF ranking /95%CI | Spearman /95%CI |',
        '|---|---:|---:|---:|---:|---:|---:|---|---|']
 for h in ['1','3','5']:
  s=sens['horizons'][h];pr=probe['horizons'][h]['metrics']['all'];pro=s['probabilities']
  lines.append(f"| H{h} | {s['branches']} | {s['mean_absolute_delta_G']:.6f} / {s['median_absolute_delta_G']:.6f} | {100*pro['0.0001']:.2f}% | {100*pro['0.001']:.2f}% | {100*pro['0.01']:.2f}% | {100*s['success_flip_rate']:.2f}% | {ci(pr['rank'],100)}% | {ci(pr['spearman'])} |")
 lines.append('| H-QAM | =H5，复用500条 | 同H5 | 同H5 | 同H5 | 同H5 | 同H5 | 同H5 | 同H5 |')
 table='\n'.join(lines)
 scores={h:probe['horizons'][h]['metrics']['all']['rank']['mean'] for h in ['1','3','5']}
 strongest=max(scores,key=lambda k:-1 if scores[k] is None else scores[k]);largest=max(['1','3','5'],key=lambda h:sens['horizons'][h]['mean_absolute_delta_G'])
 recovered=recovery['recovery_times']
 recovery_support={}
 for h in ['1','3','5']:
  curve=recovery['curves'][h];end=curve['intervention_end']['EEF_position_L2']['mean'];after=curve['replan1']['EEF_position_L2']['mean']
  recovery_support[h]=dict(EEF_after_first_to_end_ratio=after/end if end is not None and end>1e-9 and after is not None else None,
   recovery_by3=recovered[h]['recovered_by_replan3_fraction'],caveat='Available same-time samples; ratio can be affected by terminal censoring. Convergence does not alone establish policy-correction mechanism without open-loop control.')
 report('return_target_alignment.md','Return target alignment',
  '## 代码与数值检查\n\n'+json.dumps(safe(targetobj),indent=2)+
  '\n\nTarget A为原snapshot起算的完整原生discounted MC return；B为实际执行首chunk奖励；C为真实TD10chunks reward＋discount×冻结EMA V。通常H1/H3/H5的TD窗口分别46/48/50env steps，遇terminal截断。C不是B+gamma^H V，因为当前正式代码明确td_steps=10chunks。terminal成功奖励计入reward，bootstrap归零；time limit在现有协议中为terminal failure。上述约定保持不变。\n\nMC与bootstrap估计不同是有意的n-step/value备份，不能把C与A偏差自动判为bug，更不能按哪个target更好来改reward。')
 # Standalone scientific recovery figure, all physical metrics/intervals retained.
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
 labels=['intervention_end','replan1','replan2','replan3','terminal']
 for ax,key,title in zip(axes,['proprio_L2','EEF_position_L2','mean_object_position_L2'],['Policy proprio deviation','EEF position deviation (m)','Object position deviation (m)']):
  for h,color in zip(['1','3','5'],['#2563eb','#d97706','#059669']):
   vals=[recovery['curves'][h][lab][key]['mean'] for lab in labels];cis=[recovery['curves'][h][lab][key]['ci95'] for lab in labels]
   y=np.array([np.nan if v is None else v for v in vals]);lo=np.array([np.nan if v is None else v[0] for v in cis]);hi=np.array([np.nan if v is None else v[1] for v in cis])
   ax.plot(range(5),y,'o-',label=f'H{h}',color=color);ax.fill_between(range(5),lo,hi,color=color,alpha=.12)
  ax.set_xticks(range(5),['End','Replan1','Replan2','Replan3','Terminal'],rotation=25);ax.set_title(title);ax.grid(alpha=.2);ax.legend()
 fig.suptitle('Paired behavior-relative recovery; bands=state-bootstrap95%CI; terminal comparison may be censored')
 fig.savefig(REP/'policy_recovery_curve.png',dpi=160);fig.savefig(REP/'policy_recovery_curve.pdf');plt.close(fig)
 with (REP/'policy_recovery_curve.md').open('a') as f:f.write('\n\n![Recovery curve](policy_recovery_curve.png)\n\n'+json.dumps(safe(recovery_support),indent=2)+'\n')
 with (R/'reports/policy_recovery_curve.md').open('a') as f:f.write('\n\n![Recovery curve](action_horizon_alignment_20261008/policy_recovery_curve.png)\n')
 completed=json.loads((O/'collection_status.json').read_text());manifest=json.loads((D/'manifest.json').read_text())
 totalsteps=sum(r['steps'] for r in manifest['records']);totalbytes=sum(__import__('pathlib').Path(r['path']).stat().st_size for r in manifest['records']);elapsed=manifest['elapsed_seconds']
 runtime=dict(inference_gpus=8,sim_workers=16,recorded_primary_branches=1500,behavior_restore_repeats=300,
  full_collection_seconds=elapsed,preflight_cached_states=1,new_primary_branches=1485,
  primary_branches_per_minute=1485/elapsed*60,
  total_recorded_primary_env_steps=totalsteps,recorded_primary_bytes=totalbytes,
  rate_boundary='Collection time excludes cached firststate preflight; each freshstate additionally has3repeat restore rollouts. No inference load time in denominator. GPU-wide utilization includes other jobs; owned frozen servers stopped.',
  branch_steps_per_second=sum(r['steps'] for r in manifest['records'] if r['state_id']!=json.loads((D/'state_selection_manifest.json').read_text())['records'][0]['state_id'])/elapsed)
 atomic(O/'runtime_summary.json',runtime)
 answers=[]
 def add(n,q,a):answers.append(f'## {n}. {q}\n\n{a}\n')
 add(1,'Critic真实action object','[B,10,7]时间容量，mask仅前5真实env动作。对象为5-step prefix。')
 add(2,'QAM求梯度tensor shape',str(trace['QAM_terminal_gradient_shape'])+'，真实batch B=2；terminal仅前5×7坐标非零。')
 add(3,'优化单步、5-step还是10-step','直接终端Q目标为5-step；Actor预测10×32，flow VJP和velocity matching涉及整张量。10flow积分间隔不是10env步。')
 add(4,'旧诊断是否一致','一致。旧50-state及v4均实际执行首k<=5。没有action_object_mismatch。')
 for n,h in [(5,'1'),(6,'3'),(7,'5')]:add(n,f'H{h}分支数','500（100states×5candidates）；另100条独立behavior重复恢复审计。')
 add(8,'H-QAM分支数','H-QAM=H5，共用500条，无额外重复采集。')
 add(9,'restore最大误差',json.dumps({h:dict(repeat=sens['horizons'][h]['repeat_restore_max'],recorded_H5=sens['horizons'][h]['source_H5_gate_max'],initial=sens['horizons'][h]['initial_state_error_max']) for h in ['1','3','5']})+'；H-QAM同H5。')
 for n,h in [(10,'1'),(11,'3'),(12,'5'),(13,'5')]:
  s=sens['horizons'][h];add(n,'H-QAM mean/median |ΔG|' if n==13 else f'H{h} mean/median |ΔG|',f"{s['mean_absolute_delta_G']:.8f} / {s['median_absolute_delta_G']:.8f}，400个扰动branch，不包含behavior。")
 for n,h in [(14,'1'),(15,'3'),(16,'5'),(17,'5')]:add(n,'H-QAM P(|ΔG|>.01)' if n==17 else f'H{h} P(|ΔG|>.01)',f"{100*sens['horizons'][h]['probabilities']['0.01']:.2f}%")
 add(18,'success flip rate',json.dumps({h:sens['horizons'][h]['success_flip_rate'] for h in ['1','3','5']})+'；H-QAM=H5。')
 for n,h in [(19,'1'),(20,'3'),(21,'5'),(22,'5')]:add(n,'H-QAM recovery' if n==22 else f'H{h} recovery',json.dumps(recovered[h])+'；不能把未恢复/terminal-censored样本从分母删除。')
 add(23,'是否支持短扰动迅速纠正',json.dumps(recovery_support)+'；按观察趋势判断，不能仅凭曲线作完整因果机制证明。')
 add(24,'future return与训练target对齐','起点、native reward、gamma、terminal一致；A为MC，C为10chunk bootstrap，估计量不同。没有实现级错位。')
 add(25,'k/gamma/terminal是否不一致','未发现实现错误。首chunk k与TD窗口累计K分别记录，gamma^K取storeddiscount乘积；成功terminal奖励包含且bootstrap0。')
 for n,h in [(26,'1'),(27,'3'),(28,'5'),(29,'5')]:add(n,'H-QAM positivecontrol ranking' if n==29 else f'H{h} positivecontrol ranking',ci(probe['horizons'][h]['metrics']['all']['rank'],100)+'%，whole-state OOF/95%CI。')
 add(30,'state-wise Spearman',json.dumps({h:probe['horizons'][h]['metrics']['all']['spearman'] for h in ['1','3','5']}))
 add(31,'哪个horizon监督最强',f'最大mean|ΔG|为H{largest}；最高OOF ranking为H{strongest}。二者不能混称同一指标。')
 add(32,'是否支持horizon mismatch',f'{decision}；原旧H5已经一致；预注册条件：'+json.dumps(conditions)+'。REJECTED仅指本轮未支持“更长完整对象足以提高可学习性”的假设，不证明无动作因果效应。')
 add(33,'是否构造chunk-sensitive replay','先输出设计，不自动采v5。' + ('可考虑按实际5-step对象，行为prefix＋固定Near/Mid candidate，重复续跑控制noise，所有结果保留。' if decision=='ACTION_HORIZON_CONFIRMED' else '本轮不扩大同类扰动；先审计behavior quality、mixed-quality replay/较长连续行为偏移，不能自动启动online或改reward。'))
 add(34,'是否仍STOP_ACTOR_QAM','是。没有真实可靠新critic广泛gradient门槛，不启动Actor、QAM、dynamic beta或正式critic retrain。')
 body='**'+decision+'**；同时保持STOP_ACTOR_QAM。\n\n'+table+'\n\n'+('\n'.join(answers))+'\n## Actual runtime\n\n'+json.dumps(runtime,indent=2)+'\n\n所有原始数据、旧50-state诊断、v2/v3/v4报告及src源码保持原哈希。首轮新增collector字段重复TypeError保留日志/invalid文件，第二次预检修复；无样本排除，无参数调优。'
 if not (O/'previous_current_stage_summary.md').exists():
  (O/'previous_current_stage_summary.md').write_text((R/'reports/current_stage_summary.md').read_text())
 report('current_stage_summary.md','Action horizon alignment —34answers',body)
 frozenok=all(sha(path)==h for path,h in p['frozen_input_hashes'].items());assert frozenok
 report('action_horizon_conclusion.md','Action horizon conclusion',f'**{decision}**\n\n'+table+'\n\n## Fixed conditions\n\n'+json.dumps(conditions,indent=2)+'\n\nOld50/v4 already executed full5-step QAM object; no action_object mismatch. A/C different estimands but numeric target arithmetic aligned. All100train initialIDs and1500branches retained. No formalcritic/Actor/QAM/dynamic beta.\n\n## Paired sensitivity\n\n'+json.dumps(sens['paired'],indent=2)+'\n\n## Recovery summary\n\n'+json.dumps(safe(recovery_support),indent=2))
 atomic(O/'round_complete.json',dict(complete=True,decision=decision,conditions=conditions,states=100,branches=1500,H_QAM_alias=5,restore_repeat_rollouts=300,
  actor_updated=False,QAM_updated=False,formal_critics_trained=False,original_inputs_unchanged=frozenok,
  protocol_sha256=sha(D/'protocol.json'),manifest_sha256=sha(D/'manifest.json'),finished_at_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()))
 print('ACTION_HORIZON_ROUND_COMPLETE',decision,table,flush=True)
if __name__=='__main__':main()
