"""CPU paired heldout/physical diagnostics; missing full MC never imputed."""
from mixed_common_v2 import *
import csv,numpy as np
from physical_report_supplement_v1 import aggregate

def csvfile(name,rows):
 p=EXP/'metrics'/name
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,sorted({k for r in rows for k in r}),extrasaction='ignore');w.writeheader();w.writerows({k:('N/A' if v is None else v) for k,v in r.items()} for r in rows)

def main():
 protected_audit();sources={};results={m:read(EXP/'metrics/diagnostics'/f'{m}_comparison015.json') for m in ['scalar','divl']};assert all(v['passed'] for v in results.values());held=[];summary=[];grad=[];cases=read(EXP/'configs/mixed_physical_validation015.json')['cases']
 for m,j in results.items():
  for data,d in j['datasets'].items():
   held.extend(d['heldout'])
   summary.append({'method':m,'data':data,'heldout_windows':d['heldout_count'],'complete_MC_windows':d['complete_MC_windows'],'censored_windows':d['censored_windows'],'physical_states':len(d['per_state_ranking']),'ranking_action_pairs':sum(x['action_pairs'] for x in d['per_state_ranking']),'physical_sources':len({x['state_source_id'] for x in d['per_state_ranking']}),**{k:d['ranking'][k]['all10_macro'] for k in ['ranking','near_id_ranking','spearman']}})
 for case in cases:
  if not case['gradient_selected']:continue
  oldp=OLD/f"metrics/physical002/case_{case['case_id']:03d}.json";newp=EXP/f"metrics/mixed_physical015/case_{case['case_id']:03d}.json";a,b=read(oldp),read(newp);assert a['passed'] and b['passed'] and b['complete'] and a['case']==b['case'];sources[str(oldp)]=sha(oldp);sources[str(newp)]=sha(newp)
  random=next(x for x in a['rows'] if x['candidate']=='matched_random' and x['valid']);assert sha(random['trace_path'])==random['trace_sha256'];sources[random['trace_path']]=random['trace_sha256']
  for m in ['scalar','divl']:
   for data,records in [('R0',a['rows']),('R3',b['rows'])]:
    positive=next(x for x in records if x['candidate']==m+'_positive' and x['valid']);assert sha(positive['trace_path'])==positive['trace_sha256'];sources[positive['trace_path']]=positive['trace_sha256'];rms1=positive['effective_controller7d_rms'];rms2=random['effective_controller7d_rms'];ratio=max(rms1,rms2)/max(min(rms1,rms2),1e-30);valid=ratio<=1.1
    grad.append({'task_id':case['task_id'],'case_id':case['case_id'],'state_source_id':case['state_source_id'],'method':m,'data':data,'amplitude_matched':valid,'effective_amplitude_ratio':ratio,'positive_observed_discounted_return':positive['return'],'random_observed_discounted_return':random['return'],'positive_minus_random':positive['return']-random['return'] if valid else None,'positive_complete_MC':not positive['censored'],'random_complete_MC':not random['censored'],'same_frozen_continuation':True})
 gs={m:{d:aggregate([x for x in grad if x['method']==m and x['data']==d],'positive_minus_random') for d in ['R0','R3']} for m in ['scalar','divl']}
 paired=[]
 for m,j in results.items():
  a,b=j['datasets']['R3']['per_state_ranking'],j['datasets']['R0']['per_state_ranking'];old={x['case_id']:x for x in b}
  for key in ['ranking','near_id_ranking','spearman']:
   rr=[{**x,'delta':x[key]-old[x['case_id']][key]} for x in a if x[key] is not None and old[x['case_id']][key] is not None];v=aggregate(rr,'delta');paired.append({'method':m,'metric':key,'mixed_minus_rollout':v['all10_macro'],'ci_low':v['ci95'][0] if v['ci95'] else None,'ci_high':v['ci95'][1] if v['ci95'] else None,'task_coverage':v['task_coverage']})
  old={x['case_id']:x for x in grad if x['method']==m and x['data']=='R0'};rr=[{**x,'delta':x['positive_minus_random']-old[x['case_id']]['positive_minus_random']} for x in grad if x['method']==m and x['data']=='R3' and x['positive_minus_random'] is not None and old[x['case_id']]['positive_minus_random'] is not None];v=aggregate(rr,'delta');paired.append({'method':m,'metric':'positive_gradient_minus_random_observed_Return','mixed_minus_rollout':v['all10_macro'],'ci_low':v['ci95'][0] if v['ci95'] else None,'ci_high':v['ci95'][1] if v['ci95'] else None,'task_coverage':v['task_coverage']})
 csvfile('mixed_critic_per_task.csv',held);csvfile('old_vs_mixed_critic.csv',summary);csvfile('mixed_real_gradient_return.csv',grad);csvfile('mixed_critic_paired_comparisons.csv',paired)
 report=['# Mixed Critic 与旧 Critic 同来源验证','',f'UTC：{stamp()}','', '固定heldout1000窗口，两种数据配置逐样本对齐；完整MC与截断Observed Return分别列示。固定100状态、60物理来源、400个原独立动作候选用于同状态排序。新Mixed梯度仅在原30固定梯度状态追加60正方向分支，随机方向与冻结SFT续行复用原记录。任何验证结果不回灌训练、不用于调参或缩减Actor预算。','', '|Critic|数据|窗口|完整MC|截断|动作排序|Near-ID|同状态Spearman|','|---|---|---:|---:|---:|---:|---:|---:|']
 for r in summary:report.append('|'+ '|'.join(str(r[k]) for k in ['method','data','heldout_windows','complete_MC_windows','censored_windows','ranking','near_id_ranking','spearman'])+'|')
 report+=['','## 真实正梯度相对随机方向','',json.dumps(gs,ensure_ascii=False,indent=2),'','## Mixed−旧数据配对差异','',json.dumps(paired,ensure_ascii=False,indent=2),'','排序CI沿用原任务分层、物理来源聚类2000次Bootstrap。闭环成功率另用10000次。Near-ID仍为原baseline与independent_small描述性子集。Return达到520上限而未成功的分支是Observed后缀，不能称为完整MC。单训练Seed探索性；本轮同时改变Critic与主Actor数据，不能拆分归因。']
 (EXP/'reports/04_mixed_critic_comparison.md').write_text('\n'.join(report)+'\n')
 receipt={'passed':True,'utc':stamp(),'diagnostics':{m:sha(EXP/'metrics/diagnostics'/f'{m}_comparison015.json') for m in results},'actual_new_gradient_branches':sum(len(read(EXP/f"metrics/mixed_physical015/case_{c['case_id']:03d}.json")['rows']) for c in cases if c['gradient_selected']),'ranking':summary,'gradient_return':gs,'paired':paired,'files':sources,'report_sha256':sha(EXP/'reports/04_mixed_critic_comparison.md'),'no_optimizer_updates':True,'no_feedback_or_tuning':True}
 assert receipt['actual_new_gradient_branches']==60;write(EXP/'metrics/mixed_critic_comparison_complete015.json',receipt,True)
if __name__=='__main__':main()
