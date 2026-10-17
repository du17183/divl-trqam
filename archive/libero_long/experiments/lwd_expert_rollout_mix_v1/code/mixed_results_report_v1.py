"""CPU-only reports from admitted500 records and actually applied updates."""
from mixed_common_v2 import *
import argparse,csv,collections,numpy as np
from pilot8_report_v1 import bootstrap,task_ci,actual_evaluation

METHODS={'C':'Scalar QAM','D':'DIVL QAM','E':'Scalar BC＋QAM','F':'DIVL BC＋QAM','G':'Scalar TRQAM','H':'DIVL TRQAM','B':'BC'}
LIMITS='单SFT Seed0、训练Seed20261010，探索性比较。官方50初态反复用于开发，是standard/development Benchmark，不是未见Final。成功率CI按10任务等权分层、物理来源聚类，10000次Bootstrap；多重比较的区间未校正。Mixed同时改变Critic和主Actor数据，不能单独归因。'
def csvfile(name,rows,fields=None):
 p=EXP/'metrics'/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fields or sorted({k for r in rows for k in r}) or ['status'],extrasaction='ignore');w.writeheader();w.writerows({k:('N/A' if v is None else json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v) for k,v in r.items()} for r in rows)
def table(rows,fields):
 def value(v):return 'N/A' if v is None else ('%.6g'%v if isinstance(v,float) else str(v).replace('|','/'))
 return '\n'.join(['|'+'|'.join(fields)+'|','|'+'|'.join(['---']*len(fields))+'|',*['|'+'|'.join(value(r.get(k)) for k in fields)+'|' for r in rows]])
def markdown(name,title,body):
 (EXP/'reports'/name).write_text('# '+title+'\n\nUTC：'+stamp()+'\n\n'+body+'\n\n'+LIMITS+'\n')
def evaluation(key):
 p=EXP/'metrics/eval_audit'/f'{key}.json'
 if not p.exists():return None
 j=read(p);assert j['passed'] and j['episodes']==500
 proto=EXP/'configs'/f'evaluation_{key}.json';assert sha(proto)==j['protocol_sha256']
 by={t:[] for t in range(10)}
 for shard in j['shards']:
  assert sha(shard['path'])==shard['sha256'] and sha(shard['trace_path'])==shard['trace_sha256']
  d=read(shard['path']);assert d['complete'];by[d['task_id']].extend(d['records'])
 assert all(len(v)==50 and len({x['state_source_id'] for x in v})==50 for v in by.values())
 return by
def main():
 a=argparse.ArgumentParser();a.add_argument('--barrier',action='store_true');args=a.parse_args();protected_audit();verify_sources();protocol=read(EXP/'configs/protocol001.json');base=actual_evaluation('A');assert base is not None
 old={m:actual_evaluation(m) for m in ['A','B','B2',*'CDEFGH']};assert all(v is not None for v in old.values());evals={'old_'+m:v for m,v in old.items()};summaries=[];taskrows=[];training=[];source_rows=[];kl=[];resource=[]
 for run in protocol['actor_runs']:
  resultpath=EXP/'metrics/actor'/run['id']/'result.json';r=read(resultpath) if resultpath.exists() else None
  lp=EXP/'metrics/actor'/run['id']/'train_steps.jsonl';lines=[]
  if lp.exists():
   raw=lp.read_bytes();assert raw.endswith(b'\n'), 'Reporter is run after the active worker transaction closes'
   lines=[json.loads(x) for x in raw.splitlines()];assert [x['update'] for x in lines]==list(range(1,len(lines)+1)) and all(x['update_applied']==1 for x in lines)
   for d in lines:
    training.append({'run_id':run['id'],**{k:v for k,v in d.items() if not isinstance(v,(dict,list))}})
    if 'path_kl' in d:kl.append({'run_id':run['id'],**d})
    for x in d.get('main_sample_draws',[]):source_rows.append({'kind':'actor_main','run_id':run['id'],'update':d['update'],**x})
    for x in d.get('bc_indices',[]) if d.get('bc_supervised_samples_used',0) else []:source_rows.append({'kind':'BC_supervision','run_id':run['id'],'update':d['update'],**x})
  if r:resource.append({'run_id':run['id'],'kind':'Actor','updates':r['scientific_actor_updates'],'beta_updates':r['beta_optimizer_updates'],'wall_s':r['wall_s'],'allocated_gpu_hours':r['allocated_gpu_hours'],'seconds_per_compute_update':r.get('steady_compute_s'),'read_s_per_update':r.get('steady_read_s'),'global_batch':32,'micro_batch_per_gpu':8,'gradient_accumulation':1,'gpu_binding':r.get('runtime_revision'),'peak_own_memory_by_gpu_mib':{g:max(x['memory_mib'] for x in r.get('telemetry',[]) if x['gpu']==g) for g in {x['gpu'] for x in r.get('telemetry',[])}},'gate_and_diagnostic_time_included':True,'passed':r['passed']})
  for n in run['checkpoints']:
   for variant in ['params','ema_params'] if run['method']=='B' else ['params']:
    key=run['id']+f'_step{n:06d}'+('_EMA' if variant=='ema_params' else '');by=evaluation(key);row={'evaluation_id':key,'run_id':run['id'],'method':run['method'],'data':run['data'],'scheduler':run['scheduler'],'checkpoint_update':n,'variant':variant,'status':'AUDITED_COMPLETE500' if by else 'PENDING','episodes':500 if by else 0,'successes':None,'success_rate_percent':None,'mean_return':None,'mean_episode_length':None,'ci_low_percent':None,'ci_high_percent':None}
    if by:
     evals[key]=by;ci=bootstrap(by);ss=[]
     for t,rr in sorted(by.items()):
      baseline={x['state_source_id']:x for x in base[t]};tr={'evaluation_id':key,'run_id':run['id'],'method':run['method'],'data':run['data'],'scheduler':run['scheduler'],'checkpoint_update':n,'variant':variant,'task_id':t,'task_name':protocol['inherited'].get('id') and read(ROOT/'configs/libero_10_split_manifest.json')['tasks'][t]['task_name'],'episodes':50,'successes':sum(x['success'] for x in rr),'success_rate_percent':100*np.mean([x['success'] for x in rr]),'mean_return':np.mean([x['return'] for x in rr]),'mean_episode_length':np.mean([x['episode_length'] for x in rr]),'frozen_failure_to_success':sum(not baseline[x['state_source_id']]['success'] and x['success'] for x in rr),'frozen_success_to_failure':sum(baseline[x['state_source_id']]['success'] and not x['success'] for x in rr)};taskrows.append(tr);ss.append(tr)
     row.update(successes=sum(x['successes'] for x in ss),success_rate_percent=np.mean([x['success_rate_percent'] for x in ss]),mean_return=np.mean([x['mean_return'] for x in ss]),mean_episode_length=np.mean([x['mean_episode_length'] for x in ss]),ci_low_percent=ci[0]*100,ci_high_percent=ci[1]*100,physical_sources=sum(len({x['state_source_id'] for x in rr}) for rr in by.values()))
    summaries.append(row)
 for method in ['scalar','divl']:
  p=EXP/'metrics/critic'/f'mixed_{method}_train001/result.json'
  if p.exists():
   r=read(p);resource.append({'run_id':'Mixed_'+method,'kind':'Critic','updates':r['scientific_critic_updates'],'wall_s':r['wall_s'],'allocated_gpu_hours':r['allocated_gpu_hours'],'seconds_per_compute_update':r.get('steady_compute_s'),'read_s_per_update':r.get('steady_read_s'),'global_batch':128,'micro_batch_per_gpu':32,'gradient_accumulation':1,'gpu_binding':r.get('gpu_receipt'),'passed':r['passed']})
 from mixed_sampling_report_v1 import summarize
 sampling=summarize();state=read(EXP/'metrics/pipeline_state001.json')
 for interval in state.get('evaluation_resource_intervals',[]):resource.append({'run_id':json.dumps(interval['requests']),'kind':'Evaluation colocated inference and SIM','wall_s':sum(x['wall_s'] for x in interval['server_intervals']),'allocated_gpu_hours':interval['reserved_union_gpu_hours'],'intervals':interval['server_intervals'],'gpu_binding':[x['gpu'] for x in interval['server_intervals']],'updates':0,'cost_scope':interval['scope']})
 for job in state.get('finished_jobs',[]):
  if job.get('physical_gpus') and 'diagnose' in job['name'] or job.get('physical_gpus') and 'physical_infer' in job['name']:resource.append({'run_id':job['name'],'kind':'Frozen Critic diagnostics','wall_s':job['wall_s'],'allocated_gpu_hours':len(job['physical_gpus'])*job['wall_s']/3600,'updates':0,'gpu_binding':job['physical_gpus'],'cost_scope':'registered child lifetime; no duplicate colocated SIM GPU charge'})
 paired=[];recovery=[]
 def compare(x,y,label):
  if x not in evals or y not in evals:return
  left,right=evals[x],evals[y];ci=bootstrap(left,right);rate=lambda by:np.mean([np.mean([r['success'] for r in by[t]]) for t in range(10)])
  row={'comparison':label,'left':x,'right':y,'difference_pp':100*(rate(left)-rate(right)),'ci_low_pp':ci[0]*100,'ci_high_pp':ci[1]*100,'tasks':10,'episodes_per_arm':500,'physical_source_clusters_per_arm':500,'replicates':10000,'nominal_unadjusted':True};paired.append(row)
  for t in range(10):
   other={r['state_source_id']:r for r in right[t]};assert set(other)=={r['state_source_id'] for r in left[t]}
   assert all(all(r[k]==other[r['state_source_id']][k] for k in ['environment_seed','policy_seed','initial_vector_sha256']) for r in left[t])
   recovery.append({'comparison':label,'task_id':t,'failure_to_success':sum(not other[r['state_source_id']]['success'] and r['success'] for r in left[t]),'success_to_failure':sum(other[r['state_source_id']]['success'] and not r['success'] for r in left[t]),'difference_pp':100*(np.mean([r['success'] for r in left[t]])-np.mean([r['success'] for r in right[t]]))})
 key=lambda m,n=100,d=None:m+'-Mix100_step000100' if d is None else m+'-'+d+f'-Cosine500_step{n:06d}'
 for m in 'CDEFGH':
  compare(key(m),'old_'+m,m+'-Mix100−old-'+m);compare(key(m),'old_A',m+'-Mix100−A');compare(key(m),'old_B',m+'-Mix100−BC100')
 for x,y in [('D','C'),('F','E'),('H','G'),('H','D'),('E','C'),('F','D')]:compare(key(x),key(y),x+'-Mix100−'+y+'-Mix100')
 for m in 'CDEFGH':
  for n in [100,250,500]:compare(key(m,n,'R3'),key(m,n,'R0'),f'{m} Mixed−Rollout Cosine500 step{n}')
  for d in ['R0','R3']:
   for hi,lo in [(250,100),(500,250),(500,100)]:compare(key(m,hi,d),key(m,lo,d),f'{m} {d} Cosine500 {hi}−{lo}')
   for n in [100,250,500]:compare(key(m,n,d),f'BC-Cosine500_step{n:06d}',f'{m} {d}−matched BC step{n}')
 first=[]
 for m in 'CDEFGH':
  row=next(x for x in summaries if x['run_id']==m+'-Mix100');oldrate=100*np.mean([np.mean([r['success'] for r in old[m][t]]) for t in range(10)]);p=next((x for x in paired if x['comparison']==m+'-Mix100−old-'+m),{})
  first.append({'method':m,'name':METHODS[m],'old_rollout_cosine100_percent':oldrate,'mixed_cosine100_percent':row['success_rate_percent'],'mixed_successes':row['successes'],'episodes':row['episodes'],'difference_pp':p.get('difference_pp'),'ci_low_pp':p.get('ci_low_pp'),'ci_high_pp':p.get('ci_high_pp'),'status':row['status']})
 round1tasks=[x for x in taskrows if x['scheduler']=='cosine100'];lengths=[x for x in summaries if x['scheduler']=='cosine500'];matched=[x for x in lengths if x['method']=='B']
 csvfile('mixed_cosine100_results.csv',first);csvfile('mixed_cosine100_per_task.csv',round1tasks);csvfile('cosine500_all_checkpoints.csv',lengths);csvfile('paired_confidence_intervals.csv',paired);csvfile('recovery_regression.csv',recovery);csvfile('trqam_kl_dual_beta.csv',kl);csvfile('bc_budget_matched.csv',matched);csvfile('resource_summary.csv',resource);csvfile('actor_training_updates.csv',training);csvfile('actor_actual_source_draws.csv',source_rows)
 for suffix in ['success','return','length']:
  csvfile('all_methods_per_task_'+suffix+'.csv',taskrows)
 manifest=read(EXP/'data/performance2/mixed_manifest001.json');data_counts=collections.Counter(x.get('source') for x in manifest['expert_records']+manifest['rollout_records']);data_counts['heldout_excluded']=60
 allout={'utc':stamp(),'authority_sha256':protocol['authority_sha256'],'protocol_sha256':sha(EXP/'configs/protocol001.json'),'complete_24000':sum(x['episodes'] for x in summaries)==24000,'actual_new_evaluation_episodes':sum(x['episodes'] for x in summaries),'results':summaries,'mixed100':first,'per_task':taskrows,'paired':paired,'recovery':recovery,'resources':resource,'actual_critic_sampling':sampling,'data':dict(data_counts),'limits':LIMITS,'scientific_source':'admitted full500 audits, original episode records and complete applied-update logs','artifact_acceptance_pending':True}
 write(EXP/'metrics/all_metrics.json',allout)
 markdown('05_mixed_cosine100_results.md','六组Mixed100与历史同预算对照',table(first,['method','name','old_rollout_cosine100_percent','mixed_cosine100_percent','mixed_successes','episodes','difference_pp','ci_low_pp','ci_high_pp','status'])+'\n\n主池50专家＋240 Train Rollout，无来源配额；BC额外监督池仍50专家＋109成功Rollout、来源概率50:50。只列完整500的成功率；PENDING不填0。\n\n'+table(resource,['run_id','updates','beta_updates','allocated_gpu_hours','wall_s']))
 markdown('06_mixed_cosine100_per_task.md','Mixed100完整十任务指标',table(round1tasks,['method','task_id','task_name','successes','episodes','success_rate_percent','mean_return','mean_episode_length','frozen_failure_to_success','frozen_success_to_failure']))
 markdown('07_cosine500_training.md','Cosine500真实训练记录',table(resource,['run_id','updates','beta_updates','wall_s','allocated_gpu_hours','seconds_per_compute_update'])+'\n\nCosine500均独立从冻结SFT1000初始化；100/250/500为同一次500步训练轨迹的Checkpoint。不得与第一轮Cosine100拼成同一曲线。')
 markdown('08_actor_100_250_500_ablation.md','Actor训练长度与调度对照',table(lengths,['run_id','checkpoint_update','variant','successes','episodes','success_rate_percent','mean_return','mean_episode_length','status']))
 markdown('09_rollout_vs_mixed_comparison.md','同预算Rollout-only与Mixed',table([x for x in paired if 'Mixed−Rollout' in x['comparison']],['comparison','difference_pp','ci_low_pp','ci_high_pp']))
 markdown('10_bc_budget_matched.md','BC与EMA匹配预算',table(matched,['checkpoint_update','variant','successes','episodes','success_rate_percent','mean_return','mean_episode_length','status'])+'\n\nEMA共享同一条BC500训练，不增加主Actor优化器更新。E/F/G/H的BC池不加入失败Rollout。')
 kls=[]
 for rid in sorted({x['run_id'] for x in kl}):
  rr=[x for x in kl if x['run_id']==rid];kls.append({'run_id':rid,'updates':len(rr),'max_raw_path_kl':max(x['path_kl'] for x in rr),'last_lambda':rr[-1].get('lambda_effective'),'budget':rr[-1].get('kl_budget'),'over_budget_updates':sum(x['path_kl']>x['kl_budget'] for x in rr),'beta_optimizer_updates':rr[-1]['beta_optimizer_updates'],'reference_ema_updates':rr[-1]['reference_ema_updates']})
 markdown('11_trqam_kl_dual_beta.md','原生TRQAM路径KL、Dual与慢BC',table(kls,['run_id','updates','max_raw_path_kl','last_lambda','budget','over_budget_updates','beta_optimizer_updates','reference_ema_updates'])+'\n\nβ独立BC优化并更新EMA reference；主Actor通过原生路径KL/动态Dual/统计EMA训练。所有raw KL保留。若持续低于预算，不能仅凭成功率认定动态约束起关键作用。固定heldout动作/参数漂移见各checkpoint_drift.json。')
 markdown('12_per_task_analysis.md','逐任务改善与退化',table(taskrows,['run_id','checkpoint_update','variant','task_id','successes','success_rate_percent','mean_return','mean_episode_length'])+'\n\n重点Task0/3/5/6/8/9均包含；Task3冻结SFT80%、旧H42%的回退必须保留。恢复/退化配对计数见recovery_regression.csv，不只报告改善任务。')
 markdown('13_statistics.md','任务等权物理来源聚类配对统计',table(paired,['comparison','difference_pp','ci_low_pp','ci_high_pp'])+'\n\n训练Seed固定，方法和预算均预注册；逐任务每50物理来源，全部源/初态Hash/环境和策略seed在比较前逐条一致检查。')
 markdown('14_gpu_throughput.md','实际GPU成本与吞吐',table(resource,['run_id','kind','updates','wall_s','allocated_gpu_hours','seconds_per_compute_update','read_s_per_update'])+'\n\n正式训练成本包含首编译、Gate、读数据、保存与SHA。推理与同卡SIM用reserved_union_gpu_hours计一次，不重复累计显存遥测。未单独测量的通信/能耗为N/A。')
 complete=[x for x in summaries if x['episodes']==500];winner=max(complete,key=lambda x:x['success_rate_percent']) if complete else None
 markdown('final_conclusion.md','科研结论与当前完成边界',f"已完成新评测{allout['actual_new_evaluation_episodes']}/24000；{'科学预算已完成，待Excel统计验收。' if allout['complete_24000'] else '整体实验仍在运行，不能给出最终结论。'}\n\n当前完成模型中宏平均最高：{winner['evaluation_id']+' '+str(winner['success_rate_percent'])+'%' if winner else '尚无新完整500结果'}。预注册全部Checkpoint须完整展示，不能择优省略。\n\n"+table(first,['method','mixed_cosine100_percent','difference_pp','ci_low_pp','ci_high_pp','status']))
 if args.barrier:
  assert all(x['episodes']==500 for x in first) and len(round1tasks)==60
  files={f:sha(EXP/f) for f in ['reports/05_mixed_cosine100_results.md','reports/06_mixed_cosine100_per_task.md','metrics/mixed_cosine100_results.csv','metrics/mixed_cosine100_per_task.csv','metrics/paired_confidence_intervals.csv']}
  write(EXP/'metrics/mixed100_report_barrier015.json',{'complete':True,'utc':stamp(),'episodes':3000,'per_task_rows':60,'files':files,'saved_before_cosine500':True},True)
 print(json.dumps({'actual_episodes':allout['actual_new_evaluation_episodes'],'mixed100_complete':sum(x['episodes']==500 for x in first),'complete24000':allout['complete_24000']}))
if __name__=='__main__':main()
