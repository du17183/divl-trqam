"""Engineering revision021: report only admitted rates; retain frozen v1 science."""
from mixed_actor_pipeline_v1 import *
import math

REVISION = EXP / 'provenance/queue_summary_repair021'

def admitted_rates(evaluations):
 rates = {}
 for key, value in evaluations.items():
  if value.get('status') != 'AUDITED_COMPLETE500':
   continue
  assert value['episodes'] == 500 and value.get('audit_sha256'), (key, value)
  rate = value['success_rate']
  assert math.isfinite(rate) and 0 <= rate <= 1, (key, value)
  rates[key] = round(rate * 100, 3)
 return rates

class ResumeQueue(Queue):
 def refresh(self):
  for path in (EXP/'metrics/actor').glob('*/lifecycle.json'):
   self.state['actor_updates'][path.parent.name] = read(path).get('scientific_actor_updates', 0)
  self.state['completed_evaluation_episodes'] = sum(v.get('episodes',0) for v in self.state['evaluations'].values())
  rates = admitted_rates(self.state['evaluations'])
  progress(self.state)
  path = EXP/'reports/current_progress.md'
  text = path.read_text().replace('专家审计任务完成：0/10','专家审计任务完成：10/10')
  text = text.replace('最新真实成功率：新实验尚无完成500回合结果。历史九组保持原值。', '最新完整500成功率：'+json.dumps(rates,ensure_ascii=False)+'。历史九组保持原值。')
  path.write_text(text)
  (ROOT/'reports/current_stage_summary.md').write_bytes(path.read_bytes())

def priority_report_addendum():
 """Save required interpretation before any Cosine500 job can start."""
 from mixed_results_report_v1 import table, actual_evaluation
 data=read(EXP/'metrics/all_metrics.json')
 first=data['mixed100']
 assert len(first)==6 and all(row['episodes']==500 for row in first)
 tasks=[row for row in data['per_task'] if row['scheduler']=='cosine100']
 assert len(tasks)==60
 task3=[]
 for method in 'CDEFGH':
  row=next(x for x in tasks if x['method']==method and x['task_id']==3)
  old=actual_evaluation(method)[3]
  assert len(old)==50
  old_rate=100*sum(x['success'] for x in old)/50
  task3.append({'method':method,'old_task3_percent':old_rate,
                'mixed_task3_percent':row['success_rate_percent'],
                'mixed_minus_old_pp':row['success_rate_percent']-old_rate,
                'mixed_minus_SFT80_pp':row['success_rate_percent']-80,
                'SFT_failure_to_success':row['frozen_failure_to_success'],
                'SFT_success_to_failure':row['frozen_success_to_failure']})
 path=EXP/'reports/05_mixed_cosine100_results.md'
 original=path.read_text()
 backup=REVISION/'priority_report_before_addendum021.md'
 with backup.open('x') as file:
  file.write(original)
 addition='\n\n## Task3 能力退化核查\n\n冻结SFT Task3为80%；以下完整列出六种Mixed方法与其历史同方法对照，改善和退化均保留。\n\n'
 addition+=table(task3,['method','old_task3_percent','mixed_task3_percent','mixed_minus_old_pp','mixed_minus_SFT80_pp','SFT_failure_to_success','SFT_success_to_failure'])
 addition+='\n\n## 冻结Critic排序与真实梯度诊断\n\n以下复用已完成且冻结的04报告，属于Critic诊断，不是Actor闭环成功率；诊断未决定采样比例、方法去留或训练长度。Mixed同时改变Critic和主Actor数据，不能将闭环变化单独归因于其中一项。\n\n'
 addition+=(EXP/'reports/04_mixed_critic_comparison.md').read_text()
 path.write_text(original+addition)
 barrier=EXP/'metrics/mixed100_report_barrier015.json'
 before=read(barrier);assert before['complete'] and before['episodes']==3000 and before['per_task_rows']==60
 write(REVISION/'report_barrier_before_addendum021.json',before,True)
 before['files']['reports/05_mixed_cosine100_results.md']=sha(path)
 before.update(priority_interpretation_saved_before_cosine500=True,task3_rows=6,
               frozen_critic_comparison_sha256=sha(EXP/'reports/04_mixed_critic_comparison.md'))
 write(barrier,before)
 write(REVISION/'priority_report_addendum021.json',{'utc':stamp(),'passed':True,
       'report05_sha256':sha(path),'barrier_sha256':sha(barrier),
       'task3_rows':task3,'closed_loop_episodes':3000,
       'budget_or_hyperparameter_changes':False},True)

def main():
 deadline = time.monotonic()+30
 while os.getppid()!=1 and time.monotonic()<deadline:
  time.sleep(.1)
 assert os.getppid()==1 and not Path('/proc/3833199').exists()
 source = read(REVISION/'resume_sources021b.json')
 for relative, digest in source['files'].items():
  assert sha(EXP/relative)==digest,relative
 verify_sources(); protected_audit()
 for relative,digest in read(EXP/'provenance/mixed_actor_queue_ready010.json')['files'].items():
  assert sha(EXP/relative)==digest,relative
 assert read(REVISION/'cpu_summary_tests021b.json')['passed']
 assert read(REVISION/'original_1000_and_orphans_verified.json')['old_coordinator_absent']
 assert read(REVISION/'orphan_shutdown_complete021.json')['all_eight_processes_absent']
 assert read(EXP/'metrics/mixed_critic_comparison_complete015.json')['passed']
 for name in ['C-Mix100','D-Mix100']:
  result=EXP/'metrics/actor'/name/'result.json'
  assert sha(result)==source['reused_actor_result_sha256'][name]
  record=read(result)
  assert record['passed'] and record['scientific_actor_updates']==100 and record['all_saved_checkpoints_numerically_admitted']
 with (REVISION/'mixed_actor_pipeline021.lock').open('x') as lock:
  lock.write(str(os.getpid()))
 q=ResumeQueue()
 original=read(REVISION/'before/metrics/pipeline_state001.json')
 q.state.update(finished_jobs=original['finished_jobs'], resumed_from_dead_coordinator=3833199,
                original_started_utc=original['started_utc'], engineering_revision=21,
                orphan_shutdown_observation=read(REVISION/'orphan_shutdown_complete021.json'))
 q.state['evaluation_resource_intervals'] = original.get('evaluation_resource_intervals',[]) + [read(REVISION/'orphan_shutdown_complete021.json')['resource_interval']]
 write(REVISION/'coordinator_active021.json',q.state['coordinator'],True)
 try:
  # Frozen diagnostics and all complete Actor results are reused without updates.
  for methods in [('C','D'),('E','F'),('G','H')]:
   ids=[m+'-Mix100' for m in methods]
   q.train_pair(ids);q.evaluate([(x,100,'params') for x in ids]);q.report()
  q.report(True)
  assert read(EXP/'metrics/mixed100_report_barrier015.json')['complete']
  priority_report_addendum()
  for data in ['R0','R3']:
   for methods in [('C','D'),('E','F'),('G','H')]:
    ids=[m+'-'+data+'-Cosine500' for m in methods];q.train_pair(ids)
    for step in [100,250,500]:
     q.evaluate([(x,step,'params') for x in ids]);q.report()
  q.train_pair(['BC-Cosine500'])
  for step in [100,250,500]:
   q.evaluate([('BC-Cosine500',step,'params'),('BC-Cosine500',step,'ema_params')]);q.report()
  q.state.update(stage='ALL24000_EPISODES_COMPLETE_AWAITING_ARTIFACT_ACCEPTANCE',all_scientific_jobs_complete=True)
  q.report();q.refresh()
 except Exception:
  q.state.update(stage='ENGINEERING_STOP_ACTOR_QUEUE',error=traceback.format_exc())
  print(q.state['error'],flush=True);q.refresh()
 finally:
  write(REVISION/'mixed_actor_pipeline_result021.json',{**q.state,'ended_utc':stamp(),'wall_s':time.monotonic()-q.started},True)

if __name__=='__main__':
 main()
