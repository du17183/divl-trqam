"""CPU summaries from complete applied-update transactions; no sampler or model calls."""
from mixed_common_v2 import *
import collections,csv

def closed_prefix(path):
 if not path.exists():return []
 raw=path.read_bytes();raw=raw[:raw.rfind(b'\n')+1]
 rows=[json.loads(x) for x in raw.splitlines()]
 assert [r['update'] for r in rows]==list(range(1,len(rows)+1))
 assert all(r['update_applied']==1 for r in rows)
 return rows

def summarize():
 manifest=read(EXP/'data/performance2/mixed_manifest001.json')
 records={r['episode_id']:r for r in manifest['expert_records']+manifest['rollout_records']}
 assert len(records)==290
 sources=['expert','successful_rollout','failed_rollout'];rows=[];totals=[];coverage=[]
 for method in ['scalar','divl']:
  logs=closed_prefix(EXP/'metrics/critic'/f'mixed_{method}_train001/train_steps.jsonl')
  cumulative={t:collections.Counter() for t in range(10)};seen={t:set() for t in range(10)};overall=collections.Counter()
  for r in logs:
   by={t:collections.Counter() for t in range(10)}
   assert len(r['samples'])==128
   for task,episode,index in r['samples']:
    record=records[episode];assert task==record['task_id'] and 0<=index<record['transitions']
    by[task][record['source']]+=1;seen[task].add(episode)
   for t in range(10):
    cumulative[t].update(by[t]);overall.update(by[t])
    rows.append({'phase':'MixedCritic','method':method,'update':r['update'],'task_id':t,**{s:by[t][s] for s in sources},**{'cumulative_'+s:cumulative[t][s] for s in sources},'actual_distinct_trajectories':len(seen[t]),'indexed_trajectories':29,'sample_index_sha256':r['sample_index_sha256']})
   assert dict(overall)==r['cumulative_source_counts']
  for t in range(10):
   count=sum(cumulative[t].values())
   totals.append({'phase':'MixedCritic','method':method,'last_update':len(logs),'task_id':t,'actual_samples':count,**{s:cumulative[t][s] for s in sources},**{s+'_fraction':cumulative[t][s]/count if count else None for s in sources},'actual_distinct_trajectories':len(seen[t]),'indexed_trajectories':29})
  for eid,r in records.items():coverage.append({'method':method,'episode_id':eid,'task_id':r['task_id'],'source':r['source'],'indexed':True,'actual_sampled':eid in seen[r['task_id']],'valid_main_H5_windows':r['transitions'],'last_update':len(logs)})
 for name,values in [('mixed_replay_actual_sampling.csv',rows),('critic_sampling_per_task.csv',totals),('critic_actual_trajectory_coverage.csv',coverage)]:
  p=EXP/'metrics'/name;t=p.with_suffix('.csv.tmp')
  with t.open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(values[0]) if values else ['status']);w.writeheader();w.writerows(values)
  t.replace(p)
 out={'utc':stamp(),'manifest_sha256':sha(EXP/'data/performance2/mixed_manifest001.json'),'indexed_trajectories':290,'main_pool_source_quota':None,'sampler':'task balanced, within task trajectory uniform, within trajectory chunk uniform, with replacement','per_task':totals,'files':{n:sha(EXP/'metrics'/n) for n in ['mixed_replay_actual_sampling.csv','critic_sampling_per_task.csv','critic_actual_trajectory_coverage.csv']},'scientific_updates_performed':0}
 write(EXP/'metrics/mixed_sampling_summary018.json',out);return out

if __name__=='__main__':
 protected_audit();verify_sources();print(json.dumps(summarize()))
