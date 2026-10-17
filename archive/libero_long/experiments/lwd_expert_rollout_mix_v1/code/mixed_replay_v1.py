"""All 290 audited trajectories; unchanged task/trajectory/chunk draw rule."""
from mixed_common_v2 import *
import collections

class MixedReplay:
 def __init__(self,verify_files=True):
  m=read(EXP/'data/performance2/mixed_manifest001.json')
  assert m['complete'] and m['audit_passed'] and m['episodes']==290 and m['source_quota'] is None
  self.records=m['expert_records']+m['rollout_records'];self.by_task={t:[] for t in range(10)};self.by_id={};self.handles={}
  tasks=read(ROOT/'configs/libero_10_split_manifest.json')['tasks'];self.languages=[x['language'] for x in tasks];self.names=[x['task_name'] for x in tasks]
  for e in self.records:
   assert e['transitions']>0 and e['episode_id'] not in self.by_id
   self.by_task[e['task_id']].append(e);self.by_id[e['episode_id']]=e
  assert len(self.by_id)==290 and all(len(v)==29 for v in self.by_task.values())
  assert collections.Counter(e['source'] for e in self.records)=={'expert':50,'successful_rollout':109,'failed_rollout':131}
  old=read(OLD/'data/replay/manifest.json')
  if verify_files:
   for f in m['files']+old['files']:assert sha(f['hdf5'])==f['sha256'],f['hdf5']
  self.manifest_sha256=sha(EXP/'data/performance2/mixed_manifest001.json');self.last_draw=None
 def get(self,episode,index):
  import h5py
  from critic_replay_v1 import extract_transition
  e=self.by_id[episode];p=e['trajectory_path']
  if p not in self.handles:self.handles[p]=h5py.File(p,'r')
  row=extract_transition(self.handles[p][episode],index,e['task_id'],episode);row['episode_success']=e['success'];row['source']=e['source'];return row
 def samples(self,update,batch,seed):
  from critic_replay_v1 import draw_indices
  d=draw_indices(self.by_task,update,batch,seed);self.last_draw=d
  return [self.get(ep,i) for _,ep,i in d],hashlib.sha256(json.dumps(d,separators=(',',':')).encode()).hexdigest()
 def close(self):
  for f in self.handles.values():f.close()
  self.handles.clear()

class Coverage:
 def __init__(self,loader):self.loader=loader;self.total=collections.Counter();self.seen=set();self.sequence=hashlib.sha256()
 def record(self,rows,index):
  counts=collections.Counter(r.get('source','successful_rollout' if r['episode_success'] else 'failed_rollout') for r in rows)
  tasks=collections.Counter(r['task_id'] for r in rows);self.total.update(counts);self.seen.update(r['episode_id'] for r in rows);self.sequence.update(index.encode())
  n=sum(self.total.values());sources={e.get('source','successful_rollout' if e['success'] else 'failed_rollout') for e in self.loader.by_id.values()}
  return {'source_counts':dict(counts),'task_counts':dict(tasks),'cumulative_source_counts':dict(self.total),'cumulative_source_fractions':{k:self.total[k]/n for k in sorted(sources)},'actual_distinct_trajectories':len(self.seen),'indexed_trajectories':len(self.loader.by_id),'sample_sequence_sha256':self.sequence.hexdigest()}
 def final(self):
  return {'sample_sequence_sha256':self.sequence.hexdigest(),'source_counts':dict(self.total),'actual_trajectory_ids':sorted(self.seen),'actual_distinct_trajectories':len(self.seen),'indexed_trajectories':len(self.loader.by_id)}
