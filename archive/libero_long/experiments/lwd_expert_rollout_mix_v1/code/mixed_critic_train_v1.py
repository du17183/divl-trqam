"""Fresh paired Mixed2000; frozen original FP32 Critic mathematics and layout."""
from mixed_common_v2 import *
from mixed_replay_v1 import MixedReplay,Coverage
import argparse,gc,traceback

def verify_new_sources():
 verify_sources();protected_audit()
 for f,h in read(EXP/'provenance/mixed_training_sources009.json')['files'].items():assert sha(EXP/f)==h,f

def restore_local(b,cp):
 import flax.nnx as nnx
 import numpy as np
 import jax
 from critic_restore_current_layout_v1 import restore_tree
 m=read(cp/'manifest.json');assert m['complete']
 for f,v in m['files'].items():assert sha(cp/f)==v['sha256'],f
 state=nnx.state(b);loaded=restore_tree(cp/'full_train_state',{'state':state.to_pure_dict(),'update':np.asarray(m['update'],np.int32)})
 assert int(loaded['update'])==m['update'];state.replace_by_pure_dict(loaded['state']);nnx.update(b,state);jax.block_until_ready(nnx.state(b));return m

def main():
 p=argparse.ArgumentParser();p.add_argument('--method',choices=['scalar','divl'],required=True);a=p.parse_args()
 d=EXP/'metrics/critic'/('mixed_'+a.method+'_train001');d.mkdir(parents=True,exist_ok=False)
 start=time.monotonic();out={'method':a.method,'passed':False,'scientific_critic_updates':0,'start_utc':stamp(),'devices':4,'layout':'data_parallel','global_batch':128,'compute_dtype':'float32','master_dtype':'float32','initialization':'original public initialization; no old trained Critic loaded','actor_updates':0};loader=None
 def lifecycle(stage):out['stage']=stage;write(d/'lifecycle.json',{**out,'utc':stamp()});print('MIXED_CRITIC_STAGE',stage,flush=True)
 try:
  verify_new_sources()
  import flax.nnx as nnx
  import jax,numpy as np
  import pilot8_critic_v1 as old
  from critic_replay_v1 import collate
  from critic_model_v1 import encoder_config
  from critic_fullbatch_v1 import parameter_digest
  from critic_gpu_gate_contract_v1 import compare_snapshot
  from critic_pretrained_gpu_gate_v1 import bind_inference
  from openpi.models.divl_tokenizer import DIVLTokenizer
  assert len(jax.devices())==4;jax.config.update('jax_default_matmul_precision','highest')
  loader=MixedReplay();coverage=Coverage(loader);c=old.cfg(a.method);assert c.updates==2000
  tok=DIVLTokenizer(encoder_config());norm=read(read(EXP/'configs/protocol001.json')['inherited']['normalization'])
  old.EXP=EXP # Only output containment in frozen save/snapshot; cfg() retains original common config.
  b=old.bundle(a.method);data=old.layout(b,4);out.update(replay_manifest_sha256=loader.manifest_sha256,initial_complete_state_sha256=parameter_digest(nnx.state(b)),gpu_receipt=old.receipt())
  meta={'value_type':a.method,'seed':c.seed,'global_batch':128,'scientific_critic':True,'compute_dtype':'float32','master_dtype':'float32','protocol_sha256':sha(EXP/'configs/protocol001.json'),'normalization_sha256':sha(read(EXP/'configs/protocol001.json')['inherited']['normalization']),'replay_manifest_sha256':loader.manifest_sha256,'devices':4,'layout':'data_parallel','source_quota':None}
  times=[];gate=EXP/'data/gates'/('mixed_'+a.method+'_sameconfig001');gate.mkdir(parents=True,exist_ok=False)
  def batch(n):
   t=time.monotonic();rows,idx=loader.samples(n,128,c.seed);host=collate(rows,loader.languages,tok,norm);return rows,idx,host,time.monotonic()-t
  rows1,idx1,host1,read1=batch(1);rows2,idx2,host2,read2=batch(2)
  lifecycle('SAME_CONFIGURATION_SAVE_RESTORE_NEXT_UPDATE_GATE')
  t=time.monotonic();m1,_=old.update(b,c,data,host1);dt1=time.monotonic()-t;m1=old.host_metrics(m1)
  digest1=parameter_digest(nnx.state(b));cp=EXP/'data/gates'/('mixed_'+a.method+'_state1')
  old.save(b,1,cp,{**meta,'scientific_critic':False,'scope':'first-prefix same-configuration gate, not a separately trained Critic'})
  inference=bind_inference();import jax.numpy as jnp
  bh=jax.device_put(jax.tree.map(jnp.asarray,host1),data);pred1=jax.block_until_ready(inference(b.critic,bh['obs'],bh['actions'],bh['action_mask']))
  old.snapshot(pred1,gate/'inference1');del inference,pred1,bh
  m2ref,_=old.update(b,c,data,host2);m2ref=old.host_metrics(m2ref);digest2=parameter_digest(nnx.state(b));old.snapshot(nnx.state(b).to_pure_dict(),gate/'state2')
  del b;gc.collect();jax.clear_caches();b=old.bundle(a.method);data=old.layout(b,4);restore_local(b,cp)
  assert parameter_digest(nnx.state(b))==digest1,'Full state restore digest mismatch'
  inference=bind_inference();bh=jax.device_put(jax.tree.map(jnp.asarray,host1),data)
  pred1=jax.block_until_ready(inference(b.critic,bh['obs'],bh['actions'],bh['action_mask']));predcheck=compare_snapshot(pred1,gate/'inference1');del inference,pred1,bh
  t=time.monotonic();m2,_=old.update(b,c,data,host2);dt2=time.monotonic()-t;m2=old.host_metrics(m2)
  statecheck=compare_snapshot(nnx.state(b).to_pure_dict(),gate/'state2');assert parameter_digest(nnx.state(b))==digest2,'Fresh JIT next-update digest mismatch'
  assert all(abs(m2[k]-m2ref[k])<=2e-5 for k in m2)
  write(d/'same_configuration_gate.json',{'passed':True,'absolute_tolerance':2e-5,'digest1':digest1,'digest2':digest2,'complete_restore_digest_equal':True,'fresh_jit_inference':predcheck,'next_update_full_state':statecheck,'prefix_reused':True,'physical_optimizer_transactions':3,'scientific_updates_after_admission':2,'cross_layout_gate_repeated':False,'original_validated_four_gpu_mathematics_reused':True},True)
  out['same_configuration_gate_passed']=True
  with old.Telemetry() as telemetry:
   with (d/'train_steps.jsonl').open('x') as log:
    for n in range(1,2001):
     if n<=2:rows,idx,read_s,dt,values=(rows1,idx1,read1,dt1,m1) if n==1 else (rows2,idx2,read2,dt2,m2)
     else:
      assert not (EXP/'human_hold.json').exists(),'HUMAN_HOLD';rows,idx,host,read_s=batch(n);t=time.monotonic();values,_=old.update(b,c,data,host);dt=time.monotonic()-t;values=old.host_metrics(values)
     record={'update':n,'batch':128,'sample_index_sha256':idx,'samples':[(x['task_id'],x['episode_id'],x['chunk_index']) for x in rows],'read_s':read_s,'compute_s':dt,'total_s':dt+read_s,**coverage.record(rows,idx),**values}
     log.write(json.dumps(record)+'\n');log.flush();times.append((read_s,dt));out['scientific_critic_updates']=n;lifecycle('TRAINING')
     if n in [250,500,1000,2000]:
      lifecycle('SAVE_STEP_'+str(n));old.save(b,n,EXP/'checkpoints/performance2/critic'/a.method/f'step_{n:06d}',{**meta,'sample_sequence_sha256':coverage.sequence.hexdigest(),'passed':True,'scientific_critic_updates':n})
  out.update(passed=True,stage='COMPLETED2000_FROZEN',coverage=coverage.final(),steady_compute_s=float(np.mean([x[1] for x in times[2:]])),steady_read_s=float(np.mean([x[0] for x in times[2:]])),telemetry=telemetry.rows)
  out['steady_samples_s']=128/(out['steady_compute_s']+out['steady_read_s'])
 except Exception:out.update(stage='ENGINEERING_STOP',error=traceback.format_exc());print(out['error'],flush=True)
 finally:
  if loader:loader.close()
  out.update(ended_utc=stamp(),wall_s=time.monotonic()-start);out['allocated_gpu_hours']=4*out['wall_s']/3600;write(d/'result.json',out,True);lifecycle(out['stage'])
 sys.exit(0 if out['passed'] else 1)
if __name__=='__main__':main()
