"""Independent SFT1000 runs; original equations, BC pools and configured budget."""
from mixed_common_v2 import *
from mixed_replay_v1 import MixedReplay,Coverage
import argparse,gc,traceback

def run_spec(run_id):
 runs=read(EXP/'configs/protocol001.json')['actor_runs'];return next(x for x in runs if x['id']==run_id)

def checkpoint_path(run_id,number):
 return EXP/'checkpoints/performance2/actor'/run_id/f'step_{number:06d}'

def make_bundle(method,budget):
 import flax.nnx as nnx
 import jax,jax.numpy as jnp,optax
 from openpi.models.model import restore_params
 from openpi.models.pi0_config import Pi0Config
 from pilot8_common_v1 import config
 a=read(EXP/'configs/protocol001.json')['actor_inherited']
 with jax.default_device(jax.devices()[0]):
  params=restore_params(config()['sft_checkpoint']+'/params',dtype=jnp.float32,sharding=jax.sharding.SingleDeviceSharding(jax.devices()[0]))
  actor=Pi0Config(pi05=True,action_horizon=10,discrete_state_input=False,dtype='float32').load(params);actor.eval();reference=nnx.clone(actor)
  def optimizer(m):return nnx.Optimizer(m,optax.adamw(optax.cosine_decay_schedule(a['learning_rate'],budget),b1=.9,b2=.999,weight_decay=.01),wrt=nnx.Param)
  b=nnx.Dict(actor=actor,reference=reference,opt=optimizer(actor))
  if method in ['G','H']:
   beta=nnx.clone(actor);b.beta=beta;b.betaopt=optimizer(beta);b.dual=nnx.Dict(lam=nnx.Variable(jnp.asarray(1.,jnp.float32)),kl_ema=nnx.Variable(jnp.asarray(0.,jnp.float32)))
 assert all(str(x.dtype)=='float32' for x in jax.tree.leaves(nnx.state(b.actor,nnx.Param)));return b

def load_critic(method,data_source,devices):
 import flax.nnx as nnx
 from critic_model_v1 import build_pretrained
 from critic_restore_current_layout_v1 import restore_tree
 import pilot8_critic_v1 as critic_module
 from pilot8_actor_v5 import place
 value='scalar' if method in ['C','E','G'] else 'divl'
 cp=(OLD/'checkpoints/critic' if data_source=='R0' else EXP/'checkpoints/performance2/critic')/value/'step_002000'
 m=read(cp/'manifest.json');assert m['complete'] and m['update']==2000
 for f,v in m['files'].items():assert sha(cp/f)==v['sha256'],f
 critic=build_pretrained(critic_module.cfg(value));state=nnx.state(critic,nnx.Param);state.replace_by_pure_dict(restore_tree(cp/'critic_params',state.to_pure_dict()));nnx.update(critic,state);critic.eval();place(nnx.Dict(critic=critic),devices)
 return critic,sha(cp/'manifest.json')

def save_actor(b,number,path,meta):
 import flax.nnx as nnx,numpy as np
 import orbax.checkpoint as ocp
 path=Path(path);assert path.absolute().is_relative_to(EXP/'checkpoints/performance2/actor')
 allowed=read(EXP/'configs/checkpoint_storage011.json')['allowed_roots'];assert any(path.resolve().is_relative_to(Path(p).resolve()) for p in allowed)
 path.mkdir(parents=True,exist_ok=False);t=time.monotonic()
 with ocp.PyTreeCheckpointer() as w:
  w.save(str((path/'full_train_state').resolve()),{'state':nnx.state(b).to_pure_dict(),'update':np.asarray(number,np.int32)})
  w.save(str((path/'params').resolve()),nnx.state(b.actor,nnx.Param).to_pure_dict())
  if meta['method']=='B':w.save(str((path/'ema_params').resolve()),nnx.state(b.reference,nnx.Param).to_pure_dict())
 files={p.relative_to(path).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size} for p in path.rglob('*') if p.is_file()}
 m={**meta,'complete':True,'update':number,'next_update':number+1,'files':files,'bytes':sum(v['bytes'] for v in files.values()),'saved_utc':stamp(),'wall_s':time.monotonic()-t,'full_components':['actor','reference','optimizer']+(['beta','beta_optimizer','lambda','kl_ema'] if meta['method'] in ['G','H'] else [])};write(path/'manifest.json',m,True);return m

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args();run=run_spec(a.run_id);method=run['method'];budget=run['updates'];dest=EXP/'metrics/actor'/a.run_id;dest.mkdir(parents=True,exist_ok=False)
 began=time.monotonic();out={'run_id':a.run_id,'method':method,'data':run['data'],'budget':budget,'scheduler':run['scheduler'],'passed':False,'scientific_actor_updates':0,'beta_optimizer_updates':0,'reference_ema_updates':0,'start_utc':stamp(),'devices':4,'batch':32,'precision':'FP32','micro_batch_per_gpu':8,'gradient_accumulation':1};loader=bc_loader=None
 def life(stage):out['stage']=stage;write(dest/'lifecycle.json',{**out,'utc':stamp()});print('ACTOR_STAGE',a.run_id,stage,out['scientific_actor_updates'],flush=True)
 try:
  verify_sources();protected_audit()
  for f,h in read(EXP/'provenance/mixed_actor_sources010.json')['files'].items():assert sha(EXP/f)==h,f
  import flax.nnx as nnx
  import jax,numpy as np
  import pilot8_actor_v5 as old
  from pilot8_critic_v1 import Replay,host_metrics,Telemetry
  from pilot8_common_v1 import config
  from critic_fullbatch_v1 import parameter_digest
  from openpi.models.tokenizer import PaligemmaTokenizer
  from openpi.models.divl_tokenizer import DIVLTokenizer
  from critic_model_v1 import encoder_config
  assert len(jax.devices())==4;jax.config.update('jax_default_matmul_precision','highest');old.verify()
  out['runtime_revision']=old.runtime_receipt()
  # Reuse the accepted original per-method mathematical/layout gate, not its trained Actor.
  execution=read(OLD/f'configs/actor_execution_{method}001.json')
  assert execution['passed'] and execution['batch']==32
  gateid=execution['gates'][method];gp=OLD/'metrics/actor'/gateid/'result.json'
  assert sha(gp)==execution['gate_result_sha256'] and read(gp)['passed']
  assert execution.get('devices_by_method',{}).get(method,execution.get('devices'))==4
  out['inherited_kernel_gate']={'path':str(gp),'sha256':sha(gp),'scientific_updates_reused':0}
  # BC always reads the original 240 Train Rollout, regardless of the main pool.
  bc_loader=Replay('train',verify_files=True);bcdata=old.BCData(bc_loader)
  loader=MixedReplay() if run['data']=='R3' else Replay('train',verify_files=False);coverage=Coverage(loader)
  tok=PaligemmaTokenizer(200);ctok=DIVLTokenizer(encoder_config());critic=None
  if method!='B':critic,out['critic_manifest_sha256']=load_critic(method,run['data'],4)
  b=make_bundle(method,budget);data=old.place(b,4);out.update(initial_actor_sha256=parameter_digest(nnx.state(b.actor,nnx.Param)),initial_reference_sha256=parameter_digest(nnx.state(b.reference,nnx.Param)),initial_checkpoint_manifest_sha256=config()['sft_checkpoint_manifest_sha256'],replay_manifest_sha256=loader.manifest_sha256)
  assert out['initial_actor_sha256']==out['initial_reference_sha256'];critic_digest=parameter_digest(nnx.state(critic)) if critic else None
  gate_results={};pending=None;times=[]
  def batch(n):
   t=time.monotonic();host,idx=old.get_batch(loader,bcdata,tok,ctok,n,32);drawrows=[]
   from critic_replay_v1 import draw_indices
   draw=draw_indices(loader.by_task,n,32,config()['seed'])
   for task,ep,i in draw:
    r=loader.by_id[ep];drawrows.append({'task_id':task,'episode_id':ep,'chunk_index':i,'episode_success':r['success'],'source':r.get('source','successful_rollout' if r['success'] else 'failed_rollout')})
   return host,idx,drawrows,time.monotonic()-t
  def compare_pure(actual,expected):
   pairs=zip(jax.tree.flatten_with_path(actual)[0],jax.tree.flatten_with_path(expected)[0],strict=True);rows=[];maxabs=0.
   for (key,x),(key2,y) in pairs:
    assert key==key2;x=np.asarray(jax.device_get(x));y=np.asarray(y);assert x.shape==y.shape and x.dtype==y.dtype and np.isfinite(x).all() and np.isfinite(y).all()
    err=float(np.max(np.abs(x-y)));maxabs=max(maxabs,err);assert err<=2e-5;rows.append({'key':jax.tree_util.keystr(key),'shape':list(x.shape),'dtype':str(x.dtype),'max_absolute_error':err,'different_elements':int(np.count_nonzero(x!=y))})
   return {'passed':True,'absolute_tolerance':2e-5,'max_absolute_error':maxabs,'all_leaves':rows,'relative_error_used':False}
  def sameconfig(cp,n):
   nonlocal b,data
   from openpi.models.model import Observation
   from openpi.shared.nnx_utils import module_jit
   import jax.numpy as jnp
   host,idx,rows,rs=batch(n+1);obs=Observation(**jax.device_put(jax.tree.map(jnp.asarray,host['obs']),data));noise=jax.device_put(jax.random.normal(jax.random.key(20261013),(32,10,32)),data)
   inference=np.asarray(jax.device_get(jax.block_until_ready(module_jit(b.actor.sample_actions)(jax.random.key(0),obs,noise=noise,num_steps=10)))).copy();digest=parameter_digest(nnx.state(b))
   before_next=jax.tree.map(lambda x:np.asarray(jax.device_get(x)).copy(),nnx.state(b.actor,nnx.Param).to_pure_dict()) if method in ['G','H'] else None
   reference_metrics=host_metrics(old.run_update(b,critic,host,n+1,method,data));expected=jax.tree.map(lambda x:np.asarray(jax.device_get(x)).copy(),nnx.state(b).to_pure_dict());nextdigest=parameter_digest(nnx.state(b));nextnorm=None
   if before_next is not None:
    sq=count=0
    for x,y in zip(jax.tree.leaves(nnx.state(b.actor,nnx.Param).to_pure_dict()),jax.tree.leaves(before_next),strict=True):
     delta=np.asarray(jax.device_get(x)).astype(np.float64)-y.astype(np.float64);sq+=float(np.sum(delta*delta));count+=delta.size
    nextnorm={'parameter_update_l2':float(np.sqrt(sq)),'parameter_update_rms':float(np.sqrt(sq/count))};del before_next
   del b,obs,noise;gc.collect();jax.clear_caches();b=make_bundle(method,budget);data=old.place(b,4);old.restore(b,cp);assert parameter_digest(nnx.state(b))==digest
   obs=Observation(**jax.device_put(jax.tree.map(jnp.asarray,host['obs']),data));noise=jax.device_put(jax.random.normal(jax.random.key(20261013),(32,10,32)),data)
   fresh=np.asarray(jax.device_get(jax.block_until_ready(module_jit(b.actor.sample_actions)(jax.random.key(0),obs,noise=noise,num_steps=10))));inference_error=float(np.max(np.abs(fresh-inference)));assert inference_error<=2e-5 and np.array_equal(fresh,inference)
   t=time.monotonic();actual_metrics=host_metrics(old.run_update(b,critic,host,n+1,method,data));dt=time.monotonic()-t;check=compare_pure(nnx.state(b).to_pure_dict(),expected);assert parameter_digest(nnx.state(b))==nextdigest;assert all(abs(actual_metrics[k]-reference_metrics[k])<=2e-5 for k in reference_metrics)
   gate={'passed':True,'checkpoint_update':n,'absolute_tolerance':2e-5,'complete_restored_digest':digest,'next_update_digest':nextdigest,'fresh_jit_inference_maxabs':inference_error,'next_update':check,'temporary_next_updates':2 if n==budget else 1,'scientific_next_update_reused':n<budget};write(dest/f'checkpoint_{n:06d}_sameconfig.json',gate,True);gate_results[str(n)]=gate
   del expected,obs,noise,inference,fresh;gc.collect()
   if n==budget:old.restore(b,cp);assert parameter_digest(nnx.state(b))==digest;return None
   return host,idx,rows,rs,dt,actual_metrics,nextnorm
  life('TRAINING')
  with Telemetry() as telemetry,(dest/'train_steps.jsonl').open('x') as log:
   for n in range(1,budget+1):
    assert not (EXP/'human_hold.json').exists(),'HUMAN_HOLD';t0=time.monotonic()
    precalculated_norm=None
    if pending is None:
     host,idx,rows,rs=batch(n);before=None
     if method in ['G','H']:before=jax.tree.map(lambda x:np.asarray(jax.device_get(x)).copy(),nnx.state(b.actor,nnx.Param).to_pure_dict())
     t=time.monotonic();values=host_metrics(old.run_update(b,critic,host,n,method,data));dt=time.monotonic()-t
    else:host,idx,rows,rs,dt,values,precalculated_norm=pending;pending=None;before=None
    norm=precalculated_norm
    if before is not None:
     sq=count=0
     for x,y in zip(jax.tree.leaves(nnx.state(b.actor,nnx.Param).to_pure_dict()),jax.tree.leaves(before),strict=True):
      delta=np.asarray(jax.device_get(x)).astype(np.float64)-y.astype(np.float64);sq+=float(np.sum(delta*delta));count+=delta.size
     norm={'parameter_update_l2':float(np.sqrt(sq)),'parameter_update_rms':float(np.sqrt(sq/count))};del before
    if method in ['C','D','E','F']:assert parameter_digest(nnx.state(b.reference,nnx.Param))==out['initial_reference_sha256'] if n in [1,*run['checkpoints']] else True
    out.update(scientific_actor_updates=n,beta_optimizer_updates=n if method in ['G','H'] else 0,reference_ema_updates=n if method in ['G','H','B'] else 0)
    bc_counts={k:sum(x['source']==k for x in idx['bc_indices']) for k in ['legal_train_expert','successful_rollout']}
    cov={**coverage.record(rows,idx['replay_index_sha256']),'main_sample_draws':rows,'main_rl_samples_used':32} if method!='B' else {'scope':'main Replay batch prepared but unused by BC-only optimization','main_rl_samples_used':0}
    rec={'run_id':a.run_id,'method':method,'update':n,'batch':32,'learning_rate':float(read(EXP/'configs/protocol001.json')['actor_inherited']['learning_rate']*.5*(1+np.cos(np.pi*(n-1)/budget))),'read_s':rs,'compute_s':dt,'diagnostic_and_io_s':max(0,time.monotonic()-t0-rs-dt),**idx,**cov,**values,'bc_supervised_samples_used':32 if method in ['B','E','F','G','H'] else 0,'bc_source_counts':bc_counts if method in ['B','E','F','G','H'] else {},'gradient_clip_threshold':1.,'gradient_clip_scale':float(min(1.,1./max(values.get('grad_norm',0.),1e-30))),'parameter_update_norm':norm,'beta_optimizer_updates':out['beta_optimizer_updates'],'reference_ema_updates':out['reference_ema_updates']}
    log.write(json.dumps(rec)+'\n');log.flush();times.append((rs,dt));life('TRAINING')
    if n in run['checkpoints']:
     life('SAVE_AND_SAME_CONFIGURATION_GATE_'+str(n));cp=checkpoint_path(a.run_id,n)
     saved=save_actor(b,n,cp,{**out,'passed':True,'scientific_actor':True,'seed':config()['seed'],'kernel_gate':'original accepted same four-GPU FP32 equations; checkpoint gate stored separately before evaluation','sample_sequence_sha256':coverage.sequence.hexdigest()})
     hold=Replay('holdout',verify_files=False);heldrows,heldidx=hold.samples(1,16,20261012);ahost={'obs':old.actor_observations(heldrows,loader.languages,tok,read(config()['normalization']))};hold.close()
     dr=old.drift(b,ahost,data);dr['fixed_heldout_indices_sha256']=heldidx
     if method in ['G','H']:
      betadr=old.drift(nnx.Dict(actor=b.beta,reference=b.reference),ahost,data);dr.update(beta_action_drift_rms_real7=betadr['actor_action_drift_rms_real7'],beta_parameter_drift_rms=betadr['actor_parameter_drift_rms'])
     out.setdefault('checkpoint_drift',{})[str(n)]=dr;out['drift']=dr;write(dest/f'checkpoint_{n:06d}_drift.json',dr,True);pending=sameconfig(cp,n)
   out['telemetry']=telemetry.rows
  if critic:assert parameter_digest(nnx.state(critic))==critic_digest,'Frozen Critic changed'
  out.update(passed=True,stage='COMPLETED_AND_CHECKPOINTS_ADMITTED',coverage=coverage.final(),checkpoint_gate_results=gate_results,all_saved_checkpoints_numerically_admitted=True,steady_compute_s=float(np.mean([x[1] for x in times[2:]])),steady_read_s=float(np.mean([x[0] for x in times[2:]])))
 except Exception:out.update(stage='ENGINEERING_STOP',error=traceback.format_exc());print(out['error'],flush=True)
 finally:
  if loader:loader.close()
  if bc_loader:bc_loader.close()
  out.update(ended_utc=stamp(),wall_s=time.monotonic()-began);out['allocated_gpu_hours']=4*out['wall_s']/3600;write(dest/'result.json',out,True);life(out['stage'])
 sys.exit(0 if out['passed'] else 1)
if __name__=='__main__':main()
