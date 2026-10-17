"""Frozen heldout1000 and old400 independent action candidates; no updates."""
from mixed_common_v2 import *
import argparse,gc,traceback,itertools

def main():
 p=argparse.ArgumentParser();p.add_argument('--method',choices=['scalar','divl'],required=True);a=p.parse_args();began=time.monotonic();out={'passed':False,'method':a.method,'scientific_updates':0,'start_utc':stamp()}
 try:
  verify_sources();protected_audit()
  for f,h in read(EXP/'provenance/mixed_diagnostics_sources015.json')['files'].items():assert sha(EXP/f)==h,f
  import flax.nnx as nnx
  import jax,jax.numpy as jnp,numpy as np
  from scipy.stats import spearmanr
  import pilot8_critic_v1 as old
  from mixed_critic_train_v1 import restore_local
  from critic_replay_v1 import collate
  from critic_model_v1 import encoder_config
  from critic_fullbatch_v1 import parameter_digest
  from openpi.models.divl_tokenizer import DIVLTokenizer
  from physical_report_supplement_v1 import aggregate
  assert len(jax.devices())==1;jax.config.update('jax_default_matmul_precision','highest')
  loader=old.Replay('holdout',verify_files=True);tok=DIVLTokenizer(encoder_config());norm=read(read(EXP/'configs/protocol001.json')['inherited']['normalization']);cases=read(OLD/'configs/physical_diagnostic002.json')['cases'];out['datasets']={}
  @nnx.jit
  def score(model,batch):
   z=model.encoder(**batch['obs'],train=False);return jnp.minimum(*model.q_head(z,batch['actions'],batch['action_mask']))
  for source in ['R0','R3']:
   cp=(OLD/'checkpoints/critic' if source=='R0' else EXP/'checkpoints/performance2/critic')/a.method/'step_002000';b=old.bundle(a.method);data=old.layout(b,1);restore_local(b,cp);before=parameter_digest(nnx.state(b));diagnostic={}
   old.diagnostics(b,loader,tok,norm,data,a,diagnostic)
   records=diagnostic['records'];held=[];calibration=[]
   for task in range(10):
    for scope in ['complete_MC','censored_observed_suffix','all_observed_suffix']:
     rr=[x for x in records if x['task_id']==task and (scope=='all_observed_suffix' or x['complete_return_known']==(scope=='complete_MC'))]
     err=np.asarray([x['q']-x['observed_discounted_return'] for x in rr]);qs=[x['q'] for x in rr];gs=[x['observed_discounted_return'] for x in rr]
     held.append({'method':a.method,'data':source,'task_id':task,'scope':scope,'windows':len(rr),'physical_sources':len({x['state_source_id'] for x in rr}),'mse':float(np.mean(err*err)) if len(rr) else None,'rmse':float(np.sqrt(np.mean(err*err))) if len(rr) else None,'mean_absolute_td_error':float(np.mean([x['td_error'] for x in rr])) if rr else None,'spearman':float(spearmanr(qs,gs).statistic) if len(rr)>1 and np.std(qs)>0 and np.std(gs)>0 else None})
    full=sorted([x for x in records if x['task_id']==task and x['complete_return_known']],key=lambda x:x['q'])
    for bin_id,indices in enumerate(np.array_split(np.arange(len(full)),5)):
     if not len(indices):continue
     rr=[full[int(i)] for i in indices];calibration.append({'method':a.method,'data':source,'task_id':task,'bin':bin_id,'windows':len(rr),'mean_q':float(np.mean([x['q'] for x in rr])),'mean_complete_MC':float(np.mean([x['observed_discounted_return'] for x in rr]))})
   states=[];rawscores=[];input_files={}
   for case in cases:
    oldpath=OLD/f"metrics/physical002/case_{case['case_id']:03d}.json";d=read(oldpath);assert d['passed'] and d['complete'];input_files[str(oldpath)]=sha(oldpath);row=loader.get(case['episode_id'],case['chunk_index']);branches=[x for x in d['rows'] if x['valid'] and x['ranking_candidate']];assert len(branches)==4
    for branch in branches:
     assert sha(branch['trace_path'])==branch['trace_sha256'];input_files[branch['trace_path']]=branch['trace_sha256']
     with np.load(branch['trace_path']) as f:actions=f['normalized_h5_actions'].copy()
     host=collate([row],loader.languages,tok,norm);host['actions'][0]=actions;bh=jax.device_put(jax.tree.map(jnp.asarray,host),data);q=float(np.asarray(jax.device_get(jax.block_until_ready(score(b.critic,bh))))[0]);branch['mixed_score']=q
     rawscores.append({'case_id':case['case_id'],'task_id':case['task_id'],'state_source_id':case['state_source_id'],'candidate':branch['candidate'],'q':q,'observed_return':branch['return'],'return_complete':not branch['censored'],'method':a.method,'data':source})
    pairs=[(x,y) for x,y in itertools.combinations(branches,2) if abs(x['return']-y['return'])>1e-12]
    ranking=float(np.mean([(x['mixed_score']-y['mixed_score'])*(x['return']-y['return'])>0 for x,y in pairs])) if pairs else None
    by={x['candidate']:x for x in branches};base,small=by['baseline'],by['independent_small'];near=None if abs(base['return']-small['return'])<1e-12 else float((base['mixed_score']-small['mixed_score'])*(base['return']-small['return'])>0)
    qs=[x['mixed_score'] for x in branches];gs=[x['return'] for x in branches]
    states.append({'method':a.method,'data':source,'task_id':case['task_id'],'case_id':case['case_id'],'state_source_id':case['state_source_id'],'action_pairs':len(pairs),'ranking':ranking,'near_id_ranking':near,'spearman':float(spearmanr(qs,gs).statistic) if np.std(qs)>0 and np.std(gs)>0 else None})
   assert parameter_digest(nnx.state(b))==before
   out['datasets'][source]={'checkpoint':str(cp),'checkpoint_sha256':sha(cp/'manifest.json'),'readonly_complete_digest':before,'records':records,'heldout':held,'calibration':calibration,'per_state_ranking':states,'candidate_q_scores':rawscores,'ranking':{k:aggregate(states,k) for k in ['ranking','near_id_ranking','spearman']},'frozen_branch_inputs':input_files,'heldout_count':len(records),'complete_MC_windows':sum(x['complete_return_known'] for x in records),'censored_windows':sum(not x['complete_return_known'] for x in records),'AUROC':None,'AUROC_reason':'Single-class per-task subset has no valid AUROC; no substituted accuracy'}
   del b,bh;gc.collect();jax.clear_caches()
  x=out['datasets']['R0']['records'];y=out['datasets']['R3']['records'];assert [(r['episode_id'],r['chunk_index'],r['state_source_id']) for r in x]==[(r['episode_id'],r['chunk_index'],r['state_source_id']) for r in y]
  loader.close();out['passed']=True
 except Exception:out['error']=traceback.format_exc();print(out['error'],flush=True)
 finally:out.update(ended_utc=stamp(),wall_s=time.monotonic()-began);write(EXP/'metrics/diagnostics'/f'{a.method}_comparison015.json',out,True)
 sys.exit(0 if out['passed'] else 1)
if __name__=='__main__':main()
