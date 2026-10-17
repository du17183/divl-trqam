"""Read-only real EMA V bootstrap for target audit; no optimizer or retraining."""
import argparse,json,gc,time,numpy as np,pandas as pd
import jax,jax.numpy as jnp,flax.nnx as nnx,orbax.checkpoint as ocp
from openpi.models.divl_tokenizer import DIVLTokenizer
from openpi.models.divl_losses import categorical_quantile,adaptive_quantile_level,divl_q_target
from openpi.shared import normalize
from openpi.training.divl_libero_adapter import make_batch
from openpi.training.libero_lwd_training import LiberoTrainingConfig,build_critic,encoder_config
from horizon_common import R,D,O,atomic,sha,load,safe
def main():
 p=argparse.ArgumentParser();p.add_argument('--method',choices=['scalar','divl'],required=True);a=p.parse_args();method=a.method
 started=time.monotonic();requests=pd.read_parquet(O/'target_requests.parquet').to_dict('records')
 cp=R/'checkpoints/critic_replay_v2'/method/'step2000';protocol=json.loads((cp.parent/'protocol.json').read_text())
 cfg=LiberoTrainingConfig(**protocol['config']);assert protocol['td_chunks']==10 and cfg.gamma==.9999
 enc=encoder_config(R);tokenizer=DIVLTokenizer(enc);norm=normalize.load(R/'checkpoints/pi05_libero/assets/physical-intelligence/libero')
 critic=build_critic(cfg,enc)
 with ocp.PyTreeCheckpointer() as ck:restored=ck.restore(cp.resolve())
 params=nnx.state(critic,nnx.Param);params.replace_by_pure_dict(restored['target']);nnx.update(critic,jax.tree.map(jax.device_put,params));critic.eval()
 del restored,params;gc.collect()
 @nnx.jit
 def value(model,obs):return model.v_head(model.encoder(**obs,train=False))
 output=[]
 batchsize=32
 fields=['base','wrist','state','prompt','actions','action_mask','rewards','discounts']
 for start in range(0,len(requests),batchsize):
  rows=requests[start:start+batchsize];eps=[load(r['path'],fields) for r in rows];batch=make_batch([(e,0) for e in eps],tokenizer,norm,td_steps=10)
  n=len(rows)
  if n<batchsize:
   # Fixedbatch compile, copied observations only; extra values discarded.
   batch=jax.tree.map(lambda x:jnp.concatenate([x,jnp.repeat(x[-1:],batchsize-n,axis=0)]),batch)
  nextv=value(critic,batch['next_obs'])
  if method=='scalar':v=nextv;tau=np.full(batchsize,cfg.expectile);target=batch['rewards']+batch['discounts']*v
  else:
   tau=adaptive_quantile_level(nextv,tau_base=cfg.tau_base,alpha=cfg.alpha,tau_min=0,tau_max=1)
   v=categorical_quantile(nextv,critic.v_head.atoms,tau)
   target=divl_q_target(batch['rewards'],batch['discounts'],nextv,critic.v_head.atoms,tau_base=cfg.tau_base,alpha=cfg.alpha,tau_min=0,tau_max=1)
  v=np.asarray(v);target=np.asarray(target);tau=np.asarray(tau)
  for j,row in enumerate(rows):
   assert abs(float(batch['rewards'][j])-row['td_window_reward'])<1e-6 and abs(float(batch['discounts'][j])-row['td_discount'])<1e-6
   assert np.array_equal(eps[j]['state'][row['bootstrap_state_index']],eps[j]['state'][min(10,len(eps[j]['rewards']))])
   expected=row['td_window_reward']+row['td_discount']*float(v[j]);assert abs(expected-float(target[j]))<1e-6
   output.append({**row,'method':method,'frozen_EMA_next_value':float(v[j]),'tau':float(tau[j]),'Target_C':float(target[j]),'target_formula_error':abs(expected-float(target[j]))})
  if start%160==0:
   atomic(O/f'target_{method}_progress.json',dict(completed=len(output),target=len(requests),elapsed_seconds=time.monotonic()-started,training=False));print('FROZEN_TARGET',method,len(output),flush=True)
 pd.DataFrame(output).to_parquet(O/f'target_{method}.parquet',index=False)
 atomic(O/f'target_{method}_complete.json',dict(complete=True,method=method,rows=len(output),elapsed_seconds=time.monotonic()-started,
   checkpoint=str(cp),weight_key='target (EMA, not onlinecritic)',checkpoint_protocol_hash=sha(cp.parent/'protocol.json'),checkpoint_metrics_hash=sha(cp.parent/'metrics_step2000.json'),
   no_optimizer=True,actor_updated=False,critic_updated=False))
 print('FROZEN_TARGET_COMPLETE',method,len(output),flush=True)
if __name__=='__main__':main()
