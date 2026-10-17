"""Use the actual frozen pi05_libero vision weights, not a critic encoder."""
import json,numpy as np,jax,jax.numpy as jnp,flax.nnx as nnx
from openpi.policies import policy_config
from openpi.training import config
from action_sensitive_v4_common import R,D,O,REP,atomic,sha

def main():
 assert not (O/'policy_features.npz').exists()
 O.mkdir(parents=True,exist_ok=True);REP.mkdir(parents=True,exist_ok=True)
 old=np.load(R/'artifacts/replay_action_audit_v3/public_features.npz',allow_pickle=False)
 meta=json.loads(str(old['train_metadata']));selection=json.loads((R/'data/action_sensitive_replay_v3/state_selection_manifest.json').read_text())['records']
 print('METADATA_EXAMPLE',meta[0],flush=True)
 policy=policy_config.create_trained_policy(config.get_config('pi05_libero'),R/'checkpoints/pi05_libero',sample_kwargs={'num_steps':10})
 @nnx.jit
 def encode(img,a,b):
  x,_=img(a,train=False);y,_=img(b,train=False)
  return jnp.concatenate([x.mean(1),y.mean(1)],-1).astype(jnp.float32)
 cached={};observations=[]
 for row in meta:
  path=row['path'] if 'path' in row else row.get('trajectory_path')
  if path is None:
   original=json.loads((R/'data/pilot_libero_v1/manifest.json').read_text())['records']
   path=next(r['path'] for r in original if r['episode_id']==row['episode_id'])
  if path not in cached:
   with np.load(path,allow_pickle=False) as e:cached[path]={k:e[k] for k in ['base','wrist','state','prompt']}
  e=cached[path];i=row.get('chunk_index',row.get('index'))
  if i is None:i=row['timestep']
  observations.append(dict(**{'observation/image':e['base'][i],'observation/wrist_image':e['wrist'][i],'observation/state':e['state'][i]},prompt=str(e['prompt'])))
 for row in selection:
  with np.load(R/'data/action_sensitive_replay_v3/states'/f'{row["state_id"]}.npz',allow_pickle=False) as e:
   observations.append(dict(**{'observation/image':e['base'],'observation/wrist_image':e['wrist'],'observation/state':e['proprio']},prompt=str(e['instruction'])))
 features=[];keys=None
 for start in range(0,len(observations),8):
  transformed=[policy._input_transform(obs.copy()) for obs in observations[start:start+8]]
  if keys is None:
   keys=[k for k in transformed[0]['image'] if bool(transformed[0]['image_mask'][k])]
   assert len(keys)==2
   print('REAL_CAMERAS',keys,flush=True)
  imgs=[np.stack([x['image'][k] for x in transformed]) for k in keys]
  imgs=[a.astype(np.float32)/127.5-1 if a.dtype==np.uint8 else a for a in imgs]
  features.extend(np.asarray(encode(policy._model.PaliGemma.img,jnp.asarray(imgs[0]),jnp.asarray(imgs[1]))))
  if start%80==0:print('V4_POLICY_FEATURES',start+len(transformed),len(observations),flush=True)
 features=np.stack(features);features/=np.maximum(np.linalg.norm(features,axis=1,keepdims=True),1e-10)
 np.savez_compressed(O/'policy_features.npz',pool_features=features[:len(meta)],state_features=features[len(meta):],
  pool_actions=old['train_actions'],pool_masks=old['train_masks'],pool_metadata=np.asarray(json.dumps(meta)),
  state_ids=np.asarray([r['state_id'] for r in selection]),proprio=np.stack([x['observation/state'] for x in observations[len(meta):]]))
 atomic(O/'feature_provenance.json',dict(actual_frozen_pi05_checkpoint=str(R/'checkpoints/pi05_libero'),config='pi05_libero',
  weights='Actual pi05_libero PaliGemma.img, official input transforms, per-camera mean-pooled tokens, L2 normalization',
  cameras=keys,rows=len(observations),state_ids_frozen_before_candidates=True,
  contrast_v3='v3 used a separate public SigLIP critic-foundation encoder; v4 A/B use actual frozen actor visual weights',
  no_critic_embedding=True,no_oracle_inputs=True,train_only=True))
 print('V4_FEATURES_COMPLETE',features.shape,flush=True)

if __name__=='__main__':main()
