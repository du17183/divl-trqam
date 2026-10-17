"""Frozen public vision features and train-calibrated action support, one GPU."""
import json, numpy as np, flax.nnx as nnx, jax, jax.numpy as jnp
from action_sensitive_v3_common import R,D,O,sha,atomic
from openpi.training.libero_lwd_training import build_critic,encoder_config,LiberoTrainingConfig


def main():
    manifest=json.loads((D/'manifest.json').read_text());groups={}
    for row in manifest['records']:groups.setdefault(row['state_id'],[]).append(row)
    source=np.load(R/'artifacts/replay_action_audit_v3/public_features.npz')
    original_meta=json.loads(str(source['train_metadata']));original_task=np.array([r['task_id'] for r in original_meta])
    th=json.loads((R/'artifacts/replay_action_audit_v3/ood_analysis.json').read_text())['calibration']
    images=[];wrists=[];proprio=[];actions=[];returns=[];tasks=[];initial=[];ids=[];phases=[];ks=[]
    for sid,rows in sorted(groups.items()):
        rows=sorted(rows,key=lambda x:x['candidate_index']);assert [r['candidate_index'] for r in rows]==list(range(5))
        with np.load(D/'states'/f'{sid}.npz',allow_pickle=False) as s:
            images.append(s['base']);wrists.append(s['wrist']);proprio.append(s['proprio']);ks.append(int(s['valid_action_mask'].sum()))
            actions.append(s['candidate_normalized'][:,:5].reshape(5,35))
            assert np.count_nonzero(s['candidate_padded32'][:,:,7:])==0
        returns.append([r['future_return'] for r in rows]);tasks.append(rows[0]['task_id']);initial.append(rows[0]['initial_state_id']);ids.append(sid);phases.append(rows[0]['phase'])
    model=build_critic(LiberoTrainingConfig(value_type='scalar'),encoder_config(R))
    @nnx.jit
    def encode(vision,a,b):
        x,_=vision.siglip(a,train=False);y,_=vision.siglip(b,train=False)
        return jnp.concatenate([x.mean(1),y.mean(1)],axis=-1).astype(jnp.float32)
    features=[]
    for index in range(0,len(images),8):
        a=np.stack(images[index:index+8]).astype(np.float32)/127.5-1
        b=np.stack(wrists[index:index+8]).astype(np.float32)/127.5-1
        features.extend(np.asarray(encode(model.encoder.vision,jnp.asarray(a),jnp.asarray(b))))
        print('V3_PUBLIC_FEATURES',index+len(a),len(images),flush=True)
    features=np.stack(features);features/=np.maximum(np.linalg.norm(features,axis=1,keepdims=True),1e-10)
    actions=np.stack(actions).astype(np.float32);support=[];records=[]
    for i,sid in enumerate(ids):
        task=tasks[i];k=ks[i];pool_ids=np.flatnonzero(original_task==task)
        nearest=pool_ids[np.argsort(1-source['train_features'][pool_ids]@features[i])[:10]]
        pool=source['train_actions'][nearest,:k].reshape(10,-1)
        bins=[]
        for j in range(5):
            action=actions[i,j].reshape(5,7)[:k].reshape(-1)
            distance=float(np.sqrt(np.mean((pool-action)**2,axis=1)).min())
            bucket='Near-ID' if distance<=th[str(task)]['p50'] else 'Mid' if distance<=th[str(task)]['p90'] else 'Far-OOD'
            bins.append(bucket);records.append(dict(state_id=sid,task_id=task,candidate_index=j,
                 local_action_rms=distance,bucket=bucket,extreme_p99=distance>th[str(task)]['p99']))
        support.append(bins)
    np.savez_compressed(O/'v3_frozen_features.npz',features=features,proprio=np.stack(proprio),actions=actions,
        returns=np.asarray(returns,np.float32),task=np.asarray(tasks),initial=np.asarray(initial),state_ids=np.asarray(ids),
        phase=np.asarray(phases),support=np.asarray(support),k=np.asarray(ks),
        provenance=np.asarray(json.dumps(dict(encoder='Unchanged publicSigLIP; same weights/normalization as v2; no trained critic',
              v3_manifest_sha256=sha(D/'manifest.json'),ood_calibration_sha256=sha(R/'artifacts/replay_action_audit_v3/ood_analysis.json')))))
    atomic(O/'candidate_support.json',dict(calibration=th,records=records,
        rule='Same immutable original-train p50/p90/p99 thresholds and10public-feature neighbors as v2; no outcome-dependent rebinning'))
    print('V3_FEATURES_COMPLETE',len(ids),flush=True)


if __name__=='__main__':main()
