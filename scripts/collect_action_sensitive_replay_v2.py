# Public release: deployment paths anonymized; algorithm equations unchanged.
"""Train-partition local branches, native rewards and unchanged frozen continuation."""
import argparse,json,hashlib,time,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from libero.libero import benchmark
from examples.libero.main import _get_libero_env,LIBERO_ENV_RESOLUTION
from examples.libero.collect_divl import collect_episode,policy_observation
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from openpi.training.libero_replay_contract import behavior_sequences,validate_episode
from libero_snapshot_v2 import migrate_v1_for_branch,restore_snapshot

R=Path('/opt/divl-trqam');D=R/'data/action_sensitive_replay_v2'
SEED=2026100717

class NativeAudit:
    def __init__(self,env,k):self.env=env;self.k=k;self.native=[];self.next_state=None
    def step(self,a):
        out=self.env.step(a);self.native.append(float(out[1]))
        if len(self.native)==self.k:self.next_state=self.env.get_sim_state().copy()
        return out
    def check_success(self):return self.env.check_success()

class BranchPolicy:
    def __init__(self,client,candidate,seed,index,env,source,baseline):
        self.client=client;self.candidate=candidate;self.seed=seed;self.index=index;self.calls=0;self.env=env;self.source=source;self.baseline=baseline
        self.max_state_error=0.;self.max_pixel_error=0.;self.max_plan_error=0.
    def infer(self,obs):
        idx=self.index+self.calls
        if self.baseline and idx<len(self.source['rewards']):
            self.max_state_error=max(self.max_state_error,float(abs(self.env.get_sim_state()-self.source['snapshot_states'][idx]).max()))
            self.max_pixel_error=max(self.max_pixel_error,float(abs(obs['observation/image'].astype(float)-self.source['base'][idx]).max()),float(abs(obs['observation/wrist_image'].astype(float)-self.source['wrist'][idx]).max()))
        if self.calls==0:plan=self.candidate
        else:
            plan=np.asarray(self.client.infer({**obs,'_noise_seed':self.seed*1000+idx})['actions'],np.float32)
            if self.baseline and idx<len(self.source['policy_action_plan']):self.max_plan_error=max(self.max_plan_error,float(abs(plan-self.source['policy_action_plan'][idx]).max()))
        self.calls+=1;return dict(actions=plan)

def episode_group(row,fractions,port,features_path):
    started=time.monotonic();f=np.load(features_path);allmeta=json.loads(str(f['train_metadata']));acts=f['train_actions'];masks=f['train_masks']
    ids=[n for n,m in enumerate(allmeta) if m['task_id']==row['task_id']];task_std=acts[ids][masks[ids]].std(0)
    meta_to_index={(m['episode_id'],m['index']):n for n,m in enumerate(allmeta)};feat=f['train_features']
    low=f['action_low'];high=f['action_high'];f.close()
    with np.load(row['path'],allow_pickle=False) as z:source={k:z[k] for k in z.files}
    split=json.loads((R/'configs/splits/libero_factorial_v1.json').read_text());assert row['initial_state_id'] in split['tasks'][row['task_name']]['train']
    task=benchmark.get_benchmark_dict()['libero_10']().get_task(row['task_id']);env,prompt=_get_libero_env(task,LIBERO_ENV_RESOLUTION,row['seed']);client=WebsocketClientPolicy('127.0.0.1',port)
    env.seed(row['seed']);env.reset();records=[]
    try:
        for fraction in fractions:
            i=min(len(source['rewards'])-1,int(len(source['rewards'])*fraction));sid=row['episode_id']+f'_chunk{i:03d}'
            target=D/'states'/f'{sid}.json'
            if target.exists():
                saved=json.loads(target.read_text());assert saved['protocol_sha256']==hashlib.sha256((D/'protocol.json').read_bytes()).hexdigest();records.extend(saved['records']);continue
            flat=source['snapshot_states'][i];hidden=migrate_v1_for_branch(env,flat,source['snapshot_metadata'][i],source['model_xml']);k=int(source['lengths'][i]);remaining=520-int(source['lengths'][:i].sum())
            n=meta_to_index[(row['episode_id'],i)];others=[j for j in ids if allmeta[j]['episode_id']!=row['episode_id']];nn=sorted(others,key=lambda j:1-float(feat[n]@feat[j]))[:10]
            local_std=acts[nn,:k].reshape(-1,7).std(0);scale=np.clip(local_std,.1*task_std,task_std)
            local_seed=SEED+row['task_id']*100000+row['initial_state_id']*1000+row['repetition']*100+i
            rng=np.random.default_rng(local_seed);direction=rng.normal(size=(1,7)).astype(np.float32)
            base=source['actions'][i].copy();bn=2*(base-low)/(high-low+1e-6)-1
            choices=[('behavior',base)]
            for size in [.25,.5]:
                for sign in [-1,1]:
                    raw=np.clip((bn+size*sign*direction*scale+1)/2*(high-low+1e-6)+low,-1,1).astype(np.float32);raw[k:]=0
                    choices.append((f'perturb_{size}_{sign}',raw))
            state_records=[]
            for j,(name,candidate) in enumerate(choices):
                obs=restore_snapshot(env,flat,hidden);error=float(abs(env.get_sim_state()-flat).max());assert error==0
                p=policy_observation(obs,prompt);assert float(abs(p['observation/state']-source['state'][i]).max())<1e-5
                policy=BranchPolicy(client,candidate,row['seed'],i,env,source,j==0);audit=NativeAudit(env,k)
                episode=collect_episode(audit,policy,obs,prompt,max_steps=remaining,replan_steps=5,action_horizon=10,gamma=.9999)
                full,mask=behavior_sequences(episode);actual=np.concatenate([a[m] for a,m in zip(episode['actions'],episode['action_mask'],strict=True)])
                episode_id=sid+f'_cf{j}'
                episode.update(actor_bc_executed_sequence=full,actor_bc_mask=mask,critic_executed_actions=episode['actions'].copy(),critic_action_mask=episode['action_mask'].copy(),critic_k=episode['lengths'].copy(),raw_env_actions=actual,
                    env_step_rewards=np.asarray(audit.native,np.float32),normalized_actor_actions=2*(actual-low)/(high-low+1e-6)-1,
                    task_id=np.asarray(row['task_id']),task_name=np.asarray(row['task_name']),init_id=np.asarray(row['initial_state_id']),seed=np.asarray(row['seed']),split=np.asarray('train'),source=np.asarray('counterfactual_train'),episode_id=np.asarray(episode_id),state_id=np.asarray(sid),candidate_index=np.asarray(j),branch_seed=np.asarray(local_seed),split_hash=np.asarray(split['sha256']))
                validate_episode(episode);G=float(episode['mc_returns'][0]);gate_error=None
                if j==0:
                    gate_error=abs(G-float(source['mc_returns'][i]));assert gate_error<1e-5
                    assert policy.max_state_error<=1e-7 and policy.max_pixel_error==0 and policy.max_plan_error<=1e-6,('Original continuation mismatch',sid,policy.max_state_error,policy.max_pixel_error,policy.max_plan_error)
                    if i+1<len(source['snapshot_states']):assert float(abs(audit.next_state-source['snapshot_states'][i+1]).max())<=1e-7
                path=D/'episodes'/f'{episode_id}.npz';tmp=path.with_suffix('.partial')
                with tmp.open('wb') as z:np.savez_compressed(z,**episode)
                tmp.replace(path)
                cn=2*(candidate[:k]-low)/(high-low+1e-6)-1;pool=acts[nn,:k]
                rec=dict(episode_id=episode_id,state_id=sid,task_id=row['task_id'],task_name=row['task_name'],initial_state_id=row['initial_state_id'],source_episode_id=row['episode_id'],source_episode_sha256=row['sha256'],source_index=i,source_path=row['path'],candidate_index=j,candidate_name=name,
                    branch_seed=local_seed,noise_seed_rule='source_seed*1000 + source_chunk + continuation_chunk',path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),chunks=len(episode['rewards']),steps=int(episode['lengths'].sum()),success=bool(episode['success']),future_return=G,
                    initial_state_error=error,baseline_reproduction_error=gate_error,local_action_rms=float(np.sqrt(np.mean((pool-cn)**2,axis=(1,2))).min()),behavior_rms=float(np.sqrt(np.mean((cn-bn[:k])**2))),perturb_scale=scale.tolist(),k=k)
                state_records.append(rec)
            np.savez_compressed(D/'states'/f'{sid}.npz',flat=flat,metadata=np.asarray(hidden),base=source['base'][i],wrist=source['wrist'][i],proprio=source['state'][i],source_path=np.asarray(row['path']),source_index=np.asarray(i))
            payload=dict(state_id=sid,protocol_sha256=hashlib.sha256((D/'protocol.json').read_bytes()).hexdigest(),records=state_records,return_variance=float(np.var([v['future_return'] for v in state_records])),success_failure_mixture=bool(any(v['success'] for v in state_records) and any(not v['success'] for v in state_records)))
            target.write_text(json.dumps(payload));records.extend(state_records)
            print('STATE_DONE',sid,'G',[v['future_return'] for v in state_records],'elapsed',round(time.monotonic()-started,2),flush=True)
    except Exception as e:
        (D/'states'/f'{row["episode_id"]}.error.json').write_text(json.dumps(dict(source=row,error=repr(e),completed_records=records)));raise
    finally:env.close()
    return records

def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');p.add_argument('--workers',type=int,default=3);p.add_argument('--port',type=int,default=8027);args=p.parse_args()
    for name in ['episodes','states']: (D/name).mkdir(parents=True,exist_ok=True)
    manifest=json.loads((R/'data/pilot_libero_v1/manifest.json').read_text());records=manifest['records']
    if args.pilot: records=[next(row for row in records if row['task_id']==t) for t in range(10)]
    fractions=[.5] if args.pilot else [.1,.3,.5,.7,.9];collected=[];start=time.monotonic()
    if not (D/'protocol.json').exists():
        protocol=dict(version=2,seed=SEED,source_manifest_sha256=hashlib.sha256((R/'data/pilot_libero_v1/manifest.json').read_bytes()).hexdigest(),split_hash=manifest['split_hash'],states='200 train trajectories x five fixed fractions; 10 unique initial IDs/task, 2 independent policy repetitions (not20 unique initial IDs)',
            candidates='native recorded behavior plus coherent per-dimension Gaussian direction +/- .25 and .5 sigma; sigma=clamp(local10neighbor std, .1taskstd, taskstd); official quantile denormalization and native clipping',
            continuation='unchanged frozen pi05 checkpoint, same source explicit noise seeds, native binary reward, first-success termination / native max520 budget',
            inclusion='all preregistered branches retained, including identical outcomes and failures; invalid reproduction blocks completion',dev_states_used=False,actor_update=False,
            collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),target_states=1000,target_outcomes=5000)
        (D/'protocol.json').write_text(json.dumps(protocol,indent=2))
    protocol=json.loads((D/'protocol.json').read_text());assert protocol['collector_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as pool:
        jobs=[pool.submit(episode_group,row,fractions,args.port,str(R/'artifacts/replay_action_audit_v3/public_features.npz')) for row in records]
        for job in as_completed(jobs):
            collected.extend(job.result());status=dict(status='collecting',pilot=args.pilot,outcomes=len(collected),target_outcomes=len(records)*len(fractions)*5,elapsed_seconds=time.monotonic()-start)
            (D/('pilot_progress.json' if args.pilot else 'progress.json')).write_text(json.dumps(status));print('PROGRESS',json.dumps(status),flush=True)
    elapsed=time.monotonic()-start;inventory=dict(version='action_sensitive_replay_v2',split_hash=manifest['split_hash'],pilot=args.pilot,records=sorted(collected,key=lambda x:x['episode_id']),states=len(collected)//5,outcomes=len(collected),elapsed_seconds=elapsed,outcomes_per_minute=len(collected)/elapsed*60,
        purpose='critic train-only local intervention supervision; no actor updates',all_outcomes_retained=True)
    out=D/('pilot_manifest.json' if args.pilot else 'manifest.json');out.write_text(json.dumps(inventory,indent=2));print('COLLECTION_COMPLETE',json.dumps({k:v for k,v in inventory.items() if k!='records'}),flush=True)

if __name__=='__main__':main()
