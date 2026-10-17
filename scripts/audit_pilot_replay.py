# Public release: deployment paths anonymized; algorithm equations unchanged.
"""Verify frozen raw trajectories, alias fields and disjoint official initial IDs."""
from pathlib import Path
import json
import hashlib
import numpy as np
from openpi.training.libero_replay_contract import validate_episode

ROOT=Path('/opt/divl-trqam')

def main():
    split=json.loads((ROOT/'configs/splits/libero_factorial_v1.json').read_text())
    rows=[];seeds=set();maximum=0.;alias_error=0.;kset=set();terminal_count=0;timeouts=0
    for name,expected,key in [('pilot_libero_v1',200,'train'),('dev_libero_v1',100,'dev')]:
        manifest=json.loads((ROOT/'data'/name/'manifest.json').read_text());assert manifest['episodes']==expected and manifest['split_hash']==split['sha256']
        for row in manifest['records']:
            path=Path(row['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
            with np.load(path,allow_pickle=False) as f:e={k:f[k] for k in f.files}
            validate_episode(e)
            assert int(e['init_id']) in split['tasks'][str(e['task_name'])][key]
            assert str(e['split'])==key and int(e['seed']) not in seeds;seeds.add(int(e['seed']))
            assert e['state'].shape[1]==8 and e['actions'].shape[1:]==(10,7)
            assert np.array_equal(e['critic_k'],e['lengths']) and np.array_equal(e['critic_action_mask'],e['action_mask'])
            alias_error=max(alias_error,float(np.abs(e['critic_executed_actions']-e['actions']).max()))
            actual=np.concatenate([a[m] for a,m in zip(e['actions'],e['action_mask'],strict=True)])
            assert np.array_equal(actual,e['raw_env_actions'])
            offsets=np.concatenate([[0],np.cumsum(e['lengths'])]);assert offsets[-1]==len(e['env_step_rewards'])
            for i,k in enumerate(e['lengths']):
                native=sum(.9999**j*float(e['env_step_rewards'][offsets[i]+j]) for j in range(int(k)))
                maximum=max(maximum,abs(native-float(e['rewards'][i])));kset.add(int(k))
            assert np.all(e['discounts'][e['terminal']]==0)
            terminal_count+=int(e['terminal'].sum());timeouts+=int(not bool(e['success']) and int(e['lengths'].sum())==520)
            assert len(e['snapshot_states'])==len(e['rewards']) and len(e['snapshot_metadata'])==len(e['rewards'])
            rows.append({'episode_id':row['episode_id'],'split':key,'chunks':len(e['rewards']),'environment_steps':int(e['lengths'].sum()),'success':bool(e['success'])})
    assert maximum<1e-6 and alias_error==0 and len(seeds)==300
    result={'passed':True,'train_episodes':200,'dev_episodes':100,'unique_seeds':len(seeds),'env_steps':sum(r['environment_steps'] for r in rows),
        'critic_k_values':sorted(kset),'reward_native_max_error':maximum,'critic_action_alias_error':alias_error,'terminal_chunks':terminal_count,
        'benchmark_timeouts':timeouts,'gamma':.9999,'horizon':10,'replan_steps':5,'env_action_dim':7,'model_action_dim':32,'proprio_dim':8,
        'expert_included':0,'split_sha256':split['sha256'],'terminal_protocol':'Native success/termination and benchmark horizon end episode; horizon is treated as terminal failure by the unchanged original collector. No bootstrap after recorded episode terminal.',
        'records':rows}
    (ROOT/'artifacts/pilot_replay_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2),flush=True)

if __name__=='__main__':main()
