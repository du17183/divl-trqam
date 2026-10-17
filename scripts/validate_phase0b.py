# Public release: deployment paths anonymized; algorithm equations unchanged.
"""Check true full-horizon labels, terminal masks and actor coordinates."""
import json
import argparse
from pathlib import Path
import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np
from openpi import transforms
from openpi.models import pi0_config
from openpi.models.divl_config import DIVLStateEncoderConfig
from openpi.models.divl_tokenizer import DIVLTokenizer
from openpi.shared import normalize
from openpi.training import beta_training
from openpi.training.libero_replay_contract import behavior_sequences, validate_episode
from openpi.training.qam_libero_adapter import LiberoQAMBatchBuilder
from openpi.training.qam_training import QAMTrainingConfig

ROOT=Path('/opt/divl-trqam')


def synthetic(lengths):
    actions=np.zeros((len(lengths),10,7),np.float32)
    valid=np.arange(10)[None,:]<np.array(lengths)[:,None]
    sequence=np.arange(sum(lengths)*7,dtype=np.float32).reshape(-1,7)
    cursor=0
    for i,k in enumerate(lengths):
        actions[i,:k]=sequence[cursor:cursor+k];cursor+=k
    terminal=np.zeros(len(lengths),bool);terminal[-1]=True
    ep={'actions':actions,'action_mask':valid,'lengths':np.array(lengths),'terminal':terminal}
    full,mask=behavior_sequences(ep)
    offsets=np.concatenate(([0],np.cumsum(np.asarray(lengths,dtype=np.int64)[:-1])))
    for i,offset in enumerate(offsets):
        expected=sequence[offset:offset+10]
        np.testing.assert_array_equal(full[i,mask[i]],expected)
        assert mask[i].sum()==len(expected)
        assert not mask[i,len(expected):].any()
    return {'lengths':lengths,'valid_lengths':mask.sum(1).tolist(),'passed':True}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset-dir',default=str(ROOT/'data/feasibility_libero_long'));parser.add_argument('--pilot',action='store_true');args=parser.parse_args()
    gate=json.loads((ROOT/'artifacts/phase0_beta_fp32_metrics.json').read_text())
    assert gate['passed'], 'Phase0 real model gate must pass first'
    cases=[synthetic(x) for x in [[5,5,5],[5,5],[2,3,5],[4,5],[5],[2],[1]]]
    norm=normalize.load(ROOT/'checkpoints/pi05_libero/assets/physical-intelligence/libero')
    normalizer=transforms.Normalize(norm,use_quantiles=True)
    denormalizer=transforms.Unnormalize({'actions':norm['actions']},use_quantiles=True)
    cfg=pi0_config.Pi0Config(pi05=True,action_horizon=10,discrete_state_input=False)
    tok=DIVLTokenizer(DIVLStateEncoderConfig(gemma_tokenizer=str(ROOT/'assets/critic_tokenizer/tokenizer_gemma3.model')))
    builder=LiberoQAMBatchBuilder(cfg,norm,norm,tok,training_config=QAMTrainingConfig(critic_horizon=10,replan_steps=5),td_steps=1)
    episodes=[]
    for p in sorted(Path(args.dataset_dir).glob('*.npz')):
        with np.load(p,allow_pickle=False) as f: ep={k:f[k] for k in f.files}
        full,mask=validate_episode(ep)
        ep['actor_bc_executed_sequence']=full;ep['actor_bc_mask']=mask
        episodes.append((p,ep))
    candidates=[(p,e,i) for p,e in episodes for i in range(len(e['rewards']))]
    selected=np.random.default_rng(20261007).choice(len(candidates),100,replace=False)
    samples=[candidates[int(i)] for i in selected]
    maximum_target_error=maximum_roundtrip_error=0.0
    examples=[]
    for start in range(0,100,10):
        group=samples[start:start+10]
        batch=builder([(e,i) for p,e,i in group])
        targets=np.asarray(beta_training.bc_targets(batch.critic,32))
        target_mask=np.asarray(batch.critic['actor_bc_mask'])
        raw=np.stack([e['actor_bc_executed_sequence'][i] for p,e,i in group])
        expected=normalizer({'actions':raw})['actions'].astype(np.float32)
        valid=target_mask[...,None]
        target_error=float(np.max(np.abs(targets[:,:,:7]-expected)[np.broadcast_to(valid,expected.shape)]))
        roundtrip=denormalizer({'actions':targets[:,:,:7]})['actions']
        roundtrip_error=float(np.max(np.abs(roundtrip-raw)[np.broadcast_to(valid,raw.shape)]))
        maximum_target_error=max(maximum_target_error,target_error)
        maximum_roundtrip_error=max(maximum_roundtrip_error,roundtrip_error)
        np.testing.assert_array_equal(target_mask,np.stack([e['actor_bc_mask'][i] for p,e,i in group]))
        np.testing.assert_array_equal(np.asarray(batch.critic['critic_k']),[e['lengths'][i] for p,e,i in group])
        for j,(p,e,i) in enumerate(group):
            if len(examples)<8:
                examples.append({'episode':p.name,'step_id':int(np.asarray(e['lengths'][:i]).sum()),'chunk_id':i,
                                 'raw_action':raw[j].tolist(),'normalized_target':targets[j].tolist(),'mask':target_mask[j].tolist(),'critic_k':int(e['lengths'][i])})
    assert maximum_target_error<1e-5
    assert maximum_roundtrip_error<1e-5
    # Reconstruct masks without rewriting the original feasibility replay.
    result={'passed':True,'synthetic_cases':cases,'real_samples':100,'sample_seed':20261007,
            'max_normalized_target_error':maximum_target_error,'max_action_roundtrip_error':maximum_roundtrip_error,
            'critic_k_unchanged':True,'terminal_padding_loss_mask_zero':True,
            'padding_loss_exclusion_test':'artifacts/phase0_small_parameter_metrics.json',
            'original_replay_modified':False,'examples':examples}
    result['dataset_dir']=args.dataset_dir
    (ROOT/'artifacts'/('beta_target_pilot_audit.json' if args.pilot else 'beta_target_audit.json')).write_text(json.dumps(result,indent=2)+'\n')
    if args.pilot:
        with (ROOT/'reports/phase0b_beta_label_report.md').open('a') as report:
            report.write(f'\nFollow-up: 100 actual frozen train-pilot samples; target error {maximum_target_error:.3g}, roundtrip error {maximum_roundtrip_error:.3g}; masks and critic k preserved.\n')
        print(json.dumps({k:v for k,v in result.items() if k!='examples'},indent=2),flush=True)
        return
    (ROOT/'reports/phase0b_beta_label_report.md').write_text('# Phase 0B full beta label validation\n\nPASS.\n\n'
        f'100 actual LIBERO samples; normalized target error {maximum_target_error:.3g}, roundtrip error {maximum_roundtrip_error:.3g}.\n\n'
        'Seven synthetic cases cover complete H10, replan boundaries, three-chunk assembly, and 9/5/2/1 steps remaining.\n\n'
        'Critic prefixes/masks/k remain distinct from actor targets/masks. Builder normalizes beta labels using actor checkpoint stats.\n\n'
        'Terminal padding is masked out by beta_step; no raw replay files were changed.\n')
    (ROOT/'reports/current_stage_summary.md').write_text('# Current stage\n\nPhase 0 and Phase 0B PASS. Real FP32 updates/save-reload and 100-sample full label/mask audit completed.\n\n'
        'Next: LIBERO/Scalar training entry and disposable four-arm smoke test. Formal actor/QAM training is still disabled.\n')
    print(json.dumps({k:v for k,v in result.items() if k!='examples'},indent=2),flush=True)


if __name__=='__main__':main()
