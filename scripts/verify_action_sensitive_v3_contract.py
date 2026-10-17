"""Real first-branch contract, future-label-independent selection, fixed mixes."""
import copy,json,numpy as np
from action_sensitive_v3_common import R,D,O,action_bounds,candidates,select_phase,atomic,sha
from evaluate_action_sensitive_replay_v2 import first_rows


def main():
    protocol=json.loads((D/'protocol.json').read_text());pre=json.loads((O/'branches_preflight.json').read_text())
    assert pre['passed'] and len(pre['records'])==1
    group=pre['records'][0];assert group['valid'] and len(group['records'])==5
    row=group['source'];sid=row['state_id']
    with np.load(row['path'],allow_pickle=False) as source:
        i=row['source_index'];low,high=action_bounds()
        sigma=json.loads((D/'action_distribution.json').read_text())[str(row['task_id'])]['normalized']['std']
        raw,norm,padded,specs=candidates(source['actions'][i],int(source['lengths'][i]),sigma,row['branch_seed'],low,high)
        phase=json.loads(str(source['phase_features_json']))
        selected,_=select_phase(phase,source['lengths'],row['phase_index'],row['selection_seed'])
        injected=copy.deepcopy(phase)
        for j,features in enumerate(injected):features.update(future_return=(-10000 if j==selected else 10000),success=(j!=selected))
        repeated,_=select_phase(injected,source['lengths'],row['phase_index'],row['selection_seed'])
        assert selected==repeated==i,'Phase selection must ignore future labels'
    assert raw.shape==(5,10,7) and norm.shape==(5,10,7) and padded.shape==(5,10,32)
    assert not padded[:,:,7:].any() and not norm[:,5:].any()
    for j,record in enumerate(group['records']):
        first=first_rows(record['path'],['actions','action_mask'])
        assert np.array_equal(first['actions'][first['action_mask']],raw[j][first['action_mask']])
        assert record['initial_state_error']==0
    assert group['records'][0]['baseline_reproduction_error']<=1e-5
    assert [float(x[1]) for x in specs]==[0,.25,.25,.5,1.]
    cases=json.loads((D/'source_cases.json').read_text());assert len(cases)==300 and len({r['seed'] for r in cases})==300
    split=json.loads((R/'configs/splits/libero_factorial_v1.json').read_text())
    for case in cases:assert case['initial_state_id'] in split['tasks'][case['task_name']]['train']
    assert len({(c['task_id'],c['initial_state_id']) for c in cases})==300
    for path,h in protocol['frozen_input_hashes'].items():assert sha(path)==h
    result=dict(passed=True,real_branches=5,baseline_error=group['records'][0]['baseline_reproduction_error'],
                padding_25_dimensions_zero=True,phase_selection_ignores_future_labels=True,
                train_initial_state_groups=300,dev_final_groups_used=0,all_five_planned_candidates_retained=True)
    atomic(O/'v3_preflight_contract.json',result);print('V3_CONTRACT',json.dumps(result),flush=True)


if __name__=='__main__':main()
