# Public release: deployment paths anonymized; algorithm equations unchanged.
"""Ensure selective decoding preserves original chunk sampling and values."""
from pathlib import Path
import json
import numpy as np
from openpi.training.divl_libero_adapter import EpisodeReplay
from train_libero_lwd import CRITIC_REPLAY_FIELDS

ROOT=Path('/opt/divl-trqam')

def main():
    directory=ROOT/'data/pilot_libero_v1/episodes';seed=20261007
    replay=EpisodeReplay(directory,seed=seed,fields=CRITIC_REPLAY_FIELDS);replay.refresh()
    original_indices=[];paths=[]
    for path in sorted(directory.glob('*.npz')):
        with np.load(path,allow_pickle=False) as f:
            if str(f['split'])!='train':continue
            eid=len(paths);paths.append(path)
            original_indices.extend((eid,i) for i in range(len(f['rewards'])))
    assert replay.indices==original_indices
    selected=np.random.default_rng(seed).integers(len(original_indices),size=256)
    samples=replay.sample(256)
    for j,(e,i) in enumerate(samples):
        eid,expected_i=original_indices[selected[j]];assert i==expected_i and e is replay.episodes[eid]
        if j<10:
            with np.load(paths[eid],allow_pickle=False) as f:
                for key in CRITIC_REPLAY_FIELDS:np.testing.assert_array_equal(e[key],f[key])
    result={'passed':True,'episodes':len(paths),'chunks':len(original_indices),'sample_order_checked':256,
            'full_critical_arrays_checked':10,'fields':list(CRITIC_REPLAY_FIELDS),'source_data_modified':False,
            'excluded':'simulator snapshots, observable-cache JSON and unused policy diagnostics; original files intact'}
    (ROOT/'artifacts/selective_replay_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
