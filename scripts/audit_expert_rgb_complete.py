# Public release: deployment paths anonymized; algorithm equations unchanged.
from pathlib import Path
import h5py,json,hashlib,numpy as np
r=Path('/opt/divl-trqam')
manifest=json.loads((r/'artifacts/expert_hdf5_manifest.json').read_text())
files=sorted({e['path'] for e in manifest['episodes_manifest']});records=[];frames=0
for file in files:
    p=Path(file);digest=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):digest.update(block)
    with h5py.File(p,'r') as h:
        for name in h['data']:
            episode=h['data'][name];n=len(episode['actions'])
            for key,value in episode['obs'].items():
                if 'rgb' not in key and 'image' not in key:continue
                for start in range(0,n,64):
                    image=np.asarray(value[start:start+64])
                    assert image.dtype==np.uint8 and image.shape[-1]==3 and len(image)==min(64,n-start)
                    frames+=len(image)
    records.append({'path':file,'sha256':digest.hexdigest(),'bytes':p.stat().st_size})
    print('RGB_FILE_COMPLETE',p.name,frames,flush=True)
result={'passed':True,'files':records,'trajectories':manifest['episodes'],'rgb_frames_decoded':frames,'scope':'Every RGB frame decoded in bounded blocks, no expert inclusion'}
(r/'artifacts/expert_rgb_full_audit.json').write_text(json.dumps(result,indent=2))
with (r/'reports/expert_hdf5_audit.md').open('a') as f:f.write(f'\nFull RGB follow-up: all {frames} frames across {len(files)} HDF5 files decoded successfully; file SHA256 recorded in artifacts/expert_rgb_full_audit.json. Expert initial provenance remains unresolved and all experts remain excluded.\n')
print('EXPERT_RGB_AUDIT_COMPLETE',json.dumps({'frames':frames,'files':len(files)}))
