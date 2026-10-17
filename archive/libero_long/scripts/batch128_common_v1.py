# Public release: deployment paths anonymized; algorithm equations unchanged.
"""Isolated native FP32 Batch128 SFT; old experiment sources remain immutable."""
from pathlib import Path
import dataclasses, datetime, hashlib, json, os, subprocess, time

ROOT = Path(__file__).resolve().parents[1]
REV = 'batch128_revision001'
CONFIG = ROOT / 'configs/batch128_sft_v1.json'
ACTOR = '/opt/divl-trqam/.venv_actor/bin/python'
SIM = '/opt/divl-trqam/.venv_sim/bin/python'

def stamp(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def write(p,j,exclusive=False):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    if exclusive:
        with p.open('x') as f: json.dump(j,f,ensure_ascii=False,indent=2)
    else:
        tmp=Path(str(p)+'.tmp'); tmp.write_text(json.dumps(j,ensure_ascii=False,indent=2));tmp.replace(p)
def identity(pid):
    d=Path('/proc')/str(pid);s=(d/'stat').read_text().rsplit(')',1)[1].split()
    return {'pid':pid,'birth_ticks':s[19],'ppid':int(s[1]),'state':s[0],
            'argv':(d/'cmdline').read_bytes().decode().rstrip('\0').split('\0'),'cwd':os.readlink(d/'cwd')}
def exact(r,allowed_states=('R','S','D','I')):
    v=identity(r['pid']);assert all(v[k]==r[k] for k in ('birth_ticks','ppid','argv','cwd')),(r,v)
    assert v['state'] in allowed_states,v
    return v
def env(gpus):
    from run_profile_v4 import env_for
    e=env_for(gpus)
    e['JAX_COMPILATION_CACHE_DIR']=str(ROOT/'cache/batch128/jax')
    e['PYTHONUNBUFFERED']='1'
    return e
def protected():
    p=read(ROOT/'provenance/sft5_revision001/human_pause_20261010T064424Z.json')
    return {int(k):{'pid':v['pid'],'birth_ticks':str(v['birth']),'ppid':v['ppid'],'argv':v['argv'],'cwd':v['cwd']} for k,v in p['protected_after'].items()}
def resource(gpus,owners=(),minimum_free=65000):
    devices={}
    for row in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines():
        i,u,f,m,z=[x.strip() for x in row.split(',')];devices[u]={'gpu':int(i),'free_mib':float(f),'used_mib':float(m),'util':float(z)}
    own={r['pid']:r for r in owners};old=protected();events=[]
    for row in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True).splitlines():
        if not row.strip(): continue
        u,p,m=[x.strip() for x in row.split(',')]
        if devices[u]['gpu'] not in gpus: continue
        pid=int(p)
        try:v=identity(pid)
        except FileNotFoundError:continue
        if v['state']=='Z':continue
        if pid in old:exact(old[pid],('T',))
        elif pid in own:exact(own[pid])
        else:return False,{'reason':'foreign or unregistered process; never signal it','process':v}
        events.append({**v,'gpu':devices[u]['gpu'],'memory_mib':m})
    selected=[v for v in devices.values() if v['gpu'] in gpus]
    return all(v['free_mib']>=minimum_free for v in selected),{'utc':stamp(),'devices':selected,'processes':events,'minimum_free_mib':minimum_free}
def verify_sources(proof,expected):
    assert sha(proof)==expected
    j=read(proof)
    for rel,d in j['inputs_sha256'].items(): assert sha(ROOT/rel)==d,rel
    return j
def verify_checkpoint(path):
    path=Path(path);m=read(path/'manifest.json');assert m['complete']
    for rel,e in m['files'].items(): assert sha(path/rel)==e['sha256'] and (path/rel).stat().st_size==e['bytes'],rel
    return m
def names(scale):
    return {'run_id':f'sft{scale}_b128_seed0','manifest':ROOT/f'data/sft/batch128/sft{scale:02d}_manifest.json',
            'norm':ROOT/f'data/sft/batch128/sft{scale:02d}_norm.json'}

@dataclasses.dataclass(frozen=True)
class WarmupConstant:
    def create(self):
        import jax.numpy as jnp
        return lambda count: jnp.minimum((count+1)/500.,1.)*2.5e-5
def make_config(devices):
    from benchmark_fm_v3 import make_config as old
    from openpi.training.optimizer import AdamW
    cfg=old(128,devices,0)
    return dataclasses.replace(cfg,name='pi05_libero_long_batch128_v1',lr_schedule=WarmupConstant(),
        optimizer=AdamW(b1=.9,b2=.95,eps=1e-8,weight_decay=1e-10,clip_gradient_norm=1.),
        ema_decay=None,fsdp_devices=devices,num_train_steps=6000)

def loader(scale,prefetch=2,workers=4,start_index=0):
    from sft_subset_loader_v1 import NativeExpertLoader
    import numpy as np,time
    from benchmark_data_v1 import load_sample
    class BalancedLoader(NativeExpertLoader):
        def _batch(self,batch_size):
            assert batch_size==128
            index=self.producer_index;self.producer_index+=1
            rng=np.random.default_rng(np.random.SeedSequence([self.seed,index]));specs=[]
            for slot in range(128):
                task=(index*128+slot)%10
                e=self.by_task[task][int(rng.integers(len(self.by_task[task])))];anchor=int(rng.integers(e['length']-9))
                specs.append((e,anchor,self.norm))
            before=time.monotonic();xs=list(self.pool.map(load_sample,specs)) if self.pool else list(map(load_sample,specs))
            data={'images':{k:np.stack([x['images'][k] for x in xs]) for k in xs[0]['images']},
                  'image_masks':{k:np.full(128,k!='right_wrist_0_rgb',bool) for k in xs[0]['images']},
                  'state':np.stack([x['state'] for x in xs]),'actions':np.stack([x['actions'] for x in xs]),
                  'action_mask':np.stack([x['action_mask'] for x in xs]),
                  'tokenized_prompt':np.stack([self.tokens[x['task_id']][0] for x in xs]).astype(np.int32),
                  'tokenized_prompt_mask':np.stack([self.tokens[x['task_id']][1] for x in xs]).astype(bool)}
            if self.pinned:
                import torch
                def pin(x):return {k:pin(v) for k,v in x.items()} if isinstance(x,dict) else torch.from_numpy(x).pin_memory().numpy()
                data=pin(data)
            meta=[{'task_id':x['task_id'],'state_source_id':x['state_source_id'],'anchor':x['anchor'],'global_batch_index':index} for x in xs]
            return data,meta,time.monotonic()-before
    n=names(scale);return BalancedLoader(n['manifest'],n['norm'],0,workers=workers,prefetch=prefetch,pinned=True,start_index=start_index)
