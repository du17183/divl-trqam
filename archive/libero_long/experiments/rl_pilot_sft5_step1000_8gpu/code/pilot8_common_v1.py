# Public release: deployment paths anonymized; algorithm equations unchanged.
"""Independent eight-GPU routing and exact ownership; native maths unchanged."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT=Path('/data/divl-trqam/benchmark')
OLD=ROOT/'experiments/rl_pilot_sft5_step1000_2gpu'
EXP=ROOT/'experiments/rl_pilot_sft5_step1000_8gpu'
CODE=EXP/'code'
ACTOR='/opt/divl-trqam/.venv_actor/bin/python'
SIM='/opt/divl-trqam/.venv_sim/bin/python'
sys.path[:0]=[str(ROOT/'scripts'),str(OLD/'code')]
AUTHKEY=__import__('os').environ.get('DIVL_TRQAM_IPC_KEY', 'local-development-only').encode()
UUIDS={int(k):v for k,v in json.loads((EXP/'configs/physical_gpus001.json').read_text())['uuids'].items()}
FLAGS='--xla_gpu_exclude_nondeterministic_ops=true --xla_gpu_autotune_level=0'

def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(8*1024*1024):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,d,exclusive=False):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);s=json.dumps(d,indent=2,ensure_ascii=False)+'\n'
    if exclusive:
        with p.open('x') as f:f.write(s)
    else:
        t=p.with_suffix(p.suffix+'.tmp');t.write_text(s);t.replace(p)
def identity(pid):
    p=Path('/proc')/str(pid);s=(p/'stat').read_text().rsplit(')',1)[1].split()
    return {'pid':pid,'birth_ticks':s[19],'state':s[0],'ppid':int(s[1]),'argv':(p/'cmdline').read_bytes().decode().rstrip('\0').split('\0'),'cwd':os.readlink(p/'cwd')}
def exact(rec,states=('R','S','D','I')):
    now=identity(rec['pid']);assert all(now[k]==rec[k] for k in ('birth_ticks','ppid','argv','cwd')),(rec,now)
    assert now['state'] in states,now
    return now
def config():return read(EXP/'configs/protocol001.json')
def frozen():
    for rel,h in read(EXP/'provenance/sourceproof001.json')['files'].items():assert sha(ROOT/rel)==h,rel
def frozen_v3():frozen()
def environment(gpus=(),cpu=False):
    from run_profile_v4 import env_for
    gpus=tuple(gpus);assert all(g in UUIDS for g in gpus) and len(set(gpus))==len(gpus)
    e=os.environ.copy();validated=env_for(list(gpus) if gpus else [0])
    for key in ('LD_LIBRARY_PATH','XLA_FLAGS'):
        if key in validated:e[key]=validated[key]
    e.update(CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)),PILOT_PHYSICAL_GPUS=','.join(map(str,gpus)),
        JAX_CUDA_VISIBLE_DEVICES=','.join(map(str,range(len(gpus)))),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONPATH=f'{CODE}:{ROOT}/scripts:{OLD}/code:{ROOT}/vendor/openpi/src:{ROOT}/third_party/libero',
        LIBERO_CONFIG_PATH=str(ROOT/'configs/libero_runtime'),MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',
        XLA_PYTHON_CLIENT_PREALLOCATE='false',JAX_COMPILATION_CACHE_DIR=str(EXP/'cache/jax'/('_'.join(map(str,gpus)) or 'cpu')),
        TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',JAX_PLATFORMS='cpu' if cpu else 'cuda')
    if not cpu:e['XLA_FLAGS']=(e.get('XLA_FLAGS','')+' '+FLAGS).strip()
    return e
def physical_gpus():
    value=tuple(int(x) for x in os.environ['PILOT_PHYSICAL_GPUS'].split(',') if x)
    assert value and os.environ['CUDA_VISIBLE_DEVICES']==','.join(map(str,value))
    assert all(i in UUIDS for i in value)
    return value
def runtime_receipt():
    physical=physical_gpus();assert all(x in os.environ.get('XLA_FLAGS','').split() for x in FLAGS.split())
    return {'physical_gpus':physical,'logical_to_physical':{str(i):g for i,g in enumerate(physical)},
        'logical_to_uuid':{str(i):UUIDS[g] for i,g in enumerate(physical)},'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],
        'XLA_FLAGS':os.environ['XLA_FLAGS'],'master_dtype':'float32','compute_dtype':'float32','absolute_tolerance':2e-5,'threshold_changed':False}
def gpu_receipt(expected=None):
    import jax
    physical=physical_gpus();ds=jax.devices();assert len(ds)==len(physical) and all(d.platform=='gpu' for d in ds)
    out=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True)
    actual={line.split(',')[1].strip() for line in out.splitlines() if int(line.split(',')[0])==os.getpid()}
    assert actual.issubset({UUIDS[g] for g in physical}),actual
    return {**runtime_receipt(),'jax_devices':[str(d) for d in ds],'actual_allocated_uuid':sorted(actual),'pid':os.getpid()}
def resource(owned=(),requested=tuple(range(8))):
    from batch128_common_v1 import protected
    old=protected();known={r['pid']:r for r in owned};rows=[];processes=[]
    out=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    for line in out.splitlines():
        i,u,f,m,z=[x.strip() for x in line.split(',')];i=int(i)
        if i in requested:
            assert u==UUIDS[i];rows.append({'gpu':i,'uuid':u,'free_mib':float(f),'used_mib':float(m),'util_percent':float(z)})
    selected={UUIDS[g] for g in requested};out=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True)
    for line in out.splitlines():
        pid,u,m=[x.strip() for x in line.split(',')];pid=int(pid)
        if u not in selected:continue
        if pid in old:v=exact(old[pid],('T',))
        elif pid in known:v=exact(known[pid])
        else:raise RuntimeError(('Requested GPUs contain unowned process; wait, never signal',identity(pid),u))
        processes.append({**v,'uuid':u,'memory_mib':float(m)})
    return {'utc':stamp(),'gpus':rows,'processes':processes}
def verify_worker():
    frozen()
    proof=read(EXP/'provenance/execution_sources001.json')
    for rel,h in proof['files'].items():assert sha(ROOT/rel)==h,rel
    assert not (EXP/'human_hold.json').exists(),'HUMAN_HOLD'
    m=read(EXP/'data/replay/manifest.json');assert m['audit_passed'] and m['episodes']==300
    assert m['checkpoint_manifest_sha256']==config()['sft_checkpoint_manifest_sha256']
    assert sha(config()['normalization'])==config()['normalization_sha256']
    return m

class Jobs:
    def __init__(self,state,path):self.state=state;self.path=path;self.jobs={}
    def save(self):write(self.path,self.state)
    def start(self,python,script,args,gpus,name):
        assert not (EXP/'human_hold.json').exists(),'HUMAN_HOLD'
        gpus=tuple(gpus);assert not any(set(gpus)&set(r['physical_gpus']) for r in self.state['active_jobs'])
        receipt=resource(self.state['active_jobs'],gpus)
        argv=[python,'-u',str(CODE/script),*map(str,args)];logpath=EXP/'logs'/f'{name}.log';log=logpath.open('x')
        env=environment(gpus);env['MUJOCO_EGL_DEVICE_ID']=str(gpus[0])
        p=subprocess.Popen(argv,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close()
        deadline=time.monotonic()+10
        while True:
            rec=identity(p.pid)
            if rec['argv']==argv:break
            assert p.poll() is None and time.monotonic()<deadline;time.sleep(.02)
        rec.update(physical_gpus=list(gpus),log=str(logpath),started_utc=stamp(),initial_resource=receipt)
        self.jobs[p.pid]=(p,time.monotonic(),rec);self.state['active_jobs'].append(rec);self.save();return p.pid
    def finish(self,pid,wait=False):
        p,start,rec=self.jobs[pid];code=p.wait() if wait else p.poll()
        if code is None:return None
        out={**rec,'exit_code':code,'ended_utc':stamp(),'wall_s':time.monotonic()-start}
        out['reserved_gpu_hours']=len(rec['physical_gpus'])*out['wall_s']/3600
        self.state['active_jobs']=[x for x in self.state['active_jobs'] if x['pid']!=pid];self.state['finished_jobs'].append(out);del self.jobs[pid];self.save();return out
    def stop_server(self,pid,socket):
        from multiprocessing.connection import Client
        exact(self.jobs[pid][2])
        with Client(socket,family='AF_UNIX',authkey=AUTHKEY) as c:c.send({'command':'stop'});assert c.recv()['stopping']
        return self.finish(pid,True)
