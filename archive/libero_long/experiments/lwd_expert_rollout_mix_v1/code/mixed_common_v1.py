# Public release: deployment paths anonymized; algorithm equations unchanged.
from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys,time
ROOT=Path('/data/divl-trqam/benchmark')
OLD=ROOT/'experiments/rl_pilot_sft5_step1000_8gpu'
EXP=ROOT/'experiments/lwd_expert_rollout_mix_v1'
CODE=EXP/'code'
ACTOR='/opt/divl-trqam/.venv_actor/bin/python'
SIM='/opt/divl-trqam/.venv_sim/bin/python'
sys.path[:0]=[str(CODE),str(OLD/'code'),str(ROOT/'scripts'),str(ROOT/'experiments/rl_pilot_sft5_step1000_2gpu/code')]
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while b:=f.read(8*1024*1024):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,j,exclusive=False):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);text=json.dumps(j,indent=2,ensure_ascii=False)+'\n'
 if exclusive:
  with p.open('x') as f:f.write(text)
 else:
  t=p.with_suffix(p.suffix+'.tmp');t.write_text(text);t.replace(p)
def identity(pid):
 p=Path('/proc')/str(pid);s=(p/'stat').read_text().rsplit(')',1)[1].split()
 return {'pid':pid,'birth_ticks':s[19],'ppid':int(s[1]),'state':s[0],'flags':int(s[6]),'argv':(p/'cmdline').read_bytes().decode().rstrip('\0').split('\0'),'cwd':os.readlink(p/'cwd')}
def exact(r,states=('R','S','D','I')):
 v=identity(r['pid']);assert all(v[k]==r[k] for k in ['birth_ticks','ppid','argv','cwd']),(r,v);assert v['state'] in states,v;return v
def protected_audit():
 from batch128_common_v1 import protected
 rows=[exact(r,('T',)) for r in protected().values()];assert len(rows)==13;return rows
def env(gpus=(),sim=False):
 from pilot8_common_v1 import environment
 e=environment(gpus,cpu=not bool(gpus));e.update(PYTHONPATH=f'{CODE}:{OLD}/code:{ROOT}/scripts:{ROOT}/experiments/rl_pilot_sft5_step1000_2gpu/code:{ROOT}/vendor/openpi/src:{ROOT}/third_party/libero',JAX_COMPILATION_CACHE_DIR=str(EXP/'cache/performance2/jax'/('_'.join(map(str,gpus)) or 'cpu')))
 if sim:e['MUJOCO_EGL_DEVICE_ID']='0'
 return e
def resources(owned=(),requested=tuple(range(8))):
 from pilot8_common_v1 import resource
 protected_audit();return resource(owned,requested)
def assert_owned(p,r):
 if p.poll() is not None:return
 # /proc main-thread teardown is allowed only for this registered Popen.
 d=Path('/proc')/str(p.pid);s=(d/'stat').read_text().rsplit(')',1)[1].split()
 assert s[19]==r['birth_ticks'] and int(s[1])==r['ppid']
 if s[0]=='Z' or int(s[6])&4:return
 exact(r)
def verify_sources():
 p=read(EXP/'provenance/sourceproof001.json')
 for f,h in p['files'].items():assert sha(f)==h,f
 return p
def progress(state):
 write(EXP/'metrics/pipeline_state001.json',state)
 lines=['# Expert＋Rollout 混合实验进度','',f"UTC：{stamp()}",f"阶段：{state['stage']}",'','主池：全部50专家＋240 Train Rollout，无来源配额。BC池：50专家＋109成功Rollout，来源50:50。','',f"专家审计任务完成：{sum(x.get('exit_code')==0 for x in state.get('finished_jobs',[]) if x.get('kind')=='expert_audit')}/10",f"新Critic真实更新：{state.get('critic_updates',{})}",f"新Actor真实更新：{state.get('actor_updates',{})}",f"新闭环评测：{state.get('completed_evaluation_episodes',0)}/24000",'','活跃作业：']
 for r in state.get('active_jobs',[]):lines.append(f"- {r['name']} PID {r['pid']} GPU {r['physical_gpus']}")
 lines+=['',f"阻塞或错误：{state.get('error','无')}",'','最新真实成功率：新实验尚无完成500回合结果。历史九组保持原值。','剩余：专家审计 → Mixed Critic 2×2000 → Mixed100 6×100＋3000回合及优先报告 → Cosine500 12×500＋18000回合 → BC500/EMA＋3000回合 → 全统计和Excel验收。']
 t=EXP/'reports/current_progress.md.tmp';t.write_text('\n'.join(lines)+'\n');t.replace(EXP/'reports/current_progress.md')
