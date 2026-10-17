# Public release: deployment paths anonymized; algorithm equations unchanged.
"""Eight independent frozen checkpoint evaluations; identical cases/statistics."""
from pathlib import Path
import os,subprocess,json,time
from run_critic_pilot_after_collection import common
R=Path('/opt/divl-trqam');O=R/'artifacts/replay_action_audit_v3'
def main():
    assert json.loads((O/'matched_v2_training_complete.json').read_text())['passed']
    jobs=[];handles=[]
    try:
        for method,gpus in [('scalar',[1,0,2,3]),('divl',[7,6,5,4])]:
            for step,gpu in zip([250,500,1000,2000],gpus,strict=True):
                env=common();env.update(CUDA_VISIBLE_DEVICES=str(gpu),JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_MEM_FRACTION='.35',V2_EVAL_METHOD=method,V2_EVAL_STEP=str(step))
                path=R/'logs/stage_v1'/f'v2_eval_{method}_{step:04d}.log';assert not path.exists();f=path.open('w');handles.append(f)
                p=subprocess.Popen([str(R/'.venv_actor/bin/python'),'-u',str(R/'scripts/evaluate_critic_replay_v2.py')],cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT)
                jobs.append(dict(method=method,step=step,gpu=gpu,pid=p.pid,process=p,log=str(path)))
        (O/'v2_evaluation_parallel_dispatch.json').write_text(json.dumps([{k:v for k,v in j.items() if k!='process'} for j in jobs],indent=2))
        while any(j['process'].poll() is None for j in jobs):
            for j in jobs:
                if j['process'].poll() is not None and j['process'].returncode:raise RuntimeError(f'Frozen eval failed: {j["method"]} step{j["step"]}; {j["log"]}')
            time.sleep(5)
        env=common();env.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',V2_EVAL_AGGREGATE='1')
        subprocess.run([str(R/'.venv_actor/bin/python'),'-u',str(R/'scripts/evaluate_critic_replay_v2.py')],cwd=R,env=env,check=True)
    finally:
        for j in jobs:
            if j['process'].poll() is None:j['process'].terminate()
        for f in handles:f.close()
if __name__=='__main__':main()
