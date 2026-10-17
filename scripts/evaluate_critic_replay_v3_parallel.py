"""Eight checkpoint evaluations per fixed mix, original held-out50 unchanged."""
import argparse,json,os,subprocess,time
from action_sensitive_v3_common import R,O,atomic
from run_critic_pilot_after_collection import common


def main():
    p=argparse.ArgumentParser();p.add_argument('--mix',choices=['R1','R2'],required=True);args=p.parse_args()
    assert json.loads((O/f'matched_v3_{args.mix}_training_complete.json').read_text())['passed']
    jobs=[];handles=[]
    try:
        for method,gpus in [('scalar',[0,1,2,3]),('divl',[4,5,6,7])]:
            for step,gpu in zip([250,500,1000,2000],gpus,strict=True):
                env=common();env.update(CUDA_VISIBLE_DEVICES=str(gpu),JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_MEM_FRACTION='.35',V3_EVAL_METHOD=method,V3_EVAL_STEP=str(step),V3_EVAL_MIX=args.mix)
                log=R/'logs/stage_v1'/f'v3_eval_{args.mix}_{method}_{step:04d}.log';assert not log.exists();f=log.open('w');handles.append(f)
                process=subprocess.Popen([str(R/'.venv_actor/bin/python'),'-u',str(R/'scripts/evaluate_critic_replay_v3.py')],cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT)
                jobs.append(dict(method=method,step=step,gpu=gpu,pid=process.pid,process=process,log=str(log)))
        atomic(O/f'v3_evaluation_{args.mix}_dispatch.json',[{k:v for k,v in j.items() if k!='process'} for j in jobs])
        while any(j['process'].poll() is None for j in jobs):
            for j in jobs:
                if j['process'].poll() is not None and j['process'].returncode:raise RuntimeError('Evaluation failed '+j['log'])
            time.sleep(5)
        env=common();env.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',V3_EVAL_AGGREGATE='1',V3_EVAL_MIX=args.mix)
        subprocess.run([str(R/'.venv_actor/bin/python'),'-u',str(R/'scripts/evaluate_critic_replay_v3.py')],cwd=R,env=env,check=True)
    finally:
        for j in jobs:
            if j['process'].poll() is None:j['process'].terminate()
        for h in handles:h.close()


if __name__=='__main__':main()
