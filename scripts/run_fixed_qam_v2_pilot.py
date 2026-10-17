# Public release: deployment paths anonymized; algorithm equations unchanged.
"""Release training GPU memory before launching the updated policy and sim pilot."""
from pathlib import Path
import argparse,json,os,signal,socket,subprocess,time
from run_replay_v2_pipeline import common,sim_env,gpu_env
R=Path('/opt/divl-trqam');O=R/'artifacts/replay_action_audit_v3'
def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['scalar','divl'],required=True);args=p.parse_args()
    assert json.loads((O/'v2_new_gradient_metrics.json').read_text())['methods'][args.method]['passed']
    env=gpu_env(7,'.55')
    subprocess.run([str(R/'.venv_actor/bin/python'),'-u',str(R/'scripts/train_fixed_qam_v2_actor.py'),'--method',args.method],cwd=R,env=env,check=True)
    log=(R/'logs/stage_v1'/f'fixed_qam_{args.method}_server.log').open('w');script=str(R/'scripts/serve_fixed_qam_v2_actor.py')
    server=subprocess.Popen([str(R/'.venv_actor/bin/python'),'-u',script,'--method',args.method,'--port','8030'],cwd=R,env=gpu_env(7,'.25'),stdout=log,stderr=subprocess.STDOUT)
    try:
        deadline=time.monotonic()+900
        while True:
            if server.poll() is not None:raise RuntimeError('Updated FP32 policy server failed')
            try:
                with socket.create_connection(('127.0.0.1',8030),timeout=1):break
            except OSError:
                if time.monotonic()>deadline:raise TimeoutError('Updated actor startup timeout')
                time.sleep(2)
        subprocess.run([str(R/'.venv_sim/bin/python'),'-u',str(R/'scripts/evaluate_fixed_qam_v2_actor.py'),'--method',args.method],cwd=R,env=sim_env(),check=True)
    finally:
        if server.poll() is None:
            proc=Path(f'/proc/{server.pid}');cmd=proc.joinpath('cmdline').read_bytes().split(b'\0')
            assert script.encode() in cmd and proc.joinpath('cwd').resolve()==R.resolve();os.kill(server.pid,signal.SIGTERM);server.wait(timeout=20)
        log.close()

if __name__=='__main__':main()
