import subprocess
from horizon_common import R,O,atomic
from run_critic_pilot_after_collection import common
env=common();env.update(CUDA_VISIBLE_DEVICES='0',JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_MEM_FRACTION='.35')
O.mkdir(parents=True,exist_ok=True)
with (R/'logs/stage_v1/action_horizon_audit.log').open('w') as log:
 p=subprocess.Popen([str(R/'.venv_actor/bin/python'),'-u',str(R/'scripts/audit_prepare_action_horizon.py')],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
 atomic(O/'audit_dispatch.json',dict(pid=p.pid,actor_updated=False,training=False,physical_gpu=0))
 ret=p.wait();atomic(O/'audit_exit.json',dict(exit_code=ret))
 raise SystemExit(ret)
