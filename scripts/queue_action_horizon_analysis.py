"""Wait for this collection only, then run authorized bounded analysis."""
import time,json,subprocess
from horizon_common import R,D,O,atomic
from run_critic_pilot_after_collection import common
while not (D/'manifest.json').exists():
 status=json.loads((O/'collection_status.json').read_text())
 if status['stage']=='FAILED':raise RuntimeError('Collection failed; do not analyze incomplete data')
 time.sleep(10)
while json.loads((O/'collection_status.json').read_text())['stage']!='COLLECTION_COMPLETE':time.sleep(2)
env=common()
with (R/'logs/stage_v1/action_horizon_analysis_driver.log').open('w') as f:
 p=subprocess.Popen([str(R/'.venv_actor/bin/python'),'-u',str(R/'scripts/run_action_horizon_analysis.py')],cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
 atomic(O/'analysis_dispatch.json',dict(pid=p.pid,actor_updated=False,formal_training=False));raise SystemExit(p.wait())
