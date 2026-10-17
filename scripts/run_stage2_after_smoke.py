# Public release: deployment paths anonymized; algorithm equations unchanged.
import os,time,json,subprocess
from pathlib import Path
r=Path('/opt/divl-trqam')
gate=r/'artifacts/four_arm_smoke_test.json'
for _ in range(120):
    if gate.exists():
        assert json.loads(gate.read_text())['passed']
        break
    try:os.kill(2260058,0)
    except ProcessLookupError:raise RuntimeError('Smoke stopped without pass; split/audit not run')
    time.sleep(10)
else:raise TimeoutError('Waiting for four-arm smoke')
env=os.environ.copy()
env.update(PYTHONPATH=str(r)+':'+str(r/'src')+':'+str(r/'packages/openpi-client/src')+':'+str(r/'third_party/libero'),LIBERO_CONFIG_PATH=str(r/'configs/libero_runtime'),TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID='7',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
for name in ['audit_libero_split_expert.py','audit_snapshot_restore.py']:
    subprocess.run([str(r/'.venv_sim/bin/python'),'-u',str(r/'scripts'/name)],env=env,cwd=r,check=True)
print('STAGE2_COMPLETE',flush=True)
