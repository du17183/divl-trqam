# Public release: deployment paths anonymized; algorithm equations unchanged.
"""Own-process lifecycle and dependency isolation for a bounded inference test."""
import json
import os
from pathlib import Path
import socket
import subprocess
import time

ROOT = Path("/opt/divl-trqam")
ACTOR = ROOT / ".venv_actor/bin/python"
SIM = ROOT / ".venv_sim/bin/python"
PORT = 8017


def main():
    logdir = ROOT / "logs/feasibility_long"
    logdir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
           "PYTHONPATH": ":".join([str(ROOT), str(ROOT / "src"), str(ROOT / "packages/openpi-client/src")]),
           "OPENPI_DATA_HOME": str(ROOT / ".cache/openpi"),
           "HF_HOME": str(ROOT / ".cache/huggingface"),
           "XDG_CACHE_HOME": str(ROOT / ".cache"),
           "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
    actor_env = {**env, "CUDA_VISIBLE_DEVICES": "7", "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                 "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.2",
                 "LD_LIBRARY_PATH": ":".join(["/home/zwl/openpi/.cache/cube_bc_cudnn924/nvidia/cudnn/lib",
                     "/home/zwl/openpi/.venvs/openpi-b300/lib/python3.11/site-packages/nvidia/cu13/lib",
                     "/home/zwl/openpi/.venvs/openpi-b300/lib/python3.11/site-packages/nvidia/nccl/lib",
                     os.environ.get("LD_LIBRARY_PATH", "")])}
    sim_env = {**env, "CUDA_VISIBLE_DEVICES": "7", "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl",
               "MUJOCO_EGL_DEVICE_ID": "7", "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1", "LIBERO_CONFIG_PATH": str(ROOT / "configs/libero_runtime"),
               "PYTHONPATH": env["PYTHONPATH"] + ":" + str(ROOT / "third_party/libero")}
    config_dir = ROOT / "configs/libero_runtime"
    config_dir.mkdir(parents=True, exist_ok=True)
    benchmark = ROOT / "third_party/libero/libero/libero"
    paths = {"benchmark_root": str(benchmark), "bddl_files": str(benchmark / "bddl_files"),
             "init_states": str(benchmark / "init_files"), "assets": str(benchmark / "assets"),
             "datasets": "/home/researcher/lwd_libero_long/data/demos"}
    (config_dir / "config.yaml").write_text("\n".join(f"{k}: {v}" for k, v in paths.items()) + "\n")
    status_path = ROOT / "results/feasibility_run_status.json"
    def status(phase, **kw):
        status_path.write_text(json.dumps({"phase": phase, "parent_pid": os.getpid(), "time_unix": time.time(), **kw}, indent=2))
        print(phase, kw, flush=True)
    server = None
    try:
        with (logdir / "actor.log").open("w") as logfile:
            server = subprocess.Popen([str(ACTOR), "-u", str(ROOT / "scripts/serve_feasibility_policy.py"),
                                       "--root", str(ROOT), "--port", str(PORT)], env=actor_env, cwd=ROOT,
                                      stdout=logfile, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            status("loading_frozen_policy", actor_pid=server.pid)
            deadline = time.monotonic() + 600
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"Policy server exited {server.returncode}; inspect actor.log")
                try:
                    with socket.create_connection(("127.0.0.1", PORT), timeout=1):
                        break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError("Policy server startup timed out")
                    time.sleep(2)
            status("evaluating_ten_long_tasks", actor_pid=server.pid)
            with (logdir / "evaluation.log").open("w") as output:
                result = subprocess.run([str(SIM), "-u", str(ROOT / "scripts/eval_long_feasibility.py"),
                                         "--root", str(ROOT), "--port", str(PORT), "--episodes-per-task", "1"],
                                        env=sim_env, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, timeout=2400)
            if result.returncode:
                raise RuntimeError(f"Evaluation exited {result.returncode}; inspect evaluation.log")
            status("frozen_baseline_complete", actor_pid=server.pid)
    except Exception as error:
        status("failed", error=str(error))
        raise
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == "__main__":
    main()
