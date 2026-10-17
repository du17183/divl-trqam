"""Frozen official policy server for a bounded LIBERO-Long feasibility run."""
import argparse
import faulthandler
import importlib.metadata
import json
import logging
from pathlib import Path
import time

from openpi.policies import policy_config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.training import config
import openpi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8017)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)
    faulthandler.dump_traceback_later(180, repeat=True)
    started = time.monotonic()
    policy = policy_config.create_trained_policy(
        config.get_config("pi05_libero"), args.root / "checkpoints/pi05_libero",
        sample_kwargs={"num_steps": 10},
    )
    record = {
        "config": "pi05_libero", "prediction_horizon": 10,
        "model_action_dim": 32, "environment_action_dim": 7,
        "openpi_module": str(Path(openpi.__file__).resolve()),
        "load_seconds": time.monotonic() - started,
        "versions": {k: importlib.metadata.version(k) for k in ["jax", "jaxlib", "flax", "numpy", "orbax-checkpoint"]},
        "frozen": True,
    }
    (args.root / "results/feasibility_actor_ready.json").write_text(json.dumps(record, indent=2))
    print("POLICY_READY " + json.dumps(record), flush=True)
    WebsocketPolicyServer(policy=policy, host="127.0.0.1", port=args.port).serve_forever()


if __name__ == "__main__":
    main()
