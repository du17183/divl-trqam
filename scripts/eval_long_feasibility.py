"""Exercise all ten LIBERO-Long tasks with the unchanged upstream adapters."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from libero.libero import benchmark
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from examples.libero.main import _get_libero_env, LIBERO_DUMMY_ACTION, LIBERO_ENV_RESOLUTION
from examples.libero.collect_divl import collect_episode, policy_observation


class AuditEnv:
    def __init__(self, env):
        self.env = env
        self.raw_rewards = []
        self.success_rewards = []
        self.step_seconds = []

    def step(self, action):
        start = time.monotonic()
        result = self.env.step(action)
        self.step_seconds.append(time.monotonic() - start)
        self.raw_rewards.append(float(result[1]))
        self.success_rewards.append(float(self.env.check_success()))
        return result

    def check_success(self):
        return self.env.check_success()


class AuditPolicy:
    def __init__(self, client, low, high):
        self.client, self.low, self.high = client, low, high
        self.plans, self.normalized, self.seconds = [], [], []

    def infer(self, obs):
        start = time.monotonic()
        output = self.client.infer(obs)
        self.seconds.append(time.monotonic() - start)
        plan = np.asarray(output["actions"], dtype=np.float32)
        if plan.shape != (10, 7) or not np.isfinite(plan).all():
            raise ValueError(f"Invalid official LIBERO action output: {plan.shape}")
        self.plans.append(plan.copy())
        self.normalized.append(2 * (plan - self.low) / (self.high - self.low + 1e-6) - 1)
        return output


def full_behavior_targets(episode, horizon=10):
    """True future EXECUTED actions; terminal padding only after episode end."""
    sequence = np.concatenate([a[m] for a, m in zip(episode["actions"], episode["action_mask"])])
    offsets = np.cumsum(np.concatenate([[0], episode["lengths"][:-1]]))
    result = []
    for offset in offsets:
        chunk = sequence[int(offset):int(offset) + horizon]
        if len(chunk) < horizon:
            chunk = np.concatenate([chunk, np.repeat(chunk[-1:], horizon - len(chunk), axis=0)])
        result.append(chunk)
    return np.stack(result).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8017)
    parser.add_argument("--episodes-per-task", type=int, default=1)
    args = parser.parse_args()
    run = args.root / "data/feasibility_libero_long"
    run.mkdir(parents=True, exist_ok=True)
    stats_path = args.root / "checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json"
    raw_stats = json.loads(stats_path.read_text())
    stats = raw_stats.get("norm_stats", raw_stats)
    low = np.asarray(stats["actions"]["q01"], dtype=np.float32)[:7]
    high = np.asarray(stats["actions"]["q99"], dtype=np.float32)[:7]
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    client = WebsocketClientPolicy("127.0.0.1", args.port)
    records = []
    begin = time.monotonic()
    for task_id in range(suite.n_tasks):
        task = suite.get_task(task_id)
        initial_states = suite.get_task_init_states(task_id)
        env, prompt = _get_libero_env(task, LIBERO_ENV_RESOLUTION, 20261006 + task_id)
        try:
            for repetition in range(args.episodes_per_task):
                init_id = (7 + 11 * repetition) % len(initial_states)
                env.reset()
                obs = env.set_init_state(initial_states[init_id])
                for _ in range(10):
                    obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
                current = policy_observation(obs, prompt)
                assert current["observation/state"].shape == (8,)
                assert current["observation/image"].shape == (224, 224, 3)
                assert current["observation/image"].dtype == np.uint8
                audit_env, policy = AuditEnv(env), AuditPolicy(client, low, high)
                episode = collect_episode(audit_env, policy, obs, prompt, max_steps=520,
                                          replan_steps=5, action_horizon=10, gamma=0.9999)
                episode["policy_action_plan"] = np.stack(policy.plans)
                episode["policy_action_normalized"] = np.stack(policy.normalized)
                episode["actor_bc_executed_sequence"] = full_behavior_targets(episode)
                episode["env_step_rewards"] = np.asarray(audit_env.raw_rewards, dtype=np.float32)
                episode.update(task_id=np.asarray(task_id), suite=np.asarray("libero_10"),
                               init_id=np.asarray(init_id), source=np.asarray("rollout"),
                               split=np.asarray("feasibility_only"),
                               seed=np.asarray(20261006 + task_id))
                path = run / f"task{task_id:02d}_episode{repetition:02d}.npz"
                with path.with_suffix(".partial").open("wb") as f:
                    np.savez_compressed(f, **episode)
                path.with_suffix(".partial").replace(path)
                recovered = (episode["policy_action_normalized"] + 1) / 2 * (high - low + 1e-6) + low
                reward_gap = np.max(np.abs(np.asarray(audit_env.raw_rewards) - audit_env.success_rewards))
                item = {
                    "task_id": task_id, "task_name": task.name, "instruction": prompt,
                    "initial_state_id": init_id, "success": bool(episode["success"]),
                    "return": float(episode["mc_returns"][0]), "length": int(episode["lengths"].sum()),
                    "chunks": len(episode["rewards"]), "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "policy_action_shape": list(episode["policy_action_plan"].shape[1:]),
                    "normalization_roundtrip_max_error": float(np.max(np.abs(recovered - episode["policy_action_plan"]))),
                    "raw_environment_vs_collector_reward_max_error": float(reward_gap),
                    "inference_seconds_mean": float(np.mean(policy.seconds)),
                    "inference_seconds_after_first_mean": float(np.mean(policy.seconds[1:])) if len(policy.seconds) > 1 else None,
                    "env_step_seconds_mean": float(np.mean(audit_env.step_seconds)),
                    "failure_reason": None if bool(episode["success"]) else "time_limit_or_env_terminal; stage not inferred",
                }
                records.append(item)
                summary = {"suite": "libero_10", "episodes": records, "completed": len(records),
                           "target": suite.n_tasks * args.episodes_per_task,
                           "successes": sum(r["success"] for r in records),
                           "elapsed_seconds": time.monotonic() - begin,
                           "scope": "frozen-policy feasibility; not a statistical LWD comparison"}
                (args.root / "results/libero_long_frozen_feasibility.json").write_text(json.dumps(summary, indent=2))
                print("EPISODE " + json.dumps(item), flush=True)
        finally:
            env.close()
    print("EVALUATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
