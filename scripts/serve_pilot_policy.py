# Public release: deployment paths anonymized; algorithm equations unchanged.
"""Frozen official inference with reproducible explicit flow noise per request."""
import argparse
import json
import logging
from pathlib import Path
import numpy as np
from openpi.policies import policy_config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.training import config
from openpi_client import base_policy


class SeededPolicy(base_policy.BasePolicy):
    def __init__(self,policy):self.policy=policy
    @property
    def metadata(self):return self.policy.metadata
    def infer(self,obs):
        obs=dict(obs);seed=int(obs.pop('_noise_seed'))
        noise=np.random.default_rng(seed).normal(size=(10,32)).astype(np.float32)
        return self.policy.infer(obs,noise=noise)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8027);args=p.parse_args()
    logging.basicConfig(level=logging.INFO)
    root=Path('/opt/divl-trqam')
    policy=policy_config.create_trained_policy(config.get_config('pi05_libero'),root/'checkpoints/pi05_libero',sample_kwargs={'num_steps':10})
    print('POLICY_READY',flush=True)
    WebsocketPolicyServer(policy=SeededPolicy(policy),host='127.0.0.1',port=args.port).serve_forever()
