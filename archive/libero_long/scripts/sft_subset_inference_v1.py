"""Fresh native FP32 policy per checkpoint; explicit paired noise and unchanged H10 flow."""
from pathlib import Path
import argparse
import json
import os
import time
from multiprocessing.connection import Listener

ROOT = Path(__file__).resolve().parents[1]
AUTHKEY = __import__('os').environ.get('DIVL_TRQAM_IPC_KEY', 'local-development-only').encode()


class NativeActor:
    def __init__(self, checkpoint, norm_path):
        import jax
        import jax.numpy as jnp
        from openpi.models import model as model_api
        from openpi.models.pi0_config import Pi0Config
        from openpi.models.tokenizer import PaligemmaTokenizer
        from openpi.shared.nnx_utils import module_jit
        jax.config.update('jax_default_matmul_precision', 'highest')
        jax.config.update('jax_compilation_cache_dir', str(ROOT / 'cache/jax'))
        self.params = model_api.restore_params(checkpoint, dtype=jnp.float32)
        assert all(str(x.dtype) == 'float32' for x in jax.tree.leaves(self.params))
        self.model = Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False, dtype='float32').load(self.params)
        self.model.eval()
        self.sample = module_jit(self.model.sample_actions)
        self.tokenizer = PaligemmaTokenizer(200)
        self.norm = json.loads(Path(norm_path).read_text())

    def infer(self, request):
        import jax
        import jax.numpy as jnp
        import numpy as np
        from openpi.models import model as model_api
        from benchmark_io_v1 import observation_data, unnormalize_actions
        data = observation_data(request['rgb'], request['proprio'], request['language'], self.tokenizer, self.norm)
        observation = model_api.Observation(**jax.tree.map(lambda x:jnp.asarray(x)[None], data))
        noise = np.asarray(request['noise'],np.float32); assert noise.shape == (10,32) and np.isfinite(noise).all()
        before = time.monotonic()
        normalized = np.asarray(jax.block_until_ready(self.sample(jax.random.key(0), observation, noise=jnp.asarray(noise)[None], num_steps=10)))[0]
        actions = unnormalize_actions(normalized, self.norm)
        return {'normalized_h10':normalized, 'raw_h10':actions, 'native_sample_wall_s':time.monotonic()-before,
                'precision':'FP32 highest', 'real_action_dim':7, 'model_action_dim':32, 'prediction_horizon':10, 'flow_steps':10}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);parser.add_argument('--socket',required=True)
    parser.add_argument('--ready',required=True);parser.add_argument('--norm',required=True);args=parser.parse_args()
    assert not Path(args.socket).exists()
    actor=NativeActor(args.checkpoint, args.norm)
    with Listener(args.socket,family='AF_UNIX',authkey=AUTHKEY) as listener:
        Path(args.ready).write_text(json.dumps({'pid':os.getpid(),'checkpoint':args.checkpoint,'socket':args.socket,'ready':True,'normalization_path':args.norm,'normalization_sha256':__import__('hashlib').sha256(Path(args.norm).read_bytes()).hexdigest(),'jit_binding':'new module_jit after restore and FP32 precision'})+'\n')
        print('NATIVE_POLICY_SERVER_READY',args.checkpoint,flush=True)
        while True:
            with listener.accept() as connection:
                while True:
                    try: request=connection.recv()
                    except EOFError: break
                    if request.get('command')=='stop': connection.send({'stopping':True});return
                    try: connection.send({'ok':True, **actor.infer(request)})
                    except Exception as error:
                        connection.send({'ok':False,'error':str(error)});raise


if __name__=='__main__':main()
