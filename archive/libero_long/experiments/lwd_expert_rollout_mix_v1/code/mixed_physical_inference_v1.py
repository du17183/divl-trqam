"""Read-only frozen SFT/twoCritic service for bounded physical diagnostics."""
from pilot8_common_v1 import *
from mixed_common_v2 import EXP,CODE,OLD
import argparse
from multiprocessing.connection import Listener


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--socket',required=True);p.add_argument('--ready',required=True);a=p.parse_args()
    from critic_restore_current_layout_v1 import verify_repair,restore_tree
    verify_repair()
    from mixed_common_v2 import verify_sources,protected_audit
    verify_sources();protected_audit()
    for rel,h in read(EXP/'provenance/mixed_diagnostics_sources015.json')['files'].items():assert sha(EXP/rel)==h,rel
    for rel,h in read(OLD/'provenance/critic_4plus4_sources003.json')['files'].items():assert sha(ROOT/rel)==h,rel
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import numpy as np
    import orbax.checkpoint as ocp
    from openpi.models import model as model_api
    from openpi.models.pi0_config import Pi0Config
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.shared.nnx_utils import module_jit
    from sft_subset_inference_v1 import NativeActor
    from pilot8_critic_v1 import Replay,cfg
    from critic_model_v1 import build_pretrained,action_gradient,encoder_config
    from critic_replay_v1 import collate
    from openpi.models.divl_tokenizer import DIVLTokenizer
    assert physical_gpus()==(a.gpu,) and len(jax.devices())==1
    target=jax.devices()[0];jax.config.update('jax_default_matmul_precision','highest')
    c=config();actor=NativeActor.__new__(NativeActor)
    with jax.default_device(target):
        actor.params=model_api.restore_params(c['sft_checkpoint']+'/params',dtype=jnp.float32,sharding=jax.sharding.SingleDeviceSharding(target))
        actor.model=Pi0Config(pi05=True,action_horizon=10,discrete_state_input=False,dtype='float32').load(actor.params);actor.model.eval()
        actor.sample=module_jit(actor.model.sample_actions);actor.tokenizer=PaligemmaTokenizer(200);actor.norm=read(c['normalization'])
        critics={};checkpoint_shas={}
        for m in ['scalar','divl']:
            cp=EXP/f'checkpoints/performance2/critic/{m}/step_002000';manifest=read(cp/'manifest.json')
            assert manifest['complete'] and manifest['update']==2000
            for rel,rec in manifest['files'].items():assert sha(cp/rel)==rec['sha256']
            critic=build_pretrained(cfg(m));state=nnx.state(critic,nnx.Param)
            loaded=restore_tree(cp/'critic_params',state.to_pure_dict())
            state.replace_by_pure_dict(loaded);nnx.update(critic,state);critic.eval();critics[m]=critic;checkpoint_shas[m]=sha(cp/'manifest.json')
        loader=Replay('all');tokenizer=DIVLTokenizer(encoder_config())
    @nnx.jit
    def score(model,batch):
        z=model.encoder(**batch['obs'],train=False)
        return jnp.minimum(*model.q_head(z,batch['actions'],batch['action_mask']))
    @nnx.jit
    def grad(model,batch):return action_gradient(model,batch)
    ready={**identity(os.getpid()),'ready':True,'checkpoint_shas':checkpoint_shas,
           'frozen_actor_checkpoint_sha256':c['sft_checkpoint_manifest_sha256'],'physical_gpu':a.gpu,'socket':a.socket,
           'normalization_sha256':sha(c['normalization']),'no_updates':True,'jit':'fresh after FP32 masters and placement'}
    assert not Path(a.socket).exists()
    try:
        with Listener(a.socket,family='AF_UNIX',authkey=AUTHKEY) as listener:
            write(a.ready,ready,True)
            while True:
                with listener.accept() as connection:
                    while True:
                        try:req=connection.recv()
                        except EOFError:break
                        if req.get('command')=='stop':connection.send({'stopping':True});return
                        try:
                            if req.get('command')=='directions':
                                row=loader.get(req['episode_id'],req['chunk_index'])
                                row['raw'][0]=req['raw']
                                host=collate([row],loader.languages,tokenizer,actor.norm)
                                batch=jax.device_put(jax.tree.map(jnp.asarray,host),target)
                                out={'baseline_normalized_h5':np.asarray(host['actions'][0]),'gradients':{}}
                                for m,model in critics.items():out['gradients'][m]=np.asarray(jax.device_get(jax.block_until_ready(grad(model,batch))))[0]
                            elif req.get('command')=='score':
                                row=loader.get(req['episode_id'],req['chunk_index']);row['raw'][0]=req['raw']
                                host=collate([row],loader.languages,tokenizer,actor.norm);host['actions'][0]=req['normalized_h5']
                                batch=jax.device_put(jax.tree.map(jnp.asarray,host),target)
                                out={'q':{m:float(np.asarray(jax.device_get(jax.block_until_ready(score(model,batch))))[0]) for m,model in critics.items()}}
                            else:out=actor.infer(req)
                            connection.send({'ok':True,**out})
                        except Exception as ex:connection.send({'ok':False,'error':repr(ex)});raise
    finally:loader.close()


if __name__=='__main__':main()
