# Public release: deployment paths anonymized; algorithm equations unchanged.
"""Compile without execution; inspect exact XLA memory assignment for batch256."""
from pathlib import Path
import json
import flax.nnx as nnx
from flax.nnx import extract,graph
from flax.nnx.transforms.compilation import _jit_split_fn
import jax
import jax.numpy as jnp
from openpi.models.divl_tokenizer import DIVLTokenizer
from openpi.shared import normalize
from openpi.training.divl_libero_adapter import EpisodeReplay,make_batch
from openpi.training.libero_lwd_training import LiberoTrainingConfig,build_critic,encoder_config,initialize_training,critic_step
from train_libero_lwd import CRITIC_REPLAY_FIELDS

ROOT=Path('/opt/divl-trqam')

def main():
    cfg=LiberoTrainingConfig();enc=encoder_config(ROOT)
    replay=EpisodeReplay(ROOT/'data/pilot_libero_v1/episodes',seed=cfg.seed,fields=CRITIC_REPLAY_FIELDS);replay.refresh()
    samples=replay.sample(256);norm=normalize.load(ROOT/'checkpoints/pi05_libero/assets/physical-intelligence/libero')
    batch=make_batch(samples,DIVLTokenizer(enc),norm,td_steps=1);batch['critic_k']=jnp.asarray([e['lengths'][i] for e,i in samples])
    critic=build_critic(cfg,enc);target,vo,qo=initialize_training(critic,2000,learning_rate=cfg.learning_rate)
    with graph.update_context('jit'):
        args,kwargs=extract.to_tree(((critic,target,vo,qo,batch),{'config':cfg}),prefix=(None,None),
            split_fn=_jit_split_fn,check_aliasing=False,ctxtag='jit')
        compiled=critic_step.inner.lower(*args,**kwargs).compile()
    result={'batch':256,'compiled_memory':str(compiled.memory_analysis()),'execution_performed':False,
            'actor_updated':False,'dump_directory':str(ROOT/'artifacts/xla_critic_memory')}
    (ROOT/'artifacts/critic_xla_memory_analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
