# Public release: deployment paths anonymized; algorithm equations unchanged.
"""Isolate GSPMD algebra using ORIGINAL Q/V losses, Adam, clip, transaction and EMA."""
from pathlib import Path
import json,time
import numpy as np
import jax,jax.numpy as jnp,flax.nnx as nnx
from flax.traverse_util import flatten_dict
from openpi.models.divl_heads import TwinQHead,DistributionalVHead
from openpi.training.libero_lwd_training import LiberoTrainingConfig,ScalarVHead,initialize_training,critic_step
from critic_data_parallel import CriticDataParallel
R=Path('/opt/divl-trqam');O=R/'artifacts/replay_action_audit_v3'
class TinyEncoder(nnx.Module):
    def __init__(self):self.net=nnx.Linear(3,640,rngs=nnx.Rngs(7))
    def __call__(self,images,image_masks,token_ids,token_mask,train=False):
        return jnp.tanh(self.net(images['base_0_rgb']))
def main():
    # GPU float32 GEMMs otherwise permit TF32 kernels whose tiling changes with
    # the per-device batch. Isolate GSPMD algebra with full FP32 here ONLY.
    jax.config.update('jax_default_matmul_precision','highest')
    assert jax.device_count()==4
    rng=np.random.default_rng(11);obs=lambda:dict(images={'base_0_rgb':rng.normal(size=(256,3)).astype(np.float32)},image_masks={'base_0_rgb':np.ones(256,dtype=bool)},token_ids=np.zeros((256,2),dtype=np.int32),token_mask=np.ones((256,2),dtype=bool))
    host=dict(obs=obs(),next_obs=obs(),actions=rng.normal(size=(256,10,7)).astype(np.float32),action_mask=np.ones((256,10),dtype=bool),rewards=rng.random(256).astype(np.float32),discounts=np.full(256,.9,dtype=np.float32),critic_k=np.full(256,5,dtype=np.int32))
    results={}
    for method in ['scalar','divl']:
        cfg=LiberoTrainingConfig(value_type=method)
        model=nnx.Dict(encoder=TinyEncoder(),q_head=TwinQHead(7,10,rngs=nnx.Rngs(9)),v_head=(ScalarVHead if method=='scalar' else DistributionalVHead)(rngs=nnx.Rngs(8)))
        target,vo,qo=initialize_training(model,2000,learning_rate=.0002)
        initial=jax.tree.map(lambda x:np.asarray(jax.device_get(x)).copy(),nnx.state(nnx.Dict(critic=model,target=target,value_optimizer=vo,q_optimizer=qo)))
        reference=None;params_reference=None;target_reference=None;output_reference=None;target_output_reference=None;method_results=[]
        @nnx.jit
        def forward(net,b):
            z=net.encoder(**b['obs'],train=False);q1,q2=net.q_head(z,b['actions'],b['action_mask']);return q1,q2,net.v_head(z)
        for count in [1,2,4]:
            dp=CriticDataParallel(model,target,vo,qo,jax.devices()[:count]);dp.restore(initial);batch=dp.batch(host)
            m={k:float(v) for k,v in jax.device_get(critic_step(model,target,vo,qo,batch,config=cfg)).items()}
            params=jax.tree.map(lambda x:np.asarray(jax.device_get(x)).copy(),nnx.state(model,nnx.Param));target_params=jax.tree.map(lambda x:np.asarray(jax.device_get(x)).copy(),nnx.state(target,nnx.Param))
            output=jax.tree.map(lambda x:np.asarray(jax.device_get(x)).copy(),forward(model,batch));target_output=jax.tree.map(lambda x:np.asarray(jax.device_get(x)).copy(),forward(target,batch))
            if count==1:
                reference=m;params_reference=params;target_reference=target_params;output_reference=output;target_output_reference=target_output;result=dict(devices=1,passed=True)
            else:
                metric_error=max(abs(m[k]-reference[k])/max(abs(reference[k]),1.) for k in m)
                param_error=max(float(np.max(np.abs(a-b))) for a,b in zip(jax.tree.leaves(params),jax.tree.leaves(params_reference),strict=True))
                target_error=max(float(np.max(np.abs(a-b))) for a,b in zip(jax.tree.leaves(target_params),jax.tree.leaves(target_reference),strict=True))
                flat=flatten_dict(params.to_pure_dict());ref=flatten_dict(params_reference.to_pure_dict())
                differences=sorted([(float(np.max(np.abs(v-ref[k]))),'/'.join(str(x) for x in k),v.size) for k,v in flat.items() if v is not None],reverse=True)
                total=sum(x.size for x in jax.tree.leaves(params));rms=float(np.sqrt(sum(float(np.sum((a-b).astype(np.float64)**2)) for a,b in zip(jax.tree.leaves(params),jax.tree.leaves(params_reference),strict=True))/total))
                output_error=max(float(np.max(np.abs(a-b))) for a,b in zip(jax.tree.leaves(output),jax.tree.leaves(output_reference),strict=True))
                target_output_error=max(float(np.max(np.abs(a-b))) for a,b in zip(jax.tree.leaves(target_output),jax.tree.leaves(target_output_reference),strict=True))
                result=dict(devices=count,metric_relative_error=metric_error,parameter_max_error=param_error,ema_max_error=target_error,
                            parameter_global_rms_error=rms,parameter_largest_differences=differences[:10],
                            q_and_v_output_max_error=output_error,target_output_max_error=target_output_error,
                            passed=metric_error<1e-4 and rms<1e-6 and output_error<1e-4 and target_output_error<1e-5,
                            interpretation='Full FP32 dot precision isolates global reduction from TF32/BF16 tiling noise; parameter and functional errors both reported')
            method_results.append(result);print('GLOBAL_UPDATE',method,json.dumps(result),flush=True)
        results[method]=method_results
    result=dict(passed=all(x['passed'] for v in results.values() for x in v),results=results,
                original_critic_step_used=True,global_batch=256,encoder='Small deterministic FP32 encoder to isolate exact reduction; NOT a formal architecture change',
                validates=['Global mean gradient','Double-Q/value losses','Two Adam optimizers','Global gradient clip','Finite transaction','EMA'],
                diagnostic_matmul_precision='highest; diagnostic ONLY, formal BF16 unchanged',
                full_bf16_backbone_bitwise_equivalence=False,actor_updated=False)
    (O/'gspmd_global_update_audit.json').write_text(json.dumps(result,indent=2));print('GSPMD_ALGEBRA_AUDIT',json.dumps(result),flush=True)
if __name__=='__main__':main()
