"""Read native raw Actor params without modifying any saved checkpoint bytes."""
from pilot8_common_v1 import *
import argparse
from multiprocessing.connection import Listener

def verify_sources():
    verify_worker()
    from batch128_common_v1 import protected
    protected()
    for name in ['resume_phase1_sources007.json','actor_resume_sources008.json','actor_eval_restore_sources015.json']:
        for rel,h in read(EXP/'provenance'/name)['files'].items():assert sha(ROOT/rel)==h,rel

def restore_actor_params(path,folder,target):
    import jax
    import jax.numpy as jnp
    import numpy as np
    import orbax.checkpoint as ocp
    from openpi.models.model import restore_params
    from critic_restore_current_layout_v1 import restore_tree
    cp=Path(path);manifest=read(cp/'manifest.json');sh=jax.sharding.SingleDeviceSharding(target)
    if not manifest.get('scientific_actor'):
        assert folder=='params'
        return restore_params(str(cp/folder),dtype=jnp.float32,sharding=sh),{'format':'OpenPI outer params; original SFT consumer unchanged'}
    assert manifest['passed'] and manifest['update']==100 and manifest['scientific_actor_updates']==100
    assert cp.resolve().is_relative_to((EXP/'checkpoints/actor').resolve())
    template=restore_params(str(Path(config()['sft_checkpoint'])/'params'),dtype=jnp.float32,sharding=sh)
    params=restore_tree(cp/folder,template)
    del template
    component='reference' if folder=='ema_params' else 'actor'
    item={'state':{component:params},'update':np.asarray(100,np.int32)}
    def args(x):
        if isinstance(x,jax.Array):return ocp.ArrayRestoreArgs(restore_type=jax.Array,dtype=x.dtype,sharding=x.sharding,global_shape=x.shape,strict=True)
        return ocp.RestoreArgs(restore_type=np.ndarray,dtype=x.dtype)
    with ocp.PyTreeCheckpointer() as loader:
        full=loader.restore(str((cp/'full_train_state').resolve()),args=ocp.args.PyTreeRestore(item=item,restore_args=jax.tree.map(args,item),partial_restore=True))
    assert int(full['update'])==100
    other=full['state'][component]
    assert jax.tree.structure(params)==jax.tree.structure(other)
    digest=hashlib.sha256();maxabs=0.;elements=0
    for (key,x),(_,y) in zip(jax.tree.flatten_with_path(params)[0],jax.tree.flatten_with_path(other)[0],strict=True):
        a=np.asarray(jax.device_get(x));b=np.asarray(jax.device_get(y))
        assert a.shape==b.shape and a.dtype==b.dtype
        assert np.isfinite(a).all() and np.array_equal(a,b),str(key)
        maxabs=max(maxabs,float(np.max(np.abs(a-b))));elements+=a.size
        digest.update(str(key).encode());digest.update(str(a.shape).encode());digest.update(str(a.dtype).encode());digest.update(a.tobytes())
    assert maxabs==0.
    del other,full
    return params,{'format':'native unwrapped NNX Param tree','full_train_state_component':component,'same_saved_fullstate_params_exact':True,'maxabs':maxabs,'elements':elements,'parameter_digest':digest.hexdigest(),'checkpoint_rewritten':False,'absolute_tolerance':2e-5}

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=range(8),required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--checkpoint-sha',required=True);p.add_argument('--params-folder',choices=['params','ema_params'],default='params');p.add_argument('--socket',required=True);p.add_argument('--ready',required=True);a=p.parse_args()
    verify_sources();runtime=runtime_receipt();c=config().copy();cp=Path(a.checkpoint);assert sha(cp/'manifest.json')==a.checkpoint_sha
    m=read(cp/'manifest.json');assert m['complete']
    for name,v in m['files'].items():assert sha(cp/name)==v['sha256'],name
    import jax
    import numpy as np
    assert physical_gpus()==(a.gpu,)
    devices=jax.devices();assert len(devices)==1 and all(d.platform=='gpu' for d in devices),devices
    target=devices[0];jax.config.update('jax_default_matmul_precision','highest');jax.config.update('jax_compilation_cache_dir',str(EXP/'cache/jax'))
    from openpi.models.pi0_config import Pi0Config
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.shared.nnx_utils import module_jit
    from sft_subset_inference_v1 import NativeActor
    actor=NativeActor.__new__(NativeActor)
    with jax.default_device(target):
        actor.params,audit=restore_actor_params(cp,a.params_folder,target)
        assert all(str(x.dtype)=='float32' and x.devices()=={target} for x in jax.tree.leaves(actor.params))
        actor.model=Pi0Config(pi05=True,action_horizon=10,discrete_state_input=False,dtype='float32').load(actor.params)
        actor.model.eval();actor.sample=module_jit(actor.model.sample_actions);actor.tokenizer=PaligemmaTokenizer(200);actor.norm=read(c['normalization'])
    lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True).splitlines()
    actual={line.split(',')[1].strip() for line in lines if int(line.split(',')[0])==os.getpid()};assert actual.issubset(set(UUIDS.values()))
    ready={**identity(os.getpid()),'ready':True,'socket':a.socket,'checkpoint':str(cp/a.params_folder),'normalization_sha256':sha(c['normalization']),'checkpoint_manifest_sha256':a.checkpoint_sha,'precision':'FP32 highest','jit_binding':'fresh after raw FP32restore/SingleDeviceplacement','jax_visible_devices':[str(d) for d in devices],'parameter_devices':[str(target)],'physical_parameter_gpu_uuid':UUIDS[a.gpu],'actual_allocated_uuids':sorted(actual),'native_adapter_source':'unchanged NativeActor.infer from sft_subset_inference_v1.py','runtime_revision':runtime,'raw_actor_restore_audit':audit,'sourceproof_sha256':sha(EXP/'provenance/actor_eval_restore_sources015.json'),'fresh_jit_first_request_check':'pending actual observation'}
    checked=False
    assert not Path(a.socket).exists()
    with Listener(a.socket,family='AF_UNIX',authkey=AUTHKEY) as listener:
        write(a.ready,ready,True);print('PILOT_NATIVE_ACTOR_READY',json.dumps(ready),flush=True)
        while True:
            with listener.accept() as connection:
                while True:
                    try:req=connection.recv()
                    except EOFError:break
                    if req.get('command')=='stop':connection.send({'stopping':True});return
                    try:
                        with jax.default_device(target):
                            result=actor.infer(req)
                            if not checked:
                                fresh=NativeActor.__new__(NativeActor);fresh.params=actor.params
                                fresh.model=Pi0Config(pi05=True,action_horizon=10,discrete_state_input=False,dtype='float32').load(fresh.params)
                                fresh.model.eval();fresh.sample=module_jit(fresh.model.sample_actions);fresh.tokenizer=actor.tokenizer;fresh.norm=actor.norm
                                again=fresh.infer(req);errs={}
                                for key in ['normalized_h10','raw_h10']:
                                    x=np.asarray(result[key]);y=np.asarray(again[key]);errs[key]=float(np.max(np.abs(x-y)))
                                    assert np.isfinite(x).all() and np.array_equal(x,y),(key,errs[key])
                                ready['fresh_jit_first_request_check']={'passed':True,'maxabs':errs,'absolute_tolerance':2e-5,'exact_output_required':True,'actual_native_observation':True}
                                write(a.ready,ready);checked=True;del fresh,again
                        connection.send({'ok':True,**result})
                    except Exception as e:connection.send({'ok':False,'error':repr(e)});raise
if __name__=='__main__':main()
