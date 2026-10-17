# Public release: deployment paths anonymized; algorithm equations unchanged.
"""Official LIBERO transforms, preserving learned FP32 masters at inference."""
from pathlib import Path
import argparse,logging
from openpi.models import model as model_lib
from openpi.policies.policy import Policy
from openpi.training import config,checkpoints
from openpi import transforms
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from serve_pilot_policy import SeededPolicy

R=Path('/opt/divl-trqam')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['scalar','divl'],required=True);p.add_argument('--port',type=int,default=8030);args=p.parse_args();logging.basicConfig(level=logging.INFO)
    train=config.get_config('pi05_libero');folder=R/'checkpoints/fixed_qam_v2'/args.method/'inference_step0050'
    model=train.model.load(model_lib.restore_params(folder/'params',dtype=None));data=train.data.create(train.assets_dirs,train.model);norm=checkpoints.load_norm_stats(folder/'assets',data.asset_id)
    policy=Policy(model,transforms=[transforms.InjectDefaultPrompt(None),*data.data_transforms.inputs,transforms.Normalize(norm,use_quantiles=data.use_quantile_norm),*data.model_transforms.inputs],
        output_transforms=[*data.model_transforms.outputs,transforms.Unnormalize(norm,use_quantiles=data.use_quantile_norm),*data.data_transforms.outputs],sample_kwargs=dict(num_steps=10),metadata=train.policy_metadata)
    print('POLICY_READY_FP32_MASTERS',flush=True);WebsocketPolicyServer(policy=SeededPolicy(policy),host='127.0.0.1',port=args.port).serve_forever()
