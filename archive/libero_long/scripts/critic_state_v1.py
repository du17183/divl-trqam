"""Strict prerequisites and immutable full-state Critic checkpoints; no model import at module load."""
from pathlib import Path
import datetime
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def read(relative):
    path = ROOT / relative
    return json.loads(path.read_text()) if path.exists() else None


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def readiness(require_execution=False):
    missing = []
    for relative in ['metrics/sft_selection_v2.json', 'metrics/replay_stage002_complete.json',
                     'metrics/critic_contract_gate001.json']:
        if not (ROOT / relative).exists():
            missing.append(relative)
    if require_execution:
        for relative in ['metrics/critic_pretrained_gpu_gate001.json', 'configs/critic_execution_v1.json']:
            if not (ROOT / relative).exists():
                missing.append(relative)
    selection = read('metrics/sft_selection_v2.json')
    if selection and selection.get('status') == 'STOP_BASELINE_ADMISSION':
        return {'ready': False, 'stop': 'STOP_BASELINE_ADMISSION', 'missing': missing}
    replay = read('metrics/replay_stage002_complete.json')
    ready = not missing and selection and selection.get('audit_passed') and selection.get('selected')
    ready = ready and replay and replay.get('complete') and replay.get('policy_replay_collected')
    return {'ready': bool(ready), 'stop': None, 'missing': missing,
            'scope': 'file readiness only; does not initialize models, launch workers or infer performance'}


def verified_prerequisites(require_execution=False):
    status = readiness(require_execution)
    assert status['ready'], status
    protocol = read('configs/critic_protocol_v1.json')
    prep = read('provenance/critic_stage_preparation001.json')
    assert digest(ROOT / 'configs/critic_protocol_v1.json') == prep['config_sha256']
    for relative, expected in prep['source_inputs_sha256'].items():
        assert digest(ROOT / relative) == expected, relative
    proof = read('provenance/critic_execution_tools001.json')
    assert proof is not None, 'New execution sources must be frozen before any job'
    for relative, expected in proof['inputs_sha256'].items():
        assert digest(ROOT / relative) == expected, relative
    selection = read('metrics/sft_selection_v2.json')
    assert selection['complete'] and selection['audit_passed'] and selection['status'] == 'SFT_BASELINE_ADMITTED'
    stage = read('metrics/replay_stage002_complete.json')
    assert stage['stage'] == 'selected_sft_replay300_audited_critic_pending'
    assert stage['sft_selection_sha256'] == digest(ROOT / 'metrics/sft_selection_v2.json')
    manifest_path = Path(stage['replay_manifest'])
    assert digest(manifest_path) == stage['replay_manifest_sha256']
    manifest = json.loads(manifest_path.read_text())
    assert manifest['complete'] and manifest['audit_passed'] and manifest['partition'] == 'train'
    assert manifest['sft_selection_sha256'] == digest(ROOT / 'metrics/sft_selection_v2.json')
    assert manifest['checkpoint'] == selection['selected_checkpoint']
    assert manifest['common_replay_all_critic_actor_arms'] and manifest['expert_mixture_fraction'] == 0
    assert read('metrics/critic_contract_gate001.json')['passed']
    public = read('provenance/critic_public_initialization001.json')
    for item in public['files']:
        assert digest(item['path']) == item['sha256'], item['path']
    result = {'protocol': protocol, 'replay_manifest_path': str(manifest_path),
              'protocol_sha256': digest(ROOT / 'configs/critic_protocol_v1.json'),
              'replay_manifest_sha256': digest(manifest_path),
              'sft_selection_sha256': digest(ROOT / 'metrics/sft_selection_v2.json'),
              'execution_sources_sha256': digest(ROOT / 'provenance/critic_execution_tools001.json')}
    if require_execution:
        execution = read('configs/critic_execution_v1.json')
        assert execution['passed'] and execution['full_pretrained_gate_passed']
        assert execution['protocol_sha256'] == result['protocol_sha256']
        assert execution['replay_manifest_sha256'] == result['replay_manifest_sha256']
        assert execution['global_batch'] == protocol['global_batch'] == 256
        assert execution['native_full_batch_only'], 'Failed accumulation gate cannot be silently admitted'
        for relative, expected in execution['inputs_sha256'].items():
            assert digest(ROOT / relative) == expected, relative
        result.update(execution=execution, execution_config_sha256=digest(ROOT / 'configs/critic_execution_v1.json'))
    return result


def payload(bundle, update):
    import flax.nnx as nnx
    import numpy as np
    return {'state': nnx.state(bundle).to_pure_dict(), 'update': np.asarray(update, np.int32)}


def save(bundle, update, path, metadata):
    import flax.nnx as nnx
    import orbax.checkpoint as ocp
    import time
    path = Path(path)
    relative = path.absolute().relative_to(ROOT)
    assert relative.parts[:2] in [('checkpoints', 'critic'), ('data', 'gates')], 'New project checkpoint namespace only'
    path.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    with ocp.PyTreeCheckpointer() as saver:
        saver.save(str((path / 'full_train_state').resolve()), payload(bundle, update))
        saver.save(str((path / 'critic_params').resolve()), nnx.state(bundle.critic, nnx.Param).to_pure_dict())
    files = {p.relative_to(path).as_posix(): {'bytes': p.stat().st_size, 'sha256': digest(p)}
             for p in path.rglob('*') if p.is_file()}
    manifest = {**metadata, 'complete': True, 'update': update, 'next_sampler_update': update + 1,
                'saved_utc': stamp(), 'files': files, 'bytes': sum(e['bytes'] for e in files.values()),
                'save_and_hash_wall_s': time.monotonic() - began,
                'full_state_components': ['online_encoder_q_v', 'target_encoder_q_v', 'value_adam', 'q_adam', 'update']}
    atomic(path / 'manifest.json', manifest)
    return manifest


def restore(bundle, path, *, expected_metadata=None):
    import flax.nnx as nnx
    import jax
    import orbax.checkpoint as ocp
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    assert manifest['complete'] and manifest['next_sampler_update'] == manifest['update'] + 1
    for relative, expected in manifest['files'].items():
        assert digest(path / relative) == expected['sha256'], relative
    for key, expected in (expected_metadata or {}).items():
        assert manifest[key] == expected, key
    state = nnx.state(bundle)
    template = payload(bundle, manifest['update'])
    with ocp.PyTreeCheckpointer() as loader:
        loaded = loader.restore(str((path / 'full_train_state').resolve()), item=template)
    assert int(loaded['update']) == manifest['update']
    # Rebind restored weights to the fresh NNX object and to this job's declared sharding.
    placed = jax.tree.map(lambda value, reference: jax.device_put(value, reference.sharding),
                         loaded['state'], state.to_pure_dict())
    state.replace_by_pure_dict(placed)
    nnx.update(bundle, state)
    return manifest
