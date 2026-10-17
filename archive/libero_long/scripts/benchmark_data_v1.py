"""Lazy, task-balanced loader of admitted native expert observations/actions."""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import multiprocessing
import time

import h5py
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
_HANDLES = {}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def make_norm_stats():
    """Use each real frame once, only nested SFT10 Train; no held-out budget leakage."""
    path = ROOT / 'data/sft/common_norm_stats_v1.json'
    if path.exists(): return json.loads(path.read_text())
    manifest_path = ROOT / 'data/sft/sft_10_manifest.json'
    m = json.loads(manifest_path.read_text())
    assert m['admitted'] and len(m['episodes']) == 100
    actions, states, counts = [], [], []
    for e in m['episodes']:
        assert e['expert_training_eligible'] and e['native_success']
        with h5py.File(e['path'], 'r') as h:
            g = h[e['demo']]
            actions.append(g['actions'][:].astype(np.float64))
            states.append(g['proprio'][:].astype(np.float64))
            counts.append({'state_source_id': e['state_source_id'], 'task_id': e['task_id'], 'frames': len(g['actions'])})
    result = {'schema': 'train_only_common_quantile_normalization_v1', 'source_manifest_sha256': digest(manifest_path),
              'source_scope': 'exactly nested SFT10 successful Train trajectories; each actual action and observation once',
              'source_counts': counts, 'real_action_dim': 7, 'proprio_dim': 8, 'quantile_epsilon': 1e-6, 'statistics': {}}
    for key, seq in [('actions', actions), ('state', states)]:
        a = np.concatenate(seq)
        assert np.isfinite(a).all()
        result['statistics'][key] = {k: v.tolist() for k, v in {
            'mean': a.mean(axis=0), 'std': a.std(axis=0),
            'q01': np.quantile(a, .01, axis=0), 'q99': np.quantile(a, .99, axis=0)}.items()}
    with path.open('x') as f: f.write(json.dumps(result, indent=2) + '\n')
    return result


def normalize(a, stats):
    low, high = np.asarray(stats['q01'], np.float32), np.asarray(stats['q99'], np.float32)
    return (a - low) / (high - low + 1e-6) * 2 - 1


def load_sample(spec):
    """Worker imports only CPU libraries; it never initializes JAX or CUDA."""
    e, anchor, norm = spec
    if e['path'] not in _HANDLES: _HANDLES[e['path']] = h5py.File(e['path'], 'r')
    g = _HANDLES[e['path']][e['demo']]
    assert anchor + 10 <= e['length']
    images = {}
    for source, dest in [('agentview_rgb', 'base_0_rgb'), ('eye_in_hand_rgb', 'left_wrist_0_rgb')]:
        # Native dataset is raw OpenGL; one180 rotation matches the official eval adapter.
        raw = np.ascontiguousarray(g[source][anchor][::-1, ::-1])
        resized = np.asarray(Image.fromarray(raw).resize((224, 224), Image.Resampling.BILINEAR))
        images[dest] = resized.astype(np.float32) / 127.5 - 1
    images['right_wrist_0_rgb'] = np.zeros_like(images['base_0_rgb'])
    real_actions = np.asarray(g['actions'][anchor:anchor + 10], np.float32)
    actions = np.zeros((10, 32), np.float32)
    actions[:, :7] = normalize(real_actions, norm['statistics']['actions'])
    state = np.zeros((32,), np.float32)
    state[:8] = normalize(np.asarray(g['proprio'][anchor], np.float32), norm['statistics']['state'])
    mask = np.zeros((10, 32), np.float32); mask[:, :7] = 1
    return {'images': images, 'state': state, 'actions': actions, 'action_mask': mask,
            'task_id': e['task_id'], 'state_source_id': e['state_source_id'], 'anchor': anchor}


class NativeExpertLoader:
    def __init__(self, scale, seed, workers=0, prefetch=0, pinned=False, partition='train'):
        mpath = ROOT / 'data/sft' / ('sft_%02d_manifest.json' % scale)
        m = json.loads(mpath.read_text())
        assert m['admitted'], m['status']
        self.norm = make_norm_stats()
        self.manifest_sha = digest(mpath)
        if partition == 'train': episodes = m['episodes']
        else:
            split = json.loads((ROOT / 'configs/libero_10_split_manifest.json').read_text())
            admission = json.loads((ROOT / 'data/expert/native_admission_v1.json').read_text())
            ids = {x['state_source_id'] for t in split['tasks'] for x in t['partitions'][partition]}
            # Held-out FM uses native successful experts only; physical evaluation uses ALL fixed starts.
            episodes = [e for e in admission['episodes'] if e['state_source_id'] in ids and e['native_success']]
        self.by_task = [[e for e in episodes if e['task_id'] == t and e['length'] >= 10] for t in range(10)]
        assert all(self.by_task), 'Missing an eligible full real H10 task'
        self.rng = np.random.default_rng(seed)
        self.pool = concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) if workers else None
        self.prefetch_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1) if prefetch else None
        self.prefetch = prefetch
        self.pending = []
        self.pinned = pinned
        self.tokens = None
        self.host_bytes = 0
        self.load_wall_s = 0.
        self.frames_excluded_from_full_h10 = sum(9 for seq in self.by_task for _ in seq)

    def set_tokens(self, tokenizer):
        split = json.loads((ROOT / 'configs/libero_10_split_manifest.json').read_text())
        self.tokens = [tokenizer.tokenize(t['language'], state=None) for t in split['tasks']]

    def _batch(self, batch_size):
        specs = []
        for _ in range(batch_size):
            task = int(self.rng.integers(10))
            e = self.by_task[task][int(self.rng.integers(len(self.by_task[task])))]
            anchor = int(self.rng.integers(e['length'] - 9))
            specs.append((e, anchor, self.norm))
        before = time.monotonic()
        samples = list(self.pool.map(load_sample, specs)) if self.pool else [load_sample(s) for s in specs]
        assert self.tokens is not None
        data = {'images': {k: np.stack([x['images'][k] for x in samples]) for k in samples[0]['images']},
                'image_masks': {k: np.full(batch_size, k != 'right_wrist_0_rgb', bool) for k in samples[0]['images']},
                'state': np.stack([x['state'] for x in samples]),
                'tokenized_prompt': np.stack([self.tokens[x['task_id']][0] for x in samples]).astype(np.int32),
                'tokenized_prompt_mask': np.stack([self.tokens[x['task_id']][1] for x in samples]).astype(bool),
                'actions': np.stack([x['actions'] for x in samples]),
                'action_mask': np.stack([x['action_mask'] for x in samples])}
        if self.pinned:
            import torch
            def pin(v):
                if isinstance(v, dict): return {k: pin(x) for k, x in v.items()}
                return torch.from_numpy(v).pin_memory().numpy()
            data = pin(data)
        elapsed = time.monotonic() - before
        metadata = [{'task_id': x['task_id'], 'state_source_id': x['state_source_id'], 'anchor': x['anchor']} for x in samples]
        return data, metadata, elapsed

    def next(self, batch_size):
        start = time.monotonic()
        if self.prefetch_pool:
            if not self.pending:
                self.pending = [self.prefetch_pool.submit(self._batch, batch_size) for _ in range(self.prefetch)]
            data, metadata, load_wall = self.pending.pop(0).result()
            self.pending.append(self.prefetch_pool.submit(self._batch, batch_size))
        else: data, metadata, load_wall = self._batch(batch_size)
        self.load_wall_s += load_wall
        return data, metadata, time.monotonic() - start, load_wall

    def close(self):
        if self.prefetch_pool: self.prefetch_pool.shutdown(wait=True)
        if self.pool: self.pool.shutdown(wait=True)

