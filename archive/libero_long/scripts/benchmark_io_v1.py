"""Shared, audited train/inference preprocessing and real held-out labels."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
from benchmark_data_v1 import ROOT, digest, load_sample, normalize


def observation_data(raw, state, language, tokenizer, norm):
    images = {}
    for source, dest in [('agentview_rgb', 'base_0_rgb'), ('eye_in_hand_rgb', 'left_wrist_0_rgb')]:
        image = np.ascontiguousarray(raw[source][::-1, ::-1])
        images[dest] = np.asarray(Image.fromarray(image).resize((224, 224), Image.Resampling.BILINEAR)).astype(np.float32) / 127.5 - 1
    images['right_wrist_0_rgb'] = np.zeros_like(images['base_0_rgb'])
    padded_state = np.zeros(32, np.float32)
    padded_state[:8] = normalize(np.asarray(state, np.float32), norm['statistics']['state'])
    tokens, token_mask = tokenizer.tokenize(language, state=None)
    return {'images': images, 'image_masks': {k: np.array(k != 'right_wrist_0_rgb', bool) for k in images},
            'state': padded_state, 'tokenized_prompt': np.asarray(tokens, np.int32), 'tokenized_prompt_mask': np.asarray(token_mask, bool)}


def unnormalize_actions(actions, norm):
    value = np.asarray(actions, np.float32)
    assert value.shape == (10, 32) and np.isfinite(value).all()
    stats = norm['statistics']['actions']
    low, high = np.asarray(stats['q01'], np.float32), np.asarray(stats['q99'], np.float32)
    return (value[:, :7] + 1) / 2 * (high - low + 1e-6) + low


def collate(samples, task_languages, tokenizer):
    tokens = [tokenizer.tokenize(task_languages[x['task_id']], state=None) for x in samples]
    n = len(samples)
    return {'images': {k: np.stack([x['images'][k] for x in samples]) for k in samples[0]['images']},
            'image_masks': {k: np.full(n, k != 'right_wrist_0_rgb', bool) for k in samples[0]['images']},
            'state': np.stack([x['state'] for x in samples]),
            'tokenized_prompt': np.stack([x[0] for x in tokens]).astype(np.int32),
            'tokenized_prompt_mask': np.stack([x[1] for x in tokens]).astype(bool),
            'actions': np.stack([x['actions'] for x in samples]), 'action_mask': np.stack([x['action_mask'] for x in samples])}


def fixed_task_batch(task_id, tokenizer, norm):
    """One fixed full H10 anchor per successful Dev expert, all fixed Dev starts still physically evaluated."""
    split = json.loads((ROOT / 'configs/libero_10_split_manifest.json').read_text())
    admission = json.loads((ROOT / 'data/expert/native_admission_v1.json').read_text())
    ids = {e['state_source_id'] for e in split['tasks'][task_id]['partitions']['dev']}
    episodes = sorted([e for e in admission['episodes'] if e['state_source_id'] in ids and e['native_success'] and e['length'] >= 10], key=lambda x: x['state_source_id'])
    rng = np.random.default_rng(2026100900 + task_id)
    specs = [(e, int(rng.integers(e['length'] - 9)), norm) for e in episodes]
    samples = [load_sample(s) for s in specs]
    sources = [{'task_id': task_id, 'state_source_id': e['state_source_id'], 'anchor': anchor} for e, anchor, _ in specs]
    return collate(samples, [x['language'] for x in split['tasks']], tokenizer), sources
