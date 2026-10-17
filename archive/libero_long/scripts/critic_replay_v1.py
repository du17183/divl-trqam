"""Lazy access to independently audited new-policy Replay; no model or simulator imports."""
from pathlib import Path
import hashlib
import json
import h5py
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def draw_indices(by_task, update, batch_size, seed):
    """Stateless, task-balanced schedule; same draws and actions in both value arms."""
    assert update >= 1 and batch_size >= 1 and set(by_task) == set(range(10))
    rng = np.random.default_rng(np.random.SeedSequence([seed, update]))
    tasks = (np.arange(batch_size) + (update - 1) * batch_size) % 10
    rng.shuffle(tasks)
    indices = []
    for task in tasks:
        episodes = by_task[int(task)]
        assert episodes, 'No valid transitions for task; stop readiness, do not drop task'
        episode = episodes[int(rng.integers(len(episodes)))]
        indices.append((int(task), episode['episode_id'], int(rng.integers(episode['transitions']))))
    return indices


def extract_transition(group, index, task_id, episode_id):
    chunks = group['chunks']
    start = int(chunks['start_observation_index'][index])
    end = int(chunks['next_observation_index'][index])
    k = int(chunks['actual_k'][index])
    assert end - start == k and 1 <= k <= 5
    actions = np.asarray(chunks['planned_h5_normalized'][index], np.float32)
    fixed = np.asarray(chunks['planned_h5_fixed_mask'][index], np.float32)
    assert actions.shape == fixed.shape == (5, 7) and np.all(fixed == 1)
    assert np.isfinite(actions).all()
    observations = group['observations']
    raw = []
    for at in [start, end]:
        raw.append({key: observations[key][at] for key in ['agentview_rgb', 'eye_in_hand_rgb', 'proprio']})
    return {'task_id': task_id, 'episode_id': episode_id, 'chunk_index': index,
            'state_source_id': str(group.attrs['state_source_id']), 'observation_indices': [start, end],
            'raw': raw, 'actions': actions, 'action_mask': np.ones(5, bool),
            'rewards': np.float32(chunks['chunk_reward_discounted'][index]),
            'discounts': np.float32(chunks['bootstrap_discount'][index]), 'critic_k': np.int32(k),
            'native_terminal': bool(chunks['native_terminal'][index]),
            'goal_success': bool(chunks['goal_success'][index]),
            'bc_actual_h10': np.asarray(chunks['bc_actual_h10_normalized'][index], np.float32),
            'bc_actual_h10_mask': np.asarray(chunks['bc_actual_h10_mask'][index], np.float32),
            'bc_full_h10_eligible': bool(chunks['bc_full_h10_eligible'][index]),
            'collection_censored': bool(group.attrs['collection_censored'])}


def observation(raw, language, tokenizer, norm):
    stats = norm['statistics']['state']
    low = np.asarray(stats['q01'], np.float32)
    high = np.asarray(stats['q99'], np.float32)
    proprio = np.asarray(raw['proprio'], np.float32)
    assert proprio.shape == low.shape == high.shape == (8,) and np.isfinite(proprio).all()
    state = (proprio - low) / (high - low + np.float32(1e-6)) * 2 - 1
    tokens, mask = tokenizer.tokenize(language, state)
    images = {}
    for source, target in [('agentview_rgb', 'base_0_rgb'), ('eye_in_hand_rgb', 'left_wrist_0_rgb')]:
        image = np.asarray(raw[source])
        assert image.dtype == np.uint8 and image.shape == (128, 128, 3)
        rotated = np.ascontiguousarray(image[::-1, ::-1])
        resized = np.asarray(Image.fromarray(rotated).resize((224, 224), Image.Resampling.BILINEAR))
        images[target] = resized.astype(np.float32) / np.float32(127.5) - 1
    return {'images': images, 'image_masks': {key: np.asarray(True) for key in images},
            'token_ids': np.asarray(tokens, np.int32), 'token_mask': np.asarray(mask, bool)}


def collate(transitions, languages, tokenizer, norm):
    assert transitions
    def stack(items):
        if isinstance(items[0], dict):
            return {key: stack([row[key] for row in items]) for key in items[0]}
        return np.stack(items)
    result = {}
    for name, offset in [('obs', 0), ('next_obs', 1)]:
        result[name] = stack([observation(row['raw'][offset], languages[row['task_id']], tokenizer, norm)
                              for row in transitions])
    for key in ['actions', 'action_mask', 'rewards', 'discounts', 'critic_k']:
        result[key] = stack([row[key] for row in transitions])
    assert result['actions'].shape[1:] == (5, 7) and result['action_mask'].all()
    return result


class AuditedReplay:
    def __init__(self, manifest_path):
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = json.loads(self.manifest_path.read_text())
        m = self.manifest
        assert m['schema'] == 'independently_audited_selected_sft_replay_v1'
        assert m['complete'] and m['audit_passed'] and m['partition'] == 'train'
        assert m['common_replay_all_critic_actor_arms'] and m['expert_mixture_fraction'] == 0
        assert m['episodes'] in [300, 600] and m['physical_sources'] == 300
        assert sha(m['episodes_path']) == m['episodes_sha256']
        assert sha(m['transitions_path']) == m['transitions_sha256']
        self.records = json.loads(Path(m['episodes_path']).read_text())
        assert len(self.records) == m['episodes']
        split = json.loads((ROOT / 'configs/libero_10_split_manifest.json').read_text())
        self.languages = [task['language'] for task in split['tasks']]
        self.by_task = {task: [] for task in range(10)}
        self.files, self.handles = {}, {}
        for item in m['files']:
            assert sha(item['hdf5']) == item['sha256']
            self.files[(item['task_id'], item['repeat_index'])] = item['hdf5']
        for row in self.records:
            task = row['task_id']
            allowed = {x['state_source_id'] for x in split['tasks'][task]['partitions']['train']}
            assert row['state_source_id'] in allowed
            assert (task, row['repeat_index']) in self.files
            if row['transitions'] > 0:
                self.by_task[task].append(row)
        assert all(self.by_task.values()), 'Do not drop a task with no transitions'
        self.by_id = {row['episode_id']: row for row in self.records}
        assert len(self.by_id) == len(self.records)

    def samples(self, update, batch_size, seed):
        draw = draw_indices(self.by_task, update, batch_size, seed)
        rows = []
        for task, episode_id, index in draw:
            record = self.by_id[episode_id]
            key = (task, record['repeat_index'])
            if key not in self.handles:
                self.handles[key] = h5py.File(self.files[key], 'r')
            rows.append(extract_transition(self.handles[key][episode_id], index, task, episode_id))
        return rows, hashlib.sha256(json.dumps(draw, separators=(',', ':')).encode()).hexdigest()

    def close(self):
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()
