"""Strict full-state comparisons and prerequisite checks; no GPU import on load."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ABSOLUTE_TOLERANCE = 2e-5  # Existing failed accumulation gate threshold, never relaxed.


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def compare_array(left, right):
    """All leaves, exact shapes/dtypes/counters; absolute FP error only."""
    import numpy as np
    left, right = np.asarray(left), np.asarray(right)
    assert left.shape == right.shape and left.dtype == right.dtype, 'Shape/dtype mismatch'
    assert np.isfinite(left).all() and np.isfinite(right).all(), 'Nonfinite gate input'
    maximum = 0.
    changed = 0
    a, b = left.reshape(-1), right.reshape(-1)
    for first in range(0, a.size, 262144):
        x, y = a[first:first+262144], b[first:first+262144]
        changed += int(np.count_nonzero(x != y))
        if np.issubdtype(left.dtype, np.inexact):
            maximum = max(maximum, float(np.max(np.abs(x.astype(np.float64)-y.astype(np.float64)), initial=0)))
        else:
            assert np.array_equal(x, y), 'Optimizer/counter/integer mismatch'
    assert maximum <= ABSOLUTE_TOLERANCE, ('absolute_error', maximum, ABSOLUTE_TOLERANCE)
    return {'max_absolute_error':maximum, 'different_elements':changed, 'elements':int(a.size)}


def snapshot(tree, path):
    """Immutable, SHA checked full-leaf evidence, never delete failed evidence."""
    import jax
    import numpy as np
    path = Path(path)
    assert path.absolute().relative_to(ROOT).parts[:2] == ('data', 'gates')
    path.mkdir(parents=True, exist_ok=False)
    pairs, _ = jax.tree_util.tree_flatten_with_path(tree)
    rows = []
    for number, (key, leaf) in enumerate(pairs):
        value = np.asarray(jax.device_get(leaf))
        assert np.isfinite(value).all()
        target = path/('%06d.npy' % number)
        with target.open('xb') as stream:
            np.save(stream, value, allow_pickle=False)
        rows.append({'key':jax.tree_util.keystr(key), 'file':target.name,
                     'shape':list(value.shape), 'dtype':str(value.dtype),
                     'bytes':target.stat().st_size, 'sha256':sha(target)})
    manifest = {'leaves':rows, 'bytes':sum(r['bytes'] for r in rows),
                'all_leaves_saved':True, 'absolute_tolerance':ABSOLUTE_TOLERANCE}
    with (path/'manifest.json').open('x') as stream:
        json.dump(manifest, stream, indent=2)
    return manifest


def compare_snapshot(tree, path):
    import jax
    import numpy as np
    path = Path(path)
    manifest = json.loads((path/'manifest.json').read_text())
    assert manifest['all_leaves_saved'] and manifest['absolute_tolerance'] == ABSOLUTE_TOLERANCE
    pairs, _ = jax.tree_util.tree_flatten_with_path(tree)
    assert len(pairs) == len(manifest['leaves'])
    maximum, changed, elements, worst = 0., 0, 0, None
    for (key, value), record in zip(pairs, manifest['leaves'], strict=True):
        assert jax.tree_util.keystr(key) == record['key'], 'State structure mismatch'
        target = path/record['file']
        assert target.stat().st_size == record['bytes'] and sha(target) == record['sha256']
        expected = np.load(target, mmap_mode='r', allow_pickle=False)
        actual = np.asarray(jax.device_get(value))
        assert list(actual.shape) == record['shape'] and str(actual.dtype) == record['dtype']
        row = compare_array(expected, actual)
        if row['max_absolute_error'] > maximum:
            maximum, worst = row['max_absolute_error'], record['key']
        changed += row['different_elements']; elements += row['elements']
    return {'passed':True, 'max_absolute_error':maximum, 'different_elements':changed,
            'elements':elements, 'leaves':len(pairs), 'worst_leaf':worst,
            'absolute_tolerance':ABSOLUTE_TOLERANCE, 'relative_error_used':False,
            'evidence_manifest_sha256':sha(path/'manifest.json')}


def verify_sources():
    proof = ROOT/'provenance/critic_gpu_gate_tools002.json'
    record = json.loads(proof.read_text())
    assert record['absolute_tolerance'] == ABSOLUTE_TOLERANCE
    for relative, expected in record['inputs_sha256'].items():
        assert sha(ROOT/relative) == expected, relative
    return {'proof_path':str(proof), 'proof_sha256':sha(proof)}


def inspect_process(pid):
    path = Path('/proc')/str(pid)
    try:
        stat = (path/'stat').read_text().rsplit(')', 1)[1].split()
        return {'pid':int(pid), 'birth_ticks':stat[19], 'ppid':int(stat[1]), 'state':stat[0],
                'argv':(path/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0'),
                'cwd':str((path/'cwd').resolve())}
    except (FileNotFoundError, ProcessLookupError):
        return None


def preflight(gpus):
    """Fail before JAX, only inspect identities; never signal or launch a child."""
    proof = verify_sources()
    from critic_fullbatch_v1 import strict_upstream
    from critic_fullbatch_executor_v1 import validate_resources
    prereq = strict_upstream(scientific=False)
    identities = json.loads((ROOT/'provenance/takeover_20261009_thread01a111/identity_verified.json').read_text())
    for original in identities['processes']:
        if original['pid'] == 4079071:  # Superseded paused scheduler is protected, not a resource owner.
            current = inspect_process(original['pid'])
            if current and current['birth_ticks'] == original['birth_ticks']:
                for field in ['argv','ppid','cwd']:
                    assert current[field] == original[field], current
                assert current['state'] == 'T', current
            continue
        current = inspect_process(original['pid'])
        # Original supervisors and trainer must naturally exit, even if a complete file appeared first.
        assert not current or current['birth_ticks'] != original['birth_ticks'], ('original_still_present', current)
    original_snapshot = json.loads((ROOT/proof_snapshot()).read_text())
    old_checked = []
    for original in original_snapshot['old_task_processes']:
        if original.get('classification') != 'original_old_task_still_stopped':
            continue
        current = inspect_process(original['pid'])
        if current and current['birth_ticks'] == original['birth_ticks']:
            for field in ['argv','ppid','cwd']:
                assert current[field] == original[field], current
            assert current['state'] == 'T', current
        old_checked.append(current)
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            argv = (entry/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            continue  # Unreadable foreign identities are never inspected further or operated on.
        if any('critic_pretrained_gpu_gate_v1.py' in a for a in argv):
            current = inspect_process(int(entry.name))
            if current:
                assert False, ('another_gate_worker_alive_no_duplicate', current)
    assert len(gpus) == len(set(gpus)) and len(gpus) in [1,2,4,8]
    resources = validate_resources(gpus)
    return prereq, {**proof, 'resources':resources, 'old_identities':old_checked,
                    'new_worker_launched':False, 'signals_sent':False}


def proof_snapshot():
    proof = json.loads((ROOT/'provenance/critic_gpu_gate_tools002.json').read_text())
    return proof['old_identity_snapshot']
