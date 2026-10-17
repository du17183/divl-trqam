"""Existing native actual-action loader with explicit immutable subset and Train-only norm."""
import concurrent.futures
import json
import multiprocessing
from pathlib import Path
import numpy as np
from benchmark_data_v1 import digest
from benchmark_data_v2 import NativeExpertLoader as Parent


class NativeExpertLoader(Parent):
    def __init__(self, manifest, norm, seed, workers=0, prefetch=0, pinned=False, start_index=0):
        m = json.loads(Path(manifest).read_text())
        self.norm = json.loads(Path(norm).read_text())
        self.manifest_sha = digest(manifest)
        assert m['admitted'] and self.norm['source_manifest_sha256'] == self.manifest_sha
        episodes = m['episodes']
        assert {e['state_source_id'] for e in episodes} == {e['state_source_id'] for e in self.norm['source_counts']}
        self.by_task = [[e for e in episodes if e['task_id'] == t and e['length'] >= 10] for t in range(10)]
        assert all(self.by_task)
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.producer_index = self.consumed_index = start_index
        self.pool = concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) if workers else None
        self.prefetch_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1) if prefetch else None
        self.prefetch = prefetch
        self.pending = []
        self.pinned = pinned
        self.tokens = None
        self.host_bytes = 0
        self.load_wall_s = 0.
        self.frames_excluded_from_full_h10 = sum(9 for seq in self.by_task for _ in seq)
