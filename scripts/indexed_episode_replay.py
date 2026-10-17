"""Prepared bounded-cache fallback for larger replay; pilot keeps in-memory replay."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from threading import Lock
import numpy as np

class IndexedEpisodeReplay:
    def __init__(self,manifest,*,seed=0,cache_episodes=8,prefetch_workers=2):
        self.records=sorted(json.loads(Path(manifest).read_text())['records'],key=lambda r:r['episode_id'])
        self.indices=[(e,i) for e,r in enumerate(self.records) for i in range(r['chunks'])]
        self.rng=np.random.default_rng(seed);self.capacity=cache_episodes
        self.cache=OrderedDict();self.lock=Lock();self.pool=ThreadPoolExecutor(max_workers=prefetch_workers)
        if not self.indices or cache_episodes<1:raise ValueError('Nonempty replay and bounded positive cache required')
    def _load(self,e):
        with self.lock:
            if e in self.cache:
                self.cache.move_to_end(e);return self.cache[e]
        with np.load(self.records[e]['path'],allow_pickle=False) as f:episode={k:f[k] for k in f.files}
        with self.lock:
            self.cache[e]=episode;self.cache.move_to_end(e)
            while len(self.cache)>self.capacity:self.cache.popitem(last=False)
        return episode
    def sample(self,batch_size):
        selected=[self.indices[i] for i in self.rng.integers(len(self.indices),size=batch_size)]
        # Group IO once per selected episode; preserve the original RNG/sample order.
        futures={e:self.pool.submit(self._load,e) for e in sorted({e for e,i in selected})}
        episodes={e:f.result() for e,f in futures.items()}
        return [(episodes[e],i) for e,i in selected]
    def close(self):self.pool.shutdown(wait=True)
