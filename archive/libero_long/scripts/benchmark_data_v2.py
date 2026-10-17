"""Versioned stateless global-batch sampling for reproducible prefetch and checkpoint resume."""
import numpy as np
from benchmark_data_v1 import NativeExpertLoader as _NativeExpertLoader


class NativeExpertLoader(_NativeExpertLoader):
    def __init__(self, scale, seed, workers=0, prefetch=0, pinned=False, partition='train', start_index=0):
        super().__init__(scale, seed, workers, prefetch, pinned, partition)
        self.seed = seed
        self.producer_index = start_index
        self.consumed_index = start_index

    def _batch(self, batch_size):
        index = self.producer_index
        self.producer_index += 1
        self.rng = np.random.default_rng(np.random.SeedSequence([self.seed, index]))
        data, sources, wall = super()._batch(batch_size)
        for source in sources: source['global_batch_index'] = index
        return data, sources, wall

    def next(self, batch_size):
        output = super().next(batch_size)
        assert all(x['global_batch_index'] == self.consumed_index for x in output[1])
        self.consumed_index += 1
        return output

