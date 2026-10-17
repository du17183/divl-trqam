"""Real-data exact check of worker count, prefetch and checkpoint sampling continuation."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
from benchmark_data_v2 import NativeExpertLoader

ROOT = Path(__file__).resolve().parents[1]


def flatten(x, prefix=''):
    if isinstance(x, dict):
        for k, v in sorted(x.items()): yield from flatten(v, prefix + '/' + k)
    else: yield prefix, x


def main():
    from openpi.models.tokenizer import PaligemmaTokenizer
    tokenizer = PaligemmaTokenizer(200)
    serial = NativeExpertLoader(10, 20261009, workers=0, prefetch=0)
    parallel = NativeExpertLoader(10, 20261009, workers=4, prefetch=2)
    resume = NativeExpertLoader(10, 20261009, workers=0, start_index=1)
    for loader in [serial, parallel, resume]: loader.set_tokens(tokenizer)
    rows = []
    started = time.monotonic()
    try:
        for index in range(2):
            a, sources, wait, read = serial.next(4)
            b, sources_b, wait_b, read_b = parallel.next(4)
            assert sources == sources_b
            hashes = {}
            for (key, x), (other, y) in zip(flatten(a), flatten(b)):
                assert key == other
                np.testing.assert_array_equal(x, y)
                hashes[key] = hashlib.sha256(x.tobytes()).hexdigest()
            if index == 1:
                c, sources_c, _, _ = resume.next(4)
                assert sources_c == sources
                for (_, x), (_, z) in zip(flatten(a), flatten(c)): np.testing.assert_array_equal(x, z)
            rows.append({'batch_index': index, 'sources': sources, 'array_sha256': hashes,
                         'serial_wait_s': wait, 'workers4_prefetch2_wait_s': wait_b})
    finally:
        for loader in [serial, parallel, resume]: loader.close()
    value = {'passed': True, 'synthetic': False, 'checks': ['exact real sample/label/image equality for workers0 vs4',
              'exact deterministic sample order across prefetch0 vs2', 'checkpoint continuation at global batch index1'],
             'rows': rows, 'wall_s': time.monotonic() - started}
    path = ROOT / 'metrics/loader_equivalence001.json'
    with path.open('x') as f: f.write(json.dumps(value, indent=2) + '\n')
    print(json.dumps({'passed': True, 'wall_s': value['wall_s']}), flush=True)


if __name__ == '__main__': main()
