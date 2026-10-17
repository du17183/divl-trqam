"""Dependency-free publication checks; never starts training or a simulator."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ArrayScalarOps:
    """Only scalar operations needed to exercise the actual tr_dual body."""

    minimum = staticmethod(min)
    asarray = staticmethod(lambda value: value)
    clip = staticmethod(lambda value, low, high: min(max(value, low), high))


def load_dual():
    tree = ast.parse((ROOT / 'src/openpi/training/trqam.py').read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'tr_dual')
    namespace = {'jnp': ArrayScalarOps}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<actual-tr-dual>', 'exec'), namespace)
    return namespace['tr_dual']


class DualFormulaChecks(unittest.TestCase):
    def test_slack_constraint_decreases_lambda(self):
        lam, ema, metrics = load_dual()(1., 0., 0.)
        self.assertAlmostEqual(lam, .99)
        self.assertEqual(ema, 0.)
        self.assertAlmostEqual(metrics['lambda_effective'], 2.97)

    def test_violation_increases_lambda(self):
        lam, ema, metrics = load_dual()(1., 2., 2.)
        self.assertAlmostEqual(lam, 1.01)
        self.assertAlmostEqual(ema, 2.)
        self.assertTrue(metrics['kl_over_budget'])

    def test_projected_floor(self):
        lam, _, metrics = load_dual()(.01, 0., 0.)
        self.assertEqual(lam, .01)
        self.assertAlmostEqual(metrics['lambda_effective'], .03)

    def test_projected_ceiling(self):
        lam, _, _ = load_dual()(100., 2., 2.)
        self.assertEqual(lam, 100.)

    def test_raw_kl_is_retained_when_ema_input_is_clipped(self):
        _, ema, metrics = load_dual()(1., 0., 100.)
        self.assertEqual(metrics['path_kl'], 100.)
        self.assertEqual(metrics['path_kl_clipped'], 2.)
        self.assertAlmostEqual(ema, .2)

    def test_ema_uses_previous_statistic(self):
        _, ema, _ = load_dual()(1., .5, 1.)
        self.assertAlmostEqual(ema, .55)


def main():
    provenance = json.loads((ROOT / 'provenance/source_manifest.json').read_text(encoding='utf-8'))
    for relative, item in provenance['files'].items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert actual == item['public_sha256'], f'Source SHA mismatch: {relative}'

    python_files = sorted(ROOT.rglob('*.py'))
    for path in python_files:
        if '.git' not in path.parts:
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path), feature_version=(3, 11))

    # The discoverable library entry point and experiment kernel must be the same AST.
    kernel = next(ROOT.glob('archive/libero_long/**/pilot_actor_math_v1.py'))
    original = ast.dump(ast.parse(kernel.read_text(encoding='utf-8')), include_attributes=False)
    exposed = ast.dump(ast.parse((ROOT/'src/openpi/training/trqam.py').read_text(encoding='utf-8')),
                       include_attributes=False)
    assert original == exposed, 'Published TRQAM kernel differs from archived experiment kernel'

    forbidden_extensions = {'.safetensors','.pt','.pth','.ckpt','.pkl','.npy','.npz','.h5','.hdf5','.mp4'}
    secret_patterns = [r'gh[pousr]_[A-Za-z0-9]{30,}', r'github_pat_[A-Za-z0-9_]{30,}',
                       r'AKIA[0-9A-Z]{16}', r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----']
    files = [p for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.parts]
    for path in files:
        assert path.suffix.lower() not in forbidden_extensions, f'Artifact file in release: {path}'
        assert path.stat().st_size < 5_000_000, f'Unexpected large file: {path}'
        text = path.read_text(encoding='utf-8')
        private_user = 'xiao' + 'long'
        assert f'/home/{private_user}' not in text and f'/{private_user}/' not in text, f'Private path: {path}'
        for pattern in secret_patterns:
            assert not re.search(pattern, text), f'Possible credential in {path}'

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(DualFormulaChecks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    assert result.wasSuccessful(), 'Dual formula checks failed'
    print(json.dumps({'source_sha_files': len(provenance['files']), 'python_files': len(python_files),
                      'release_files': len(files), 'dual_checks': result.testsRun,
                      'kernel_ast_equal': True, 'publication_scan': 'passed'}))


if __name__ == '__main__':
    main()
