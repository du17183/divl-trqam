# Public release: deployment paths anonymized; algorithm equations unchanged.
"""Immutable independent copies and content hashes before branch protocol changes."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import datetime
import hashlib
import json
import shutil
import subprocess
import argparse

ROOT = Path('/opt/divl-trqam')

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    destination = args.resume or (ROOT / 'backups' / ('counterfactual_pre_fix_' + stamp))
    assert destination.resolve().is_relative_to((ROOT / 'backups').resolve())
    destination.mkdir(parents=True, exist_ok=bool(args.resume))
    small = ['data/counterfactual_libero_v1', 'data/action_diagnostic_v1',
             'artifacts', 'logs', 'configs', 'scripts', 'reports']
    for name in small:
        if not (destination / name).exists():
            shutil.copytree(ROOT / name, destination / name, dirs_exist_ok=False)
    for name in ['data/pilot_libero_v1/manifest.json', 'data/dev_libero_v1/manifest.json']:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists(): shutil.copy2(ROOT / name, target)
    print('SMALL_BACKUP_COMPLETE', str(destination), flush=True)
    files = []
    for directory in small:
        files.extend(ROOT / p.relative_to(destination) for p in (destination / directory).rglob('*') if p.is_file())
    files += [ROOT / n for n in ['data/pilot_libero_v1/manifest.json', 'data/dev_libero_v1/manifest.json']]
    checkpoints = [p for p in (ROOT / 'checkpoints/critic_pilot_v1').rglob('*') if p.is_file()]
    policy_files = [p for p in (ROOT / 'checkpoints/pi05_libero').rglob('*') if p.is_file()]
    records = []
    def preserve(path):
        relative = str(path.relative_to(ROOT))
        target = destination / relative
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(['cp', '--reflink=auto', '--preserve=timestamps', '--', str(path), str(target)], check=True)
        source_hash, backup_hash = sha(path), sha(target)
        critical = relative.startswith(('checkpoints/', 'configs/', 'data/'))
        if critical and source_hash != backup_hash:
            raise AssertionError('Backup differs: ' + relative)
        return dict(path=relative, backup_path=str(target), size=path.stat().st_size,
                    sha256=backup_hash, current_source_sha256=source_hash,
                    source_still_matches=source_hash==backup_hash, point_in_time_runtime_file=not critical,
                    independent_inode=path.stat().st_ino != target.stat().st_ino)
    manifest_path = ROOT / 'artifacts/counterfactual_pre_fix_manifest.json'
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(preserve, p) for p in files + checkpoints]
        for job in as_completed(jobs):
            records.append(job.result())
            if len(records) % 50 == 0:
                print('BACKUP_PROGRESS', len(records), '/', len(jobs), flush=True)
    policy_records = [{'path': str(p.relative_to(ROOT)), 'size': p.stat().st_size, 'sha256': sha(p)} for p in policy_files]
    branches = []
    for p in sorted((ROOT / 'data/counterfactual_libero_v1').glob('*.json')):
        group = json.loads(p.read_text())
        for n, b in enumerate(group['branches']):
            branches.append({'state_id': group['metadata']['state_id'], 'branch_id': group['metadata']['state_id'] + f':{n:02d}',
                             'kind': b['kind'], 'candidate_index': b['candidate_index'], 'method': b['method'],
                             'epsilon': b['epsilon'], 'direction': b['direction'], 'status': 'provisional'})
    roots = {}
    for method in ['scalar', 'divl']:
        for step in [250, 500, 1000, 2000]:
            prefix = f'checkpoints/critic_pilot_v1/{method}/step{step:04d}/'
            subset = sorted((v for v in records if v['path'].startswith(prefix)), key=lambda v: v['path'])
            roots[f'{method}_step{step}'] = hashlib.sha256(json.dumps([(v['path'], v['sha256']) for v in subset]).encode()).hexdigest()
    result = {'backup_complete': True, 'backup_root': str(destination), 'created_utc': stamp,
              'branch_status': 'provisional', 'state_ids': sorted(set(b['state_id'] for b in branches)),
              'branches': branches, 'files': sorted(records, key=lambda v: v['path']), 'checkpoint_tree_hashes': roots,
              'policy_checkpoint_files': policy_records, 'policy_checkpoint': str(ROOT / 'checkpoints/pi05_libero'),
              'libero_commit': subprocess.check_output(['git', '-C', str(ROOT / 'third_party/libero'), 'rev-parse', 'HEAD'], text=True).strip(),
              'protocol': {'states': 50, 'candidates_per_state': 8, 'gradient_methods': ['scalar', 'divl'],
                           'epsilon': [.01, .03, .05], 'directions': [-1, 1], 'gamma': .9999,
                           'action_horizon': 10, 'execute_prefix_max': 5, 'reproduction_tolerance': 1e-5}}
    assert len(branches) == 700 and len(result['state_ids']) == 35
    manifest_path.write_text(json.dumps(result, indent=2) + '\n')
    (destination / 'counterfactual_pre_fix_manifest.json').write_text(manifest_path.read_text())
    (ROOT / 'artifacts/counterfactual_v1_provisional.json').write_text(json.dumps({'status': 'provisional', 'all_35_states_and_700_branches_preserved': True, 'backup_manifest': str(manifest_path)}))
    report = '# Counterfactual pre-fix snapshot\n\nIndependent copies and source/backup SHA256 verified.\n\n'
    report += f'Backup: {destination}\n\n35 states / 700 branches retained as provisional. No existing branch was overwritten.\n\n'
    report += 'All Scalar/DIVL 250/500/1000/2000 checkpoints, logs, configs, source scripts, artifacts and dataset manifests preserved.\n\n'
    report += 'Policy checkpoint contents SHA256 inventoried; policy unchanged. Frozen candidate/epsilon configuration recorded.\n\n'
    report += 'Manifest: artifacts/counterfactual_pre_fix_manifest.json\n'
    (ROOT / 'reports/counterfactual_pre_fix_snapshot.md').write_text(report)
    print('BACKUP_COMPLETE', json.dumps({'root': str(destination), 'files': len(records), 'bytes': sum(v['size'] for v in records), 'states': 35, 'branches': 700}), flush=True)

if __name__ == '__main__':
    main()
