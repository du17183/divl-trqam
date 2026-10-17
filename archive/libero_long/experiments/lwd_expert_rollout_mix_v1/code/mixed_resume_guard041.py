# Public release: deployment paths anonymized; algorithm equations unchanged.
"""Resume041: audit historical protected identities without touching their processes."""
import hashlib,json,pathlib,runpy,sys
ROOT=pathlib.Path('/data/divl-trqam/benchmark')
EXP=ROOT/'experiments/lwd_expert_rollout_mix_v1';CODE=EXP/'code';PROOF=EXP/'provenance/resume_bc_eval041'
sys.path.insert(0,str(CODE))
_installed=False

def classify_protected(rec,stat_reader,identity_reader):
    try:birth,state=stat_reader(rec['pid'])
    except FileNotFoundError:return {'pid':rec['pid'],'historical_birth_ticks':str(rec['birth_ticks']),'status':'ORIGINAL_IDENTITY_ABSENT'}
    if str(birth)!=str(rec['birth_ticks']):return {'pid':rec['pid'],'historical_birth_ticks':str(rec['birth_ticks']),'status':'PID_REUSED_DO_NOT_INSPECT_OR_SIGNAL'}
    try:actual=identity_reader(rec['pid'])
    except FileNotFoundError:return {'pid':rec['pid'],'historical_birth_ticks':str(rec['birth_ticks']),'status':'ORIGINAL_IDENTITY_ABSENT'}
    assert all(actual[k]==rec[k] for k in ['birth_ticks','ppid','argv','cwd']),(rec,actual)
    return {'pid':rec['pid'],'status':'ORIGINAL_IDENTITY_STILL_PROTECTED','actual':actual}

def stat_reader(pid):
    s=(pathlib.Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
    return s[19],s[0]

def audit_historical_protected():
    import mixed_common_v2 as c
    from batch128_common_v1 import protected
    rows=[classify_protected(rec,stat_reader,c.identity) for rec in protected().values()]
    assert len(rows)==13
    return rows

def install_guard():
    global _installed
    if _installed:return
    import mixed_common_v2 as c
    p=PROOF/'sources041.json';proof=c.read(p)
    for name,digest in proof['files'].items():assert c.sha(name)==digest,name
    admission=c.read(PROOF/'admission041.json')
    assert admission['explicit_user_resume'] and admission['eval_only'] and admission['cpu_tests_passed']
    assert c.sha(PROOF/'user_resume041.txt')==admission['user_resume_sha256']
    assert c.sha(CODE/'mixed_actor_pipeline_v3.py')=='29e01971c09308103e30d489057940a7700df7d1e2b1f8986992ba82366c2172'
    c.protected_audit=audit_historical_protected
    _installed=True

def bootstrap(target):
    assert target in ['mixed_eval_inference_v1.py','mixed_eval_shard_v1.py','mixed_results_report_v1.py','mixed_snapshot_v1.py','mixed_artifact_snapshot_v1.py']
    install_guard();runpy.run_path(str(CODE/target),run_name='__main__')
