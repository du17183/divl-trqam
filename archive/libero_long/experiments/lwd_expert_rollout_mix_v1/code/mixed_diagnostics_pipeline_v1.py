"""Fixed validation-only: reuse400 ranking branches and30 random branches;60 new gradients."""
from mixed_common_v2 import *
import collections

def run(q):
 from multiprocessing.connection import Client
 from pilot8_common_v1 import AUTHKEY
 if (EXP/'metrics/mixed_critic_comparison_complete015.json').exists():assert read(EXP/'metrics/mixed_critic_comparison_complete015.json')['passed'];return
 q.state['stage']='MIXED_CRITIC_FROZEN_HELDOUT_AND_ACTION_RANKING';q.refresh();jobs=[]
 for method,gpu in [('scalar',0),('divl',4)]:
  p=EXP/'metrics/diagnostics'/f'{method}_comparison015.json'
  if p.exists():assert read(p)['passed'];continue
  pid=None
  while pid is None:
   pid=q.launch('mixed_critic_diagnose_v1.py',['--method',method],(gpu,),method+'_diagnose015')
   if pid is None:q.refresh();time.sleep(10)
  jobs.append(pid)
 q.wait_all(jobs)
 q.state['stage']='MIXED_Q_GRADIENT_NATIVE_RETURN_VALIDATION60';q.refresh()
 budget=read(EXP/'configs/mixed_physical_validation015.json');pending=collections.deque()
 for c in budget['cases']:
  if not c['gradient_selected']:continue
  p=EXP/f"metrics/mixed_physical015/case_{c['case_id']:03d}.json"
  if p.exists():assert read(p)['passed'] and read(p)['complete'];continue
  pending.append(c['case_id'])
 servers={};workers={};idle=[]
 try:
  if pending:
   for g in range(8):
    sock=f'/tmp/pi05mixphysical015_g{g}.sock';ready=EXP/'metrics/diagnostics'/f'physical_gpu{g}_ready015.json';pid=None
    while pid is None:
     pid=q.launch('mixed_physical_inference_v1.py',['--gpu',g,'--socket',sock,'--ready',ready],(g,),f'mixed_physical_infer_g{g}_015')
     if pid is None:q.refresh();time.sleep(10)
    servers[g]=(pid,sock,ready);idle.append(g)
   deadline=time.monotonic()+3600
   while not all(v[2].exists() for v in servers.values()):
    for pid,_,_ in servers.values():assert q.poll(pid) is None,'Physical inference failed before readiness'
    assert time.monotonic()<deadline;q.refresh();time.sleep(5)
   while pending or workers:
    for g in list(idle):
     if not pending:break
     case=pending[0];pid=q.launch('mixed_physical_worker_v1.py',['--case-id',case,'--gpu',g,'--socket',servers[g][1]],(g,),f'mixed_physical_case{case:03d}_015',SIM,shared=True)
     if pid is not None:pending.popleft();workers[pid]=(g,case);idle.remove(g)
    for pid,(g,case) in list(workers.items()):
     out=q.poll(pid)
     if out:
      del workers[pid];idle.append(g);assert out['exit_code']==0,out
      j=read(EXP/f'metrics/mixed_physical015/case_{case:03d}.json');assert j['passed'] and j['complete'],j
    for pid,_,_ in servers.values():
     p,r,_=q.jobs[pid];assert_owned(p,r);assert p.poll() is None
    q.refresh();time.sleep(3)
 finally:
  for pid in list(workers):
   while pid in q.jobs:q.poll(pid);q.refresh();time.sleep(3)
  for pid,sock,ready in servers.values():
   if pid not in q.jobs:continue
   p,r,_=q.jobs[pid]
   if p.poll() is None and Path(sock).exists():
    exact(r)
    with Client(sock,family='AF_UNIX',authkey=AUTHKEY) as c:c.send({'command':'stop'});assert c.recv()['stopping']
   while pid in q.jobs:q.poll(pid);q.refresh();time.sleep(3)
 pid=None
 while pid is None:pid=q.launch('mixed_critic_report_v1.py',[],(),'mixed_critic_report015')
 q.wait_all([pid]);assert read(EXP/'metrics/mixed_critic_comparison_complete015.json')['passed']
