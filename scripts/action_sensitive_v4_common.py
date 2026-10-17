# Public release: deployment paths anonymized; algorithm equations unchanged.
"""Independent v4 artifacts; immutable v3 and fixed diagnostic inputs."""
from pathlib import Path
import json,hashlib,numpy as np
R=Path('/opt/divl-trqam')
D=R/'data/action_sensitive_replay_v4'
O=R/'artifacts/action_sensitive_replay_v4'
REP=R/'reports/action_sensitive_v4_20261007'
SEED=2026100731
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.partial')
 tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));tmp.replace(p)
def action_bounds():
 x=json.loads((R/'checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json').read_text())['norm_stats']['actions']
 return np.asarray(x['q01'],np.float32)[:7],np.asarray(x['q99'],np.float32)[:7]
def normalized(raw,low,high):return (2*(raw-low)/(high-low+1e-6)-1).astype(np.float32)
def raw_action(norm,low,high):return np.clip((norm+1)/2*(high-low+1e-6)+low,-1,1).astype(np.float32)
def support_bucket(a,pool,th):
 dist=float(np.sqrt(np.mean((pool-a.reshape(-1))**2,axis=1)).min())
 return ('Near-ID' if dist<=th['p50'] else 'Mid' if dist<=th['p90'] else 'Far-OOD'),dist

def gt_features(env,obs):
 """Current geometry only. No return, future success, clocks or candidate ID."""
 core=env.env;robot=core.robots[0];eef=np.asarray(obs['robot0_eef_pos'],float)
 quat=np.asarray(obs['robot0_eef_quat'],float)
 objects=getattr(core,'objects_dict',{});fixtures=getattr(core,'fixtures_dict',{})
 entries=dict(objects);entries.update(fixtures)
 assert len(entries)<=24
 blocks=[];names=[]
 for name,obj in sorted(entries.items()):
  body=env.sim.model.body_name2id(obj.root_body);pos=np.asarray(env.sim.data.body_xpos[body]);q=np.asarray(env.sim.data.body_xquat[body])
  geoms={env.sim.model.geom_name2id(g) for g in getattr(obj,'contact_geoms',[])}
  robot_contacts=0;other_contacts=0
  grip=robot.gripper.contact_geoms
  if isinstance(grip,dict):grip=[g for v in grip.values() for g in v]
  grip={env.sim.model.geom_name2id(g) for g in grip}
  for c in env.sim.data.contact[:env.sim.data.ncon]:
   a,b=int(c.geom1),int(c.geom2)
   robot_contacts+=int((a in geoms and b in grip) or (b in geoms and a in grip))
   other_contacts+=int((a in geoms and b not in geoms and b not in grip) or (b in geoms and a not in geoms and a not in grip))
  grasp=bool(core._check_grasp(gripper=robot.gripper,object_geoms=getattr(obj,'contact_geoms',[]))) if geoms else False
  blocks.append(np.r_[1.,pos,q,pos-eef,robot_contacts,other_contacts,float(grasp)])
  names.append(name)
 padded=np.zeros((24,14),np.float32)
 for i,b in enumerate(blocks):padded[i]=b
 return np.r_[eef,quat,padded.ravel()].astype(np.float32),names
