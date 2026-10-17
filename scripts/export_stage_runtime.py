# Public release: deployment paths anonymized; algorithm equations unchanged.
import importlib.metadata as m,json
from pathlib import Path
r=Path('/opt/divl-trqam')
names=['jax','jaxlib','flax','numpy','orbax-checkpoint','optax','jaxtyping','etils','gemma','kauldron','tensorflow-cpu','tensorflow','ml-dtypes','torch','transformers','sentencepiece']
out={}
for name in names:
    try:
        d=m.distribution(name);out[name]={'version':d.version,'installed_path':str(d.locate_file(''))}
    except m.PackageNotFoundError:out[name]=None
(r/'artifacts/stage_runtime_versions.json').write_text(json.dumps(out,indent=2))
(r/'configs/stage_runtime_freeze.txt').write_text('\n'.join(sorted({d.metadata['Name']+'=='+d.version for d in m.distributions()}))+'\n')
print(json.dumps(out,indent=2))
