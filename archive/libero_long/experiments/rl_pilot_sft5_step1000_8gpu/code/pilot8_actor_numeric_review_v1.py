"""Observe cross-device reduction; no change to original same-layout gate."""
from pilot8_common_v1 import *
def cross_compare(tree,path):
    import jax,numpy as np
    man=read(Path(path)/'manifest.json');pairs,_=jax.tree_util.tree_flatten_with_path(tree);assert len(pairs)==len(man['leaves'])
    maximum=0.;worst=None;dot=0.;left2=0.;right2=0.;moment_count=0;elements=0
    for (key,value),r in zip(pairs,man['leaves'],strict=True):
        assert jax.tree_util.keystr(key)==r['key'];p=Path(path)/r['file'];assert sha(p)==r['sha256']
        x=np.load(p,mmap_mode='r');y=np.asarray(jax.device_get(value));assert x.shape==y.shape and x.dtype==y.dtype
        ismoment='mu' in r['key'] and 'opt' in r['key']
        if ismoment:moment_count+=1
        for i in range(0,x.size,262144):
            a=x.reshape(-1)[i:i+262144];b=y.reshape(-1)[i:i+262144];assert np.isfinite(a).all() and np.isfinite(b).all()
            if np.issubdtype(a.dtype,np.inexact):
                a=a.astype(np.float64);b=b.astype(np.float64);err=float(np.max(np.abs(a-b),initial=0))
                if err>maximum:maximum,worst=err,r['key']
                if ismoment:dot+=float(a@b);left2+=float(a@a);right2+=float(b@b)
            else:assert np.array_equal(a,b),'Counter/integer discrepancy'
        elements+=x.size
    assert moment_count and left2>0 and right2>0,'No real postclip optimizer gradient found'
    return {'max_absolute_error_observed':maximum,'worst_leaf':worst,'leaves':len(pairs),'elements':elements,'postclip_first_moment_cosine':dot/np.sqrt(left2*right2),'postclip_first_moment_norm_ratio':np.sqrt(right2/left2),'postclip_moment_leaves':moment_count,'cross_configuration_diagnostic_only':True,'same_configuration_restore_absolute_threshold_unchanged':2e-5,'source_sha256':sha(Path(path)/'manifest.json')}
