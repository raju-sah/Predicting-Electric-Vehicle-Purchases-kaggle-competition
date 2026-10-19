"""Feature-weighted local ridge regression of predictions on observed file scores.

Distances define locality around an existing anchor. They are not labels or
fabricated hidden scores. The requested score only defines a bounded direction.
"""
import numpy as np
CONFIGS={
 'simple':(),
 'distance':('global','local'),
 'score_normalized':('global','normalized'),
 'consensus':('global','consensus'),
 'combined':('global','normalized','consensus','spread'),
}
def fit(s,fam,P,anchor,center,mode='simple',alpha=.01,band=1e-4):
    s=np.asarray(s);fam=np.asarray(fam);P=np.asarray(P)
    counts={g:sum(fam==g) for g in np.unique(fam)}
    base=np.exp((s-s.max())/band)/np.array([counts[g] for g in fam]);base/=base.sum()
    W=np.broadcast_to(base[:,None],P.shape).copy()
    terms=[]
    if 'global' in CONFIGS[mode]:
        d=np.sqrt(np.mean((P-anchor)**2,axis=1));terms.append((d/max(np.median(d),1e-4))[:,None]**2)
    if 'local' in CONFIGS[mode]:
        d=np.abs(P-anchor);terms.append((d/np.maximum(np.median(d,axis=0),.001))**2)
    if 'normalized' in CONFIGS[mode]:
        d=np.abs(P/s[:,None]-anchor/center);terms.append((d/np.maximum(np.median(d,axis=0),.001))**2)
    if 'consensus' in CONFIGS[mode]:
        consensus=np.median(P,axis=0);d=np.abs(P-consensus);terms.append((d/np.maximum(np.median(d,axis=0),.001))**2)
    if 'spread' in CONFIGS[mode]:
        # Each file's distributional distance from the anchor, computed without labels.
        q=np.quantile(P,[.1,.25,.5,.75,.9],axis=1).T;aq=np.quantile(anchor,[.1,.25,.5,.75,.9]);d=np.sqrt(np.mean((q-aq)**2,axis=1));terms.append((d/max(np.median(d),1e-4))[:,None]**2)
    if terms:
        distance=sum(terms)/len(terms)
        W*=.2+.8/(1+distance)
    W/=W.sum(axis=0)
    x=(s-center)/1e-4;mx=np.sum(W*x[:,None],axis=0);my=np.sum(W*P,axis=0)
    vx=np.sum(W*x[:,None]**2,axis=0)-mx**2
    slope=(np.sum(W*x[:,None]*P,axis=0)-mx*my)/(vx+alpha)
    intercept=my-mx*slope
    residual=P-intercept-x[:,None]*slope
    r2=1-np.sum(W*residual**2,axis=0)/np.maximum(np.sum(W*(P-my)**2,axis=0),1e-12)
    return intercept,slope,r2

def gated_direction(s,actual,fam,P,anchor,center,mode='simple',ordinal=None):
    b0,slope,r2=fit(s,fam,P,anchor,center,mode)
    stable=[]
    for g in np.unique(fam):
        keep=fam!=g
        if keep.sum()<2:continue
        b=fit(s[keep],fam[keep],P[keep],anchor,center,mode)[1]
        stable.append(np.sign(b)==np.sign(slope))
    stability=np.mean(stable,axis=0) if stable else np.zeros(len(anchor))
    b=fit(actual,fam,P,anchor,center,mode)[1];tieagree=np.sign(b)==np.sign(slope)
    top=(actual==center)
    if ordinal is not None:top &= ordinal
    localagree=np.ones(len(anchor),dtype=bool)
    if top.sum()>1:
        x=(s[top]-center-1e-7)/9e-7;ff=fam[top];cnt={g:sum(ff==g) for g in np.unique(ff)};w=1/np.array([cnt[g] for g in ff]);w=w/w.sum();xc=x-w@x
        local=(w*xc)@P[top]/(w@(xc**2)+.05);localagree=np.sign(local)==np.sign(slope)
    confidence=np.clip(r2,0,1)*stability*(stability>=.8)*tieagree*localagree*(r2>=.2)
    return 3.7*slope,confidence,dict(r2=r2,stability=stability,tie_agree=tieagree,local_agree=localagree)
