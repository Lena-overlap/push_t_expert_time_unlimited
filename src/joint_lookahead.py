"""Joint SE(2) contact lookahead."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from geometry import predict_iou, CAPS


LOW=np.array([-.29,-.28]); HIGH=np.array([.25,.28])
APP_LOW=np.array([-.26,-.25]); APP_HIGH=np.array([.21,.25])
W=np.array([1.,1.25,.070])

@dataclass(frozen=True)
class Face:
    xy: tuple
    normal: tuple

@dataclass(frozen=True)
class Proposal:
    face: str
    distance: float
    estimate: float
    mode: str='push'
    tail: tuple=()
    predicted_steps: int=0
    predicted_iou: float=0.

# From the original closed-tip rotation primitive, plus intermediate positions.
# Extra points are geometry-based proposals, NOT new measured calibration.
ADDED={
 'xp_lever_lower':('xp_center',(-.060,-.043)),
 'xp_lever_upper':('xp_center',(-.060,.043)),
 'xp_mid_lower':('xp_center',(-.060,-.028)),
 'xp_mid_upper':('xp_center',(-.060,.028)),
 'yp_stem_tip':('yp_stem_mid',(.037,-.0175)),
 'yn_stem_tip':('yn_stem_mid',(.037,.0175)),
 'yp_bar_far':('yp_bar_near',(-.052,-.060)),
 'yn_bar_far':('yn_bar_near',(-.052,.060)),
}

def augmented(priors,faces):
    """Keep calibrated values unchanged. Extrapolate lever torque only for new points."""
    rates={k:np.array(v,float) for k,v in priors.items()}; fs=dict(faces)
    for name,(base,point) in ADDED.items():
        f=fs[base]; n=np.asarray(f.normal,float); d=np.asarray(point)-f.xy
        lever=d[0]*n[1]-d[1]*n[0]
        v=rates[base].copy(); v[2]=np.clip(v[2]+180.*lever,-13.,13.)
        rates[name]=v;fs[name]=Face(point,tuple(n))
    return rates,fs

def loss(q):
    q=np.asarray(q,float)
    angle=np.arctan2(np.sin(q[...,2]),np.cos(q[...,2]))
    return np.linalg.norm(np.concatenate([q[...,:2]*W[:2],(angle*W[2])[...,None]],axis=-1),axis=-1)

def rotate_batch(a,xy):
    c=np.cos(a);s=np.sin(a)
    return np.stack([c*xy[...,0]-s*xy[...,1],s*xy[...,0]+c*xy[...,1]],axis=-1)

class Lattice:
    def __init__(self,priors,faces):
        self.names=tuple(priors);self.face_count=len(self.names)
        self.rates=np.stack([priors[k] for k in self.names]);self.points=np.array([faces[k].xy for k in self.names]);self.normals=np.array([faces[k].normal for k in self.names]);
        self.caps=np.array([CAPS.get(k,.065 if k.startswith('x') else .075) for k in self.names]);self.faces=faces

    def expand(self,q,tool,budget,goal,heading,last=None):
        """Vectorized candidate states, time and reach for a batch of parent poses.
        Step estimates are soft heuristics, not hardware guarantees.
        """
        q=np.atleast_2d(q);tool=np.atleast_2d(tool);budget=np.asarray(budget).reshape(-1)
        a=q[:,None,2]; R=heading+a
        normal=rotate_batch(R,self.normals[None,:,:]); point=goal+rotate_batch(heading,q[:,:2])[:,None,:]+rotate_batch(R,self.points[None,:,:])
        approach=point-.024*normal; end0=point+.018*normal
        feasible=np.all((approach>=APP_LOW)&(approach<=APP_HIGH),axis=-1)&np.all((end0>=LOW)&(end0<=HIGH),axis=-1)
        vxy=rotate_batch(a,self.rates[None,:,:2]);v=np.concatenate([vxy,np.broadcast_to(self.rates[None,:,2:3],(*a.shape[:1],self.face_count,1))],axis=-1)
        vw=v*W; ideal=-np.sum(q[:,None,:]*W*vw,axis=-1)/np.maximum(np.sum(vw**2,axis=-1),1e-10)
        axial=-np.sum(q[:,None,:2]*vxy,axis=-1)/np.maximum(np.sum(vxy**2,axis=-1),1e-10)
        # No artificial 2mm moves in far states. Fine adjustment is useful near target.
        near=loss(q)<.025; minimum=np.where(near,.0025,.009)[:,None]
        grid=np.broadcast_to(np.array([.010,.020,.035,.050,.075,.100]),(*ideal.shape,6))
        distances=np.concatenate([grid,ideal[...,None],axial[...,None],(.65*ideal)[...,None],(1.2*ideal)[...,None],np.broadcast_to(.004,(*ideal.shape,1))],axis=-1)
        distances=np.minimum(distances,self.caps[None,:,None]); distances=np.round(distances,4)
        end=end0[:,:,None,:]+normal[:,:,None,:]*distances[...,None]
        valid=feasible[...,None]&(distances>=minimum[...,None])&np.all((end>=LOW)&(end<=HIGH),axis=-1)
        # Conservative but not a hard deadline: allow an 8-step estimate slack.
        # The forward controller uses real executed steps for all actual limits.
        steps=np.ceil(40+np.linalg.norm(tool[:,None,:]-approach,axis=-1)[...,None]/.0045+np.maximum(distances,0)/.0034)
        valid &= steps<=budget[:,None,None]+8
        if last is not None: valid &= (np.arange(self.face_count)[None,:,None]!=np.asarray(last)[:,None,None])
        child=q[:,None,None,:]+v[:,:,None,:]*distances[...,None];child[...,2]=np.arctan2(np.sin(child[...,2]),np.cos(child[...,2]));
        # The servo has a finite material tracking range. Do not propose arbitrarily huge turns.
        valid &= np.abs(v[:,:,None,2]*distances)<=np.deg2rad(38)
        costs=loss(child)+.000045*steps
        costs=np.where(valid,costs,np.inf)
        return child.reshape(len(q),-1,3),end.reshape(len(q),-1,2),steps.reshape(len(q),-1),costs.reshape(len(q),-1),distances.reshape(len(q),-1),np.repeat(np.arange(self.face_count),distances.shape[-1])

    def plan(self,q,eef,remaining,goal,heading,limit=8,closed=True):
        budget=remaining-(0 if closed else 24)-7
        if budget<15: return []
        q=np.asarray(q);goal=np.asarray(goal); tool=np.asarray(eef)[:2]
        out=self.expand(q,tool,[budget],goal,heading)
        states,tools,times,costs,distances,face_ids=[x[0] if i<5 else x for i,x in enumerate(out)]
        # Deduplicate clipped/equal distances before ranking contact proposals.
        ids=[];seen=set()
        for i in np.argsort(costs):
            if not np.isfinite(costs[i]):continue
            key=(face_ids[i],round(float(distances[i]),3))
            if key not in seen:seen.add(key);ids.append(i)
        if not ids:return []
        ids=np.asarray(ids,int); states=states[ids];tools=tools[ids];times=times[ids];costs=costs[ids];distances=distances[ids];faces=face_ids[ids]
        best=costs.copy();best_time=times.copy();best_state=states.copy();tails=[() for _ in ids]
        # For each first move retain its two best continuations, then try the
        # next move from each. This preserves temporarily worsening actions.
        parent=np.arange(len(ids)); q0=states; t0=tools; elapsed=times; last=faces; seq=[() for _ in ids]
        for _ in range(3):
            if not len(parent):break
            eligible=elapsed<budget-18
            if not np.any(eligible):break
            pidx=np.where(eligible)[0]
            child,tool2,dt,cost2,d2,f2=self.expand(q0[pidx],t0[pidx],budget-elapsed[pidx],goal,heading,last[pidx])
            total=elapsed[pidx,None]+dt;rank=cost2+.000045*elapsed[pidx,None]
            nxt=[]
            for row,pi in enumerate(pidx):
                for j in np.argsort(rank[row])[:2]:
                    if not np.isfinite(rank[row,j]):continue
                    base=parent[pi]; chain=seq[pi]+((self.names[f2[j]],float(d2[row,j])),)
                    value=float(rank[row,j]);
                    if value<best[base]:best[base]=value;best_time[base]=total[row,j];best_state[base]=child[row,j];tails[base]=chain
                    nxt.append((base,child[row,j],tool2[row,j],total[row,j],f2[j],chain,value))
            # At most 2 continuation states per first move, even at depth 4.
            if not nxt:break
            nxt.sort(key=lambda x:x[-1]);kept=[];counts={}
            for x in nxt:
                if counts.get(x[0],0)<2:kept.append(x);counts[x[0]]=counts.get(x[0],0)+1
            parent=np.array([x[0] for x in kept]);q0=np.stack([x[1] for x in kept]);t0=np.stack([x[2] for x in kept]);elapsed=np.array([x[3] for x in kept]);last=np.array([x[4] for x in kept]);seq=[x[5] for x in kept]
        # Exact T polygon scoring of the best predicted endpoint for each
        # first action. Favor predicted completion over a short partial move.
        ious=np.array([predict_iou(x) for x in best_state]);final_cost=.75*loss(best_state)+.026*(1-ious)+.00004*best_time-.008*(ious>=.88)
        opts=[]
        for i in np.argsort(final_cost):
            face=self.names[faces[i]];d=float(distances[i]);tail=tails[i]
            opts.append(Proposal(face,d,float(final_cost[i]),'trim' if not tail else 'push',tail,int(best_time[i]),float(ious[i])))
        # Prioritize different contacts, then alternatives with meaningfully different lengths.
        # Force a high-quality one-contact candidate when near goal.
        selected=[];chosen=set()
        for p in opts:
            if p.face not in chosen: selected.append(p);chosen.add(p.face)
            if len(selected)>=min(4,limit):break
        if loss(q)<.035:
            terminal=min((p for p in opts if not p.tail),key=lambda p:p.estimate,default=None)
            if terminal is not None and terminal not in selected:selected.insert(0,terminal)
        for p in opts:
            if len(selected)>=limit:break
            if not any(p.face==s.face and abs(p.distance-s.distance)<.007 for s in selected):selected.append(p)
        return selected[:limit]

def terminal_distance(q,delta,max_distance):
    """Real local response, not an assumed contact model. Maximize exact polygon IoU."""
    q=np.asarray(q);delta=np.asarray(delta);grid=np.linspace(0,max(0.,max_distance),13)
    return float(grid[np.argmax([predict_iou(q+delta*d) for d in grid])])
