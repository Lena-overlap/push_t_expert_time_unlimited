"""Calibrated contact dynamics and staged contact planning."""
from __future__ import annotations

import json
import numpy as np
from itertools import product
from pathlib import Path
from scipy.optimize import least_squares, minimize_scalar
from geometry import Plan, PRIORS as SHORT_PRIORS, POINTS, NORMALS, CAPS, rot, predict_iou


LONG_SAMPLES = {
    'xp_center': [(1., .09770561, -.10313273),(1., -.02033278, -.02023573)],
    'yp_bar_near': [(.20695348,1.,-7.16993335)],
    'yp_stem_mid': [(.05808682,1.,8.34257911)],
    'yn_bar_near': [(.14622645,-1.,4.35017714)],
    'yn_stem_mid': [(-.06411702,-1.,-9.18392642)],
}
standard_priors = {k: np.asarray(v,float) for k,v in SHORT_PRIORS.items()}
for k,vs in LONG_SAMPLES.items():
    w=.50 if k[0]=='x' else .75
    standard_priors[k]=(1-w)*standard_priors[k]+w*np.mean(vs,axis=0)

wide_priors = {k: v.copy() for k,v in standard_priors.items()}
WIDE_SAMPLES={
    'yn_stem_mid': [(-.0045335592/.0462566581,-1.,np.deg2rad(-14.0351483)/.0462566581)],
    'yp_bar_near': [(.0090310959/.0658884914,1.,np.deg2rad(-20.7656697)/.0658884914)],
    'yp_stem_mid': [(-.0028634405/.0312698967,1.,np.deg2rad(7.3618448)/.0312698967)],
}
for k,vs in WIDE_SAMPLES.items():
    wide_priors[k]=.65*wide_priors[k]+.35*np.mean(vs,axis=0)

class ContactDynamics:
    """Common sequential contact model, parameterized by the calibrated route."""
    W = np.array([1., 1.35, .075])
    predict_iou = staticmethod(predict_iou)

    def __init__(self, priors, yaw_cap_deg, travel_per_step, reserve):
        self.PRIORS = priors
        self.yaw_cap = np.deg2rad(yaw_cap_deg)
        self.travel_per_step = travel_per_step
        self.reserve = reserve

    def advance(self,q,k,d,rates,heading=None):
        q=np.asarray(q,float).copy(); v=np.asarray(rates[k],float)
        q[:2]+=rot(q[2] if heading is None else heading)@v[:2]*d
        q[2]+=v[2]*d
        return q

    def rollout(self,q,seq,d,rates,first_heading=None):
        q=np.asarray(q,float).copy(); states=[]
        for i,(k,s) in enumerate(zip(seq,d)):
            q=self.advance(q,k,s,rates,first_heading if i==0 else None); states.append(q.copy())
        return q,states

    def fit(self,q,seq,rates,upper,lower=None,first_heading=None,guess=None):
        """Optimize the actual sequential nonlinear pose, not a frozen Jacobian."""
        upper=np.asarray(upper,float); lower=np.zeros(len(seq)) if lower is None else np.asarray(lower,float)
        if np.any(upper<=lower): return None
        def fun(d):
            end,states=self.rollout(q,seq,d,rates,first_heading)
            excess=np.maximum(0.,np.abs([s[2] for s in states])-self.yaw_cap)
            return np.r_[self.W*end,.15*excess,.006*d]
        if guess is None:
            A=np.column_stack([np.r_[rot(q[2] if i or first_heading is None else first_heading)@rates[k][:2],rates[k][2]] for i,k in enumerate(seq)])
            guess=np.linalg.lstsq(self.W[:,None]*A,-self.W*np.asarray(q),rcond=None)[0]
        initial=np.clip(guess,lower+1e-9,upper-1e-9)
        result=least_squares(fun,initial,bounds=(lower,upper),max_nfev=35,ftol=1e-6,xtol=1e-6,gtol=1e-6)
        d=result.x
        # For the last contact, evaluate actual polygon IoU along the feasible line.
        if len(seq)==1:
            def obj(s):
                end,_=self.rollout(q,seq,[s],rates,first_heading)
                return -predict_iou(end)+max(0.,abs(end[2])-self.yaw_cap)*5.
            grid=np.linspace(lower[0],upper[0],25)
            vals=[obj(s) for s in grid]; j=int(np.argmin(vals)); candidates=[float(d[0]),float(grid[j])]
            lo,hi=grid[max(0,j-1)],grid[min(len(grid)-1,j+1)]
            if hi>lo: candidates.append(float(minimize_scalar(obj,bounds=(lo,hi),method='bounded',options={'xatol':.0001}).x))
            d=np.array([min(candidates,key=obj)])
        return d

    def assess(self,q,seq,d,rates,world_goal,target_yaw,eef,remaining):
        R=rot(target_yaw); tool=np.asarray(eef,float)[:2]; q=np.asarray(q,float).copy(); steps=0; max_yaw=abs(q[2]);
        for k,s in zip(seq,d):
            n=R@rot(q[2])@NORMALS[k]; point=world_goal+R@(q[:2]+rot(q[2])@POINTS[k]); approach=point-.024*n; end=point+(s+.018)*n
            if np.any(approach<[-.26,-.25]) or np.any(approach>[.21,.25]) or np.any(end<[-.29,-.28]) or np.any(end>[.25,.28]): return None
            steps+=int(np.ceil(40+np.linalg.norm(tool-approach)/.007+s/self.travel_per_step))
            q=self.advance(q,k,s,rates); max_yaw=max(max_yaw,abs(q[2])); tool=end
        if max_yaw>self.yaw_cap or steps+self.reserve>remaining: return None
        return q,steps,max_yaw

    def live_tail(self,q,seq,remaining_upper,rates,first_heading,guess):
        """Optimize remaining travel while first contact stays engaged.
        Its displacement is expressed in its frozen contact-start frame; later
        contacts use their predicted new frames. No extra contacts are introduced.
        """
        if not seq or remaining_upper<.00015: return None
        caps=[remaining_upper]+[CAPS[k] for k in seq[1:]]
        lo=[0.]+[.0025]*(len(seq)-1)
        return self.fit(q,seq,rates,caps,lo,first_heading=first_heading,guess=guess)

standard = ContactDynamics(standard_priors, 16., .0027, 4)
wide = ContactDynamics(wide_priors, 28., .004, 8)

class CalibratedPlanner:
    predict_iou=staticmethod(predict_iou)
    def __init__(self,route,model):
        self.base=wide if route=='wide' else standard
        self.route=route; self.data=model['routes'][route]; self.PRIORS={};self.variants={}
        for k,old in self.base.PRIORS.items():
            r=self.data[k]; old=np.asarray(old,float)
            if r['n']>=3:
                # modest shrinkage for small groups; preserve direction normalization
                w=r['n']/(r['n']+3); v=w*np.asarray(r['median'])+(1-w)*old
                samples=np.asarray(r['samples']); cross_axis=1 if k[0]=='x' else 0
                # Real joint vectors spanning typical variation in yaw AND cross drift.
                ids=set()
                for axis in (cross_axis,2):
                    order=np.argsort(samples[:,axis]);ids.update(order[int(round(q*(len(order)-1)))] for q in (.15,.50,.85))
                self.variants[k]=[samples[i] for i in sorted(ids)]
            else:
                v=old;self.variants[k]=[old]
            self.PRIORS[k]=v
    def rollout(self,*a,**kw):return self.base.rollout(*a,**kw)
    def live_tail(self,*a,**kw):return self.base.live_tail(*a,**kw)
    def solve(self,pose,world_goal,target_yaw,eef,remaining_steps,contacts_left,rates=None,blocked=()):
        pose=np.asarray(pose,float);rates={**self.PRIORS,**(rates or {})};current=predict_iou(pose)
        names=[k for k in self.PRIORS if k not in blocked];best=None
        if remaining_steps<25 or contacts_left<=0:return None
        for count in range(1,min(contacts_left,3)+1):
            if remaining_steps<40*count:continue
            for seq in product(names,repeat=count):
                if any(a==b for a,b in zip(seq,seq[1:])):continue
                d=self.base.fit(pose,seq,rates,[CAPS[k] for k in seq],np.full(count,.0025))
                if d is None or (count>1 and np.any(d<.003)):continue
                info=self.base.assess(pose,seq,d,rates,np.asarray(world_goal),target_yaw,eef,remaining_steps)
                if info is None:continue
                q,steps,max_yaw=info;iou=predict_iou(q)
                if iou<=current+.002:continue
                # Last contact uncertainty cannot be corrected by a subsequent stroke.
                before=self.rollout(pose,seq[:-1],d[:-1],rates)[0];k=seq[-1];scores=[]
                for v in [*self.variants[k],rates[k]]:
                    direction=np.r_[rot(before[2])@v[:2],v[2]];weighted=self.base.W*direction
                    dist=float(np.clip(-(self.base.W*before)@weighted/max(weighted@weighted,1e-12),0.,CAPS[k]))
                    scores.append(predict_iou(before+direction*dist))
                robust=float(np.quantile(scores,.2))
                # A missed initial contact costs an entire attempt. Soft penalty,
                # not a seed lookup or a blanket ban on otherwise viable sequences.
                r=self.data[seq[0]]; miss=r['first_no_motion']/max(1,r['first_attempts']+3)
                # Penalize large uncertain intermediate yaw and terminal cross drift.
                score=1-iou+.60*max(0.,iou-robust)+.00025*steps+.007*count+.12*max_yaw+.15*abs(q[2])+.18*miss
                if best is None or score<best.score:
                    best=Plan(tuple(seq),d,iou,steps,q,float(score));best.variant_min_iou=robust
        return best

def load_model(path=None):
    path=Path(path) if path else Path(__file__).resolve().parents[1] / 'config' / 'contact_response_model.json'
    model=json.loads(path.read_text(encoding='utf-8'))
    if model.get('version')!=1:raise ValueError('Unsupported contact response model')
    return model


class OrderedPlanner(CalibratedPlanner):
    def _search(self,pose,world_goal,target_yaw,eef,remaining,seqs,rates):
        current=predict_iou(pose); best=None
        for seq in seqs:
            count=len(seq)
            if remaining<40*count: continue
            d=self.base.fit(pose,seq,rates,[CAPS[k] for k in seq],np.full(count,.0025))
            if d is None or (count>1 and np.any(d<.003)):continue
            info=self.base.assess(pose,seq,d,rates,np.asarray(world_goal),target_yaw,eef,remaining)
            if info is None:continue
            q,steps,max_yaw=info;iou=predict_iou(q)
            if iou<=current+.002:continue
            before=self.rollout(pose,seq[:-1],d[:-1],rates)[0];k=seq[-1];scores=[]
            for v in [*self.variants[k],rates[k]]:
                direction=np.r_[rot(before[2])@v[:2],v[2]];weighted=self.base.W*direction
                dist=float(np.clip(-(self.base.W*before)@weighted/max(weighted@weighted,1e-12),0.,CAPS[k]))
                scores.append(predict_iou(before+direction*dist))
            robust=float(np.quantile(scores,.2))
            r=self.data[seq[0]];miss=r['first_no_motion']/max(1,r['first_attempts']+3)
            score=1-iou+.60*max(0.,iou-robust)+.00025*steps+.007*count+.12*max_yaw+.15*abs(q[2])+.18*miss
            if best is None or score<best.score:
                best=Plan(tuple(seq),d,iou,steps,q,float(score));best.variant_min_iou=robust
        return best

    def solve(self,pose,world_goal,target_yaw,eef,remaining_steps,contacts_left,rates=None,blocked=()):
        q=np.asarray(pose,float); r={**self.PRIORS,**(rates or {})}; blocked=set(blocked)
        # Only select X-first at the beginning, when both axes need material
        # travel, and the initial rotation really has aligned the object.
        # A Y pair can correct X cross-drift; the reverse order cannot.
        if contacts_left==3 and abs(q[0])>=.035 and abs(q[1])>=.012 and abs(q[2])<=np.deg2rad(4):
            x='xp_center' if q[0]<0 else 'xn_center'
            prefix='yp' if q[1]<0 else 'yn'; bar=prefix+'_bar_near'; stem=prefix+'_stem_mid'
            seqs=[(x,bar,stem),(x,stem,bar)]
            seqs=[s for s in seqs if not blocked.intersection(s)]
            p=self._search(q,world_goal,target_yaw,eef,remaining_steps,seqs,r)
            # Do not force an obviously bad partial trajectory. For imperfect
            # priors, retain a feasible high-overlap alternative if available.
            if p is not None and p.predicted_iou>=.80:
                p.order_policy='x_first';return p
        p=super().solve(q,world_goal,target_yaw,eef,remaining_steps,contacts_left,rates,blocked)
        if p is not None:p.order_policy='unrestricted'
        return p


def first_cap(q, name, goal, yaw):
    """Bound frozen first-stroke endpoint by the inherited reach envelope."""
    R=rot(yaw); n=R@rot(q[2])@NORMALS[name]
    point=np.asarray(goal)+R@(q[:2]+rot(q[2])@POINTS[name])
    approach=point-.024*n
    if np.any(approach<[-.26,-.25]) or np.any(approach>[.21,.25]): return 0.
    start=point+.018*n; low=np.array([-.29,-.28]); high=np.array([.25,.28])
    if np.any(start<low) or np.any(start>high): return 0.
    upper=CAPS[name]
    for i in range(2):
        if n[i]>1e-9: upper=min(upper,(high[i]-start[i])/n[i])
        elif n[i]<-1e-9: upper=min(upper,(low[i]-start[i])/n[i])
    return max(0.,float(upper))

class StagedPlanner(OrderedPlanner):
    def _candidate(self,q,goal,yaw,eef,remaining,seq,rates):
        count=len(seq)
        if remaining<40*count: return None
        caps=[CAPS[k] for k in seq]; caps[0]=first_cap(q,seq[0],goal,yaw)
        if caps[0]<=.003: return None
        d=self.base.fit(q,seq,rates,caps,np.full(count,.0025))
        if d is None or (count>1 and np.any(d<.003)):return None
        info=self.base.assess(q,seq,d,rates,np.asarray(goal),yaw,eef,remaining)
        if info is None:return None
        end,steps,max_yaw=info; iou=predict_iou(end)
        if iou<=predict_iou(q)+.002:return None
        before=self.rollout(q,seq[:-1],d[:-1],rates)[0]; name=seq[-1]; scores=[]
        for v in [*self.variants[name],rates[name]]:
            direction=np.r_[rot(before[2])@v[:2],v[2]]; weighted=self.base.W*direction
            cap=caps[0] if count==1 else CAPS[name]
            dist=float(np.clip(-(self.base.W*before)@weighted/max(weighted@weighted,1e-12),0.,cap))
            scores.append(predict_iou(before+direction*dist))
        robust=float(np.quantile(scores,.2)); r=self.data[seq[0]]
        miss=r['first_no_motion']/max(1,r['first_attempts']+3)
        score=1-iou+.60*max(0.,iou-robust)+.00025*steps+.007*count+.12*max_yaw+.15*abs(end[2])+.18*miss
        p=Plan(tuple(seq),d,iou,steps,end,float(score));p.variant_min_iou=robust
        return p

    def _best(self,q,goal,yaw,eef,remaining,seqs,rates):
        best=None
        for seq in seqs:
            p=self._candidate(q,goal,yaw,eef,remaining,seq,rates)
            if p is not None and (best is None or p.score<best.score):best=p
        return best

    def _partial_axis(self,q,goal,yaw,eef,remaining,names,axis,rates):
        # IoU can remain identically zero while the orthogonal axis is far
        # away. A necessary main stroke is still useful in that situation.
        best=None
        for name in names:
            direction=np.r_[rot(q[2])@rates[name][:2],rates[name][2]]
            if direction[axis]*q[axis]>=0 or abs(direction[axis])<.5:continue
            upper=min(first_cap(q,name,goal,yaw),abs(q[axis]/direction[axis]))
            for frac in (1.,.85,.70,.55,.40):
                d=upper*frac
                if d<.003:continue
                info=self.base.assess(q,(name,),np.array([d]),rates,np.asarray(goal),yaw,eef,remaining)
                if info is None:continue
                end,steps,max_yaw=info
                improvement=abs(q[axis])-abs(end[axis])
                if improvement<.001:continue
                score=-improvement+.30*abs(end[1-axis])+.04*abs(end[2])+.00005*steps
                p=Plan((name,),np.array([d]),predict_iou(end),steps,end,float(score));p.variant_min_iou=p.predicted_iou
                if best is None or p.score<best.score:best=p
                break
        return best

    def solve(self,pose,world_goal,target_yaw,eef,remaining_steps,contacts_left,rates=None,blocked=()):
        q=np.asarray(pose,float); r={**self.PRIORS,**(rates or {})}; blocked=set(blocked)
        if contacts_left<=0 or remaining_steps<40:return None
        names=[k for k in r if k not in blocked]; xs=[k for k in names if k.startswith('x')]; ys=[k for k in names if k.startswith('y')]
        x='xp_center' if q[0]<0 else 'xn_center'; prefix='yp' if q[1]<0 else 'yn'
        same_y=[k for k in ys if k.startswith(prefix)]
        # A negligible axis may be skipped. A materially needed X is never
        # silently replaced by Y just because a long-horizon IoU is low.
        if contacts_left==3 and abs(q[0])>.006:
            if x not in xs:return None
            seqs=[(x,a,b) for a in same_y for b in same_y if a!=b]
            # Preserve the ordered plan if feasible, but remove
            # the arbitrary predicted-IoU >= .80 gate.
            best=self._best(q,world_goal,target_yaw,eef,remaining_steps,seqs,r)
            if best is not None:
                best.order_policy='x_y_y';return best
            # If reach/time prevents a three-contact prediction, try X+Y, or
            # just the reachable X stroke. No Y-first fallback.
            seqs=[(x,a) for a in same_y]
            best=self._best(q,world_goal,target_yaw,eef,remaining_steps,seqs,r)
            if best is None:best=self._partial_axis(q,world_goal,target_yaw,eef,remaining_steps,[x],0,r)
            if best is not None:best.order_policy='x_partial_replan'
            return best
        if contacts_left>=2 and abs(q[1])>.004:
            # Main Y stroke is chosen jointly with final correction. The final
            # contact can be X, either Y face, or opposite Y if actually needed.
            seqs=[(a,b) for a in same_y for b in names if b!=a]
            best=self._best(q,world_goal,target_yaw,eef,remaining_steps,seqs,r)
            if best is None:best=self._partial_axis(q,world_goal,target_yaw,eef,remaining_steps,same_y,1,r)
            if best is not None:best.order_policy='y_then_trim'
            return best
        # No material Y remaining or final contact: choose best single action.
        best=self._best(q,world_goal,target_yaw,eef,remaining_steps,[(k,) for k in names],r)
        if best is not None:best.order_policy='final_trim'
        return best
