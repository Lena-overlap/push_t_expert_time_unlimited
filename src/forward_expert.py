"""Forward-only expert: staged control, contact feedback, and real episode execution."""
from __future__ import annotations
from dataclasses import dataclass, field
from types import SimpleNamespace
import math
import numpy as np
from scipy.spatial.transform import Rotation
from dataset import DemoRecorder, make_expert, stage_phase
from geometry import FACE_BY_NAME, rot, CAPS
from contact_planner import CalibratedPlanner
from joint_lookahead import Lattice, Proposal, augmented, loss, terminal_distance

EXTRA={'xp_upper':('xp_center',-3.0),'xp_lower':('xp_center',3.0),
       'xn_upper':('xn_center',3.0),'xn_lower':('xn_center',-3.0)}
LOW=np.array([-.29,-.28]); HIGH=np.array([.25,.28])

def wrapped(a): return float(np.arctan2(np.sin(a),np.cos(a)))


def valid(m): return bool(m['upright'] and m['on_table'])


def ok(m,env): return valid(m) and m['iou']>=env.overlap_threshold


def pose(env):
    d=env.sim.data
    xy=d.body_xpos[env.cube_body_id,:2].copy()
    R=d.body_xmat[env.cube_body_id].reshape(3,3)
    yaw=math.atan2(R[1,0],R[0,0]); goal=d.body_xpos[env.target_body_id,:2].copy()
    return xy,yaw,goal,np.r_[rot(-env.target_yaw)@(xy-goal),wrapped(yaw-env.target_yaw)]


def signature(env,obs):
    d=env.sim.data
    # Includes robot, T and clutter positions/velocities and the target.
    return np.r_[d.qpos,d.qvel,d.body_xpos[env.target_body_id],
                 d.body_xmat[env.target_body_id],obs['robot0_eef_pos']].copy()


@dataclass
class EpisodeRecord:
    actions: list
    tags: list
    sig: np.ndarray
    q: np.ndarray
    eef: np.ndarray
    metrics: dict
    streak: int=0
    closed: bool=False
    history: tuple=()
    rates: dict=field(default_factory=dict)
    trace: list=field(default_factory=list)
    @property
    def steps(self): return len(self.actions)


class ReplayMismatch(RuntimeError): pass


class Runner:
    def __init__(self,env,reset,stable,max_steps=1000):
        self.env=env; self.reset=reset; self.stable=stable; self.limit=int(max_steps); self.recording=None; self.best_forward=None
        self.executed=0
    def start(self):
        self.obs=self.reset(); self.actions=[]; self.tags=[]; self.trace=[]; self.streak=0
        self.m=self.env.task_metrics()
        self.initial_sig=signature(self.env,self.obs)
        return self.node(False)
    def node(self,closed,history=(),rates=None):
        return EpisodeRecord(self.actions.copy(),self.tags.copy(),signature(self.env,self.obs),pose(self.env)[3],
                    np.asarray(self.obs['robot0_eef_pos']).copy(),dict(self.m),self.streak,closed,history,dict(rates or {}),self.trace.copy())
    def step(self,a,tag):
        if len(self.actions)>=self.limit: return False
        a=np.asarray(a,float)
        if a.shape!=(7,) or not np.isfinite(a).all() or np.max(np.abs(a))>1.00001:
            raise ValueError('Invalid 7D OSC action')
        self.obs,_,_,_=self.env.step(a.copy()); self.executed+=1
        self.actions.append(a.copy()); self.tags.append(tag)
        self.m=self.env.task_metrics(); self.streak=self.streak+1 if ok(self.m,self.env) else 0
        xy,yaw,_,q=pose(self.env); eef=self.obs['robot0_eef_pos']
        self.trace.append(dict(step=len(self.actions),stage=tag[0],phase=tag[1],iou=self.m['iou'],success=ok(self.m,self.env),success_streak=self.streak,object_x=xy[0],object_y=xy[1],object_yaw_rad=yaw,error_x_m=q[0],error_y_m=q[1],error_yaw_deg=np.rad2deg(q[2]),eef_x=eef[0],eef_y=eef[1],eef_z=eef[2],**{f'action_{i}':float(x) for i,x in enumerate(a)}))
        if self.recording is not None:
            self.recording.action(a,*tag); self.recording.observe(self.obs,self.env,self.m,ok(self.m,self.env))
            if valid(self.m) and (self.best_forward is None or self.m["iou"]>self.best_forward.metrics["iou"]):
                self.best_forward=self.node(False)
        return True
    def reached(self):
        return self.streak>=self.stable and self.obs['robot0_eef_pos'][2]>=.895
    def replay_video(self,node,frame):
        init=self.start()
        if not np.allclose(init.sig,self.expected_initial,atol=2e-6,rtol=1e-6):
            raise ReplayMismatch('Reset did not reproduce initial robot/object/clutter state')
        if frame is not None: frame(0,self.m,("initial","reset"),ok(self.m,self.env))
        for a,tag in zip(node.actions,node.tags):
            self.step(a,tag)
            if frame is not None: frame(len(self.actions),self.m,tag,ok(self.m,self.env))
        if not np.allclose(signature(self.env,self.obs),node.sig,atol=2e-5,rtol=1e-5) or self.streak!=node.streak:
            raise ReplayMismatch('Action replay did not reproduce recorded episode state / success streak')
    def move(self,desired,wrist=None,cap=.70,grip=1.):
        eef=np.asarray(self.obs['robot0_eef_pos']); r=np.zeros(3)
        if wrist is not None:
            current=self.env.sim.data.site_xmat[self.env.robots[0].eef_site_id['right']].reshape(3,3)
            r=np.clip(Rotation.from_matrix(wrist@current.T).as_rotvec()/.5,-.5,.5)
        return np.r_[np.clip((np.asarray(desired)-eef)*18.,-cap,cap),r,grip]
    def release(self):
        # Same vertical release as staged baseline; stop only after stable original task success.
        while len(self.actions)<self.limit and ok(self.m,self.env) and not self.reached():
            eef=np.asarray(self.obs['robot0_eef_pos'])
            self.step(self.move(np.r_[eef[:2],.95]),('search','release'))
        return self.reached()
    def prepare(self,closed):
        # No lateral action until verified lifted. Gripper changes above the T.
        for _ in range(30):
            eef=np.asarray(self.obs['robot0_eef_pos'])
            if eef[2]>=.896: break
            if not self.step(self.move(np.r_[eef[:2],.91],grip=1. if closed else -1.),('search','lift')): return False
        if self.obs['robot0_eef_pos'][2]<.896: return False
        if not closed:
            for _ in range(24):
                if not self.step(np.r_[np.zeros(6),1.],('search','close')): return False
        return valid(self.m) and self.obs['robot0_eef_pos'][2]>=.885


def finish_contact(r):
    # Try the normal stable success path first. If success was transient while
    # lifting, keep the physically safe state available for the next correction.
    if r.release(): return True
    return valid(r.m) and r.prepare(True)


def push(r,p,faces):
    from geometry import tee_robot_contact
    from translation_controller import stop_buffer
    from rotation_controller import RepositioningPushTExpert
    env=r.env; f=faces[p.face]; xy,yaw,_,_=pose(env); n=rot(yaw)@f.normal
    point=xy+rot(yaw)@f.xy; approach=point-.024*n
    if np.any(approach<[-.26,-.25]) or np.any(approach>[.21,.25]): return False
    wrist=RepositioningPushTExpert.wrist_rotation(SimpleNamespace(environment=env),n)
    # Approach and descend with original gains, tolerances, and bounded phase duration.
    for name,z,timeout,tol in [('approach',.90,45,.010),('descend',.824,30,.007)]:
        goal=np.r_[approach,z]; done=False
        for _ in range(timeout):
            eef=np.asarray(r.obs['robot0_eef_pos']); a=r.move(goal,wrist)
            if np.linalg.norm(eef-goal)<tol and (name!='approach' or np.linalg.norm(a[3:6])<.25):
                done=True; break
            if name=='approach' and eef[2]<.885: return False
            if not valid(r.m) or not r.step(a,('search',name)): return False
            if ok(r.m,env): return finish_contact(r)
        if not done: return False
    origin=xy.copy(); eef0=np.asarray(r.obs['robot0_eef_pos'])[:2].copy(); last=0.; stalls=0; seek=0; acquired=False; anchor=None
    target_distance=p.distance; samples=[]; last_adapt=-1.; max_distance=min(CAPS.get(p.face,.075),p.distance+.015)
    # Frozen stroke direction for measured distance; material tracking only after verified contact.
    for _ in range(84):
        cur,ang,_,_=pose(env); eef=np.asarray(r.obs['robot0_eef_pos']); motion=float((cur-origin)@n)
        prog=max(0.,motion-last); last=motion
        if ok(r.m,env): return finish_contact(r)
        if not valid(r.m): return False
        # Terminal strokes estimate the local SE(2) response from *actual* motion,
        # and optimize the stopping distance against polygon IoU. Do not change
        # contact or command direction merely because the prior was inaccurate.
        q=pose(env)[3]; samples.append((motion,q.copy()))
        if p.mode=='trim' and motion>=.003 and motion-last_adapt>=.002 and len(samples)>=3 and np.linalg.norm(q[:2])<.035 and abs(q[2])<.65:
            prev=next((x for x in reversed(samples[:-1]) if motion-x[0]>=.003),None)
            if prev is not None:
                d=motion-prev[0]; v=(q-prev[1])/d;v[2]=wrapped(q[2]-prev[1][2])/d
                if np.linalg.norm(v[:2])<3 and abs(v[2])<30:
                    upper=min(.018,max(0.,max_distance-motion),max(0.,(r.limit-len(r.actions)-12)*.002))
                    planned=motion+terminal_distance(q,v,upper)
                    target_distance=float(np.clip(planned,max(motion,target_distance-.008),max(motion,min(max_distance,target_distance+.008))))
                    last_adapt=motion
        gap=target_distance-motion; _,hit=tee_robot_contact(env)
        acquired |= bool(hit or motion>.0006)
        if hit and anchor is None: anchor=rot(-ang)@(eef[:2]-cur)
        if acquired: stalls=0 if prog>.00025 else stalls+1
        else: seek+=1
        if gap<=stop_buffer(prog,margin_m=.00065,lookahead_steps=1.2,maximum_m=.0035) or seek>=24 or stalls>=10: break
        if (eef[:2]-eef0)@n>target_distance+.035: break
        if abs(wrapped(ang-yaw))>np.deg2rad(38): break
        cap=float(np.clip(gap*20,.055,.70)) if acquired else .24
        cmd=n*cap; desired_wrist=wrist
        # Co-rotate with material face if turning. Do not chase slipped contact beyond envelope.
        if anchor is not None and abs(wrapped(ang-yaw))>np.deg2rad(4):
            live=rot(ang)@f.normal; t=np.array([-live[1],live[0]]); err=cur+rot(ang)@anchor-eef[:2]
            if live@n<.79 or abs(err@t)>.018 or abs(err@live)>.025: break
            cmd=live*np.clip(min(cap,.34)+14*(err@live),.035,.40)+t*np.clip(24*(err@t),-.20,.20)
            rz=np.eye(3); rz[:2,:2]=rot(wrapped(ang-yaw)); desired_wrist=rz@wrist
        if cmd@n<=0 or np.any(eef[:2]+.05*cmd<LOW) or np.any(eef[:2]+.05*cmd>HIGH): break
        a=r.move(np.r_[eef[:2],.824],desired_wrist,cap=.25); a[:2]=cmd
        if not r.step(a,('joint','trim' if p.mode=='trim' else 'push')): return False
    if ok(r.m,env): return finish_contact(r)
    # Disengage along the last face normal, then rise; require real clearance.
    eef=np.asarray(r.obs['robot0_eef_pos']); target=np.r_[np.clip(eef[:2]-.016*n,LOW,HIGH),eef[2]]
    for _ in range(18):
        if np.linalg.norm(np.asarray(r.obs['robot0_eef_pos'])-target)<.006: break
        if not valid(r.m) or not r.step(r.move(target),('search','retreat')): return False
        if ok(r.m,env): return finish_contact(r)
    return r.prepare(True)


def close_pose(a,b,xy=.004,yaw_deg=4.):
    return np.linalg.norm(a[:2]-b[:2])<=xy and abs(wrapped(a[2]-b[2]))<=np.deg2rad(yaw_deg)


@dataclass
class Attempt:
    start: np.ndarray
    end: np.ndarray
    face: str
    distance: float
    clear: bool
    elapsed: int

    @property
    def stalled(self):
        return close_pose(self.start,self.end,xy=.0015,yaw_deg=1.5)


class LocalMemory:
    """Conservative per-episode estimates and short cycle history."""
    def __init__(self,priors,faces):
        self.priors={k:np.asarray(v,float).copy() for k,v in priors.items()}
        self.rates={k:v.copy() for k,v in self.priors.items()}
        self.faces=faces; self.history=[]

    def record(self,start,end,p,clear,elapsed):
        entry=Attempt(np.array(start),np.array(end),p.face,p.distance,bool(clear),elapsed)
        self.history.append(entry)
        if not clear or p.face not in self.faces:return
        delta=rot(-start[2])@(end[:2]-start[:2]); travel=float(delta@self.faces[p.face].normal)
        if travel<.004:return
        measured=np.r_[delta/travel,wrapped(end[2]-start[2])/travel]
        if not np.isfinite(measured).all() or np.linalg.norm(measured[:2])>2.8 or abs(measured[2])>22:return
        w=min(.60,travel/(travel+.018))
        self.rates[p.face]=(1-w)*self.rates[p.face]+w*measured

    def recent(self,q,p):
        return [x for x in self.history[-16:] if x.face==p.face and close_pose(x.start,q)]

    def blocked(self,q,p):
        near=self.recent(q,p)
        # Failed approaches / zero response are a face-level warning only at
        # this local pose, not a permanent prohibition across the episode.
        if any(x.stalled and not x.clear for x in near):return True
        if sum(x.stalled for x in near)>=2:return True
        if any(abs(x.distance-p.distance)<.006 for x in near):return True
        return False

    def rank(self,q,p):
        v=self.rates[p.face]; pred=q+np.r_[rot(q[2])@v[:2],v[2]]*p.distance
        pred[2]=wrapped(pred[2]); visits=sum(close_pose(x.start,pred) for x in self.history[-12:])
        # Preserve lookahead's estimate, penalize returns to previously visited
        # configurations, and slightly prefer real immediate progress.
        return p.estimate+.010*min(visits,3)+.004*len(self.recent(q,p))+.10*float(loss(pred))

    def options(self,q,eef,remaining,goal,heading):
        opts=Lattice(self.rates,self.faces).plan(q,eef,remaining,goal,heading,limit=12,closed=True)
        return sorted((p for p in opts if not self.blocked(q,p)),key=lambda p:self.rank(q,p))


def safe_boundary(expert,obs):
    if float(obs['robot0_eef_pos'][2])<.896:return False
    if expert.repair is not None or expert.pending_repair is not None:return False
    d=expert.delegate
    if d is not None:
        return (d.phase in ('plan','four_lift','release') and d._four_active is None and d._four_planned is None)
    return expert.rotator is not None and expert.rotator.done


def baseline_exhausted(expert,obs):
    if not safe_boundary(expert,obs):return False
    d=expert.delegate
    if d is None:return bool(expert.handoff_blocked or (expert.rotator.done and not expert.rotator.completed))
    return bool(getattr(d,'joint_terminal',False) or (getattr(d,'large_route',False) and getattr(d,'joint_attempts',0)>=3) or getattr(d,'_v5_terminal',False))


def grip_closed(expert):
    if expert.delegate is not None:return expert.delegate.grip_mode=='closed'
    return bool(expert.rotator is not None and expert.rotator.closed)


def rotate_forward(r,kind):
    from rotation_controller import ArcRotation, ContinuousRotation
    remaining=r.limit-len(r.actions)
    if remaining<80:return False
    cls=ArcRotation if kind=='arc' else ContinuousRotation
    obj=cls(r.env,goal_deg=2.5,budget_steps=min(240 if kind=='arc' else 150,remaining),max_attempts=3 if kind=='arc' else 2)
    obj.closed=True; obj._phase('plan')
    while not obj.done and len(r.actions)<r.limit and valid(r.m):
        if ok(r.m,r.env):return r.release()
        if not r.step(obj.action(r.obs),('loop','rotate_'+obj.phase)):return False
    if ok(r.m,r.env):return r.release()
    return valid(r.m) and r.prepare(True)


def run_forward(r,args,angle,model):
    init=r.start(); r.expected_initial=init.sig.copy()
    recorder=getattr(r, "recorder_factory", DemoRecorder)(); recorder.observe(r.obs,r.env,r.m,ok(r.m,r.env))
    r.recording=recorder; r.best_forward=init
    expert=make_expert(r.env,args); idle=0; reason='step_cap'; log=[]
    # Preserve the original staged controller. The only differences from
    # staged-only evaluation are the configured horizon and early termination.
    while len(r.actions)<r.limit and valid(r.m) and not r.reached():
        if not ok(r.m,r.env) and baseline_exhausted(expert,r.obs):reason='staged_exhausted';break
        before=pose(r.env)[3]; eef=np.asarray(r.obs['robot0_eef_pos']).copy()
        cmd=expert.action(r.obs); r.step(cmd,stage_phase(expert))
        after=pose(r.env)[3]
        quiet=(np.linalg.norm(cmd[:6])<.025 and close_pose(before,after,.0002,.2) and np.linalg.norm(r.obs['robot0_eef_pos']-eef)<.0005)
        idle=idle+1 if quiet and safe_boundary(expert,r.obs) else 0
        if idle>=15 and not ok(r.m,r.env):reason='staged_idle';break
    final=r.node(grip_closed(expert)); win=final if r.reached() else None
    if win is not None:reason='success'
    elif not valid(r.m):reason='invalid_pose'
    elif len(r.actions)<r.limit:
        # Only leave the baseline at a verified disengaged boundary; do not
        # abandon a running contact or pretend that a partially closed gripper is closed.
        if not safe_boundary(expert,r.obs):reason='unsafe_handoff'
        elif not r.prepare(final.closed):reason='prepare_failed'
        else:
            priors=dict(CalibratedPlanner('wide' if abs(angle)>60.05 else 'standard',model).PRIORS)
            for k,(base,turn) in EXTRA.items():priors[k]=np.r_[priors[base][:2],turn]
            priors,faces=augmented(priors,FACE_BY_NAME); memory=LocalMemory(priors,faces)
            best_loss=float(loss(pose(r.env)[3])); best_iou=float(r.m['iou']); stagnant=0
            reason='step_cap'
            while r.limit-len(r.actions)>=35 and valid(r.m):
                if ok(r.m,r.env):r.release()
                if r.reached():break
                if not r.prepare(True):reason='prepare_failed';break
                if r.reached():break
                q=pose(r.env)[3]; yaw=abs(np.rad2deg(q[2])); remaining=r.limit-len(r.actions)
                # Large residual yaw is usually cheaper to correct with an
                # existing rotation primitive. If that stalls, try another
                # contact / translation instead of restarting the same FSM.
                rot_opts=[Proposal('rotation:'+k,0.,0.,'rotation') for k in (('arc','continuous') if yaw>60 else ('continuous','arc'))]
                rot_opts=[p for p in rot_opts if remaining>=80 and yaw>4.5 and not memory.blocked(q,p)]
                p=None
                if yaw>18 and rot_opts:p=rot_opts[0]
                else:
                    options=memory.options(q,r.obs['robot0_eef_pos'],remaining,r.env.target_xy,r.env.target_yaw)
                    if options:p=options[0]
                    if rot_opts and (not options or (yaw>6 and stagnant>=2)):p=rot_opts[0]
                if p is None:reason='no_new_contact';break
                t=len(r.actions); iou=float(r.m['iou'])
                clear=rotate_forward(r,p.face.split(':')[1]) if p.mode=='rotation' else push(r,p,faces)
                end=pose(r.env)[3]; elapsed=len(r.actions)-t
                memory.record(q,end,p,clear,elapsed)
                log.append(dict(trial=len(log)+1,mode=p.mode,face=p.face,distance_m=p.distance,parent_steps=t,steps=len(r.actions),elapsed=elapsed,parent_iou=iou,iou=r.m['iou'],yaw_deg=r.m['yaw_error_degrees'],position_m=r.m['position_error'],clear=clear,success=r.reached(),predicted_terminal_iou=p.predicted_iou,actual_yaw_change_deg=np.rad2deg(wrapped(end[2]-q[2]))))
                if r.reached():break
                if not valid(r.m):reason='invalid_pose';break
                if not clear and not r.prepare(True):reason='disengage_failed';break
                value=float(loss(pose(r.env)[3])); current_iou=float(r.m['iou'])
                improved=value<best_loss-.002 or current_iou>best_iou+.015
                best_loss=min(best_loss,value); best_iou=max(best_iou,current_iou)
                stagnant=0 if improved else stagnant+1
                if stagnant>=6:reason='no_progress_6_contacts';break
            # A successful pose near the protective cap can still settle if
            # enough steps remain; a new contact is not needed.
            if ok(r.m,r.env) and not r.reached():r.release()
            if r.reached():win=r.node(True,('forward_loop',));reason='success'
            elif not valid(r.m):reason='invalid_pose'
            elif r.limit-len(r.actions)<35 and reason=='step_cap':reason='insufficient_protection_budget'
    best=r.best_forward; r.recording=None
    return init,final,best,win,recorder if win is not None else None,log,reason
