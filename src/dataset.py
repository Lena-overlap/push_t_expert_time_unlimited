"""Scene initialization, trajectory recording and optional video."""
from __future__ import annotations

import numpy as np
import json
from pathlib import Path
from types import SimpleNamespace


STEPS=400
BINS=((0,15),(15,30),(30,45),(45,60),(60,75),(75,90))


def cases(seed=5200, per_bin=10, angle_seed=20261124):
    """Uniform magnitudes within each stratum; exactly balanced signs."""
    if per_bin < 2 or per_bin % 2: raise ValueError('per-bin must be positive and even')
    rng=np.random.default_rng(angle_seed)
    result=[]
    for lo,hi in BINS:
        signs=np.tile([1.,-1.],per_bin//2); rng.shuffle(signs)
        # Avoid boundaries, so exact-reset tolerance never changes the route.
        for sign in signs:
            angle=float(round(sign*rng.uniform(lo+.10,hi-.10),4))
            result.append((seed+len(result),angle))
    return result


def band(angle):
    a=abs(float(angle))
    return next(f'{lo}-{hi}' for lo,hi in BINS if lo<=a<hi or (hi==90 and a==90))


def controller_args(args):
    return SimpleNamespace(rotation_model=args.rotation_model, close_steps=24,
        stop_margin_mm=2., max_buffer_mm=6., lookahead_steps=1.5,
        slowdown_start_mm=30., slow_command=.12, fast_command=.70,
        y_guard_limit_deg=4., y_guard_worsen_deg=1.25)


def make_expert(env,args):
    from baseline import StagedBaselineExpert
    from translation_controller import StagedTranslationController
    def factory(e):
        c=controller_args(args)
        return StagedTranslationController(e,response_model=args.response_model,model_path=c.rotation_model,face_model_path=None,
            gripper_change_steps=c.close_steps,feedback_stop_margin_m=c.stop_margin_mm/1000,
            feedback_max_buffer_m=c.max_buffer_mm/1000,feedback_lookahead_steps=c.lookahead_steps,
            feedback_slowdown_start_m=c.slowdown_start_mm/1000,feedback_slow_command=c.slow_command,
            feedback_fast_command=c.fast_command,feedback_yaw_guard_deg=0.,y_guard_enabled=True,
            y_guard_max_error_deg=c.y_guard_limit_deg,y_guard_worsening_deg=c.y_guard_worsen_deg,
            motion_evidence_enabled=True)
    return StagedBaselineExpert(env,factory,episode_steps=getattr(env,"expert_step_limit",STEPS),budget_steps=240)


def reseed(env, seed):
    np.random.seed(seed)
    if hasattr(env, "rng"):
        try:
            env.rng = np.random.default_rng(seed)
        except (AttributeError, TypeError):
            pass


def exact_yaw_reset(env, seed, signed_degrees, reseed):
    # First do the normal environment reset, including target and clutter.
    # Change ONLY the initial object's yaw, before executing any robot action.
    reseed(env,seed)
    env.reset()
    joint=env.cube.joints[0]
    qpos=np.asarray(env.sim.data.get_joint_qpos(joint),float).copy()
    if qpos.shape != (7,):
        raise ValueError('Expected a Tee free joint with 7 position coordinates')
    yaw=float(env.target_yaw)-np.deg2rad(signed_degrees)
    qpos[3:]=[np.cos(yaw/2),0,0,np.sin(yaw/2)]
    env.sim.data.set_joint_qpos(joint,qpos)
    env.sim.data.set_joint_qvel(joint,np.zeros(6))
    env.sim.forward()
    # Robosuite's observations must be refreshed after changing simulator pose.
    if hasattr(env,'_update_observables'):
        env._update_observables(force=True)
    return env._get_observations(force_update=True)


def make_env(record, max_steps=STEPS):
    from environment import PandaPushTRandom, default_metadata
    kwargs=dict(default_metadata()['env_kwargs'])
    kwargs.update(horizon=max_steps, ignore_done=True, has_renderer=False,
                  has_offscreen_renderer=record, use_camera_obs=False, hard_reset=False)
    env=PandaPushTRandom(**kwargs)
    env.expert_step_limit=int(max_steps)
    return env


def stage_phase(expert):
    d=expert.delegate
    if expert.repair is not None: return 'closed_repair',expert.repair.phase
    if expert.pending_repair is not None: return 'repair_disengage','retreat/lift'
    if d is not None:
        return ('small_'+str(d.mode) if not expert.handoff else 'translation'),str(d.phase)
    if expert.rotator is not None: return 'initial_rotation',expert.rotator.phase
    return 'initial','reset'


def annotate(frame, label, step, m, stage, phase, success, max_steps=STEPS):
    from PIL import Image,ImageDraw,ImageFont
    im=Image.fromarray(np.asarray(frame,dtype=np.uint8)).convert('RGB')
    draw=ImageDraw.Draw(im)
    try: font=ImageFont.truetype('DejaVuSans.ttf',16)
    except OSError: font=ImageFont.load_default()
    draw.rectangle((0,0,im.width,79),fill=(22,28,36))
    lines=(f'{label} | step {step:03d}/{max_steps} | {step/20:.2f}s',
           f'IoU {m["iou"]:.3f} | yaw {m["yaw_error_degrees"]:.2f} deg | pos {1000*m["position_error"]:.1f} mm',
           f'{stage} / {phase}' + (' | SUCCESS' if success else ''))
    for i,line in enumerate(lines):
        draw.text((8,5+i*24),line,font=font,fill=(135,240,167) if i==2 and success else (245,245,245))
    return np.asarray(im)


class Video:
    def __init__(self, path, env, args):
        import imageio.v2 as imageio
        self.path=Path(path); self.env=env; self.args=args
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.writer=imageio.get_writer(str(path),fps=20,codec='libx264',pixelformat='yuv420p',macro_block_size=16)
        self.frames=0
    def append(self, label,step,m,stage,phase,ok):
        a=self.args
        frame=self.env.sim.render(camera_name=a.camera,width=a.width,height=a.height,depth=False)
        # robosuite MuJoCo offscreen readback is bottom-up.
        frame=np.asarray(frame)[::-1].copy()
        if frame.shape!=(a.height,a.width,3): raise ValueError(f'Unexpected camera image shape {frame.shape}')
        self.writer.append_data(annotate(frame,label,step,m,stage,phase,ok,getattr(a,"max_steps",STEPS))); self.frames+=1
    def close(self):
        self.writer.close()


class DemoRecorder:
    def __init__(self):
        self.obs=[]; self.actions=[]; self.phases=[]; self.success=[]; self.ious=[]; self.target=[]
    def observe(self,obs,env,metrics,ok):
        # Exact environment observation keys, separate from privileged task
        # geometry; downstream policy decides which observations to use.
        values={}
        for k,v in obs.items():
            a=np.asarray(v)
            if a.dtype.kind in 'biuf':values[k]=a.copy()
        self.obs.append(values)
        self.target.append(np.r_[np.asarray(env.target_xy,float),float(env.target_yaw)])
        self.success.append(bool(ok)); self.ious.append(float(metrics['iou']))
    def action(self,a,stage,phase):
        a=np.asarray(a,float)
        if a.shape!=(7,) or not np.isfinite(a).all():raise ValueError('Invalid demonstration action')
        self.actions.append(a.copy()); self.phases.append((str(stage),str(phase)))
    def stats(self):
        a=np.asarray(self.actions); s=np.asarray(self.success,bool)
        diff=np.linalg.norm(np.diff(a[:,:6],axis=0),axis=1) if len(a)>1 else np.zeros(1)
        streak=0
        for ok in s[1:][::-1]:
            if not ok:break
            streak+=1
        # Prefix ending with the first 20 steps of the *final* stable run;
        # never truncate at a transient early success.
        prefix=len(a)-streak+20 if streak>=20 else len(a)
        return dict(action_delta_mean=float(diff.mean()),action_delta_p95=float(np.quantile(diff,.95)),
                    action_delta_max=float(diff.max()),demo_prefix_steps=int(prefix),
                    idle_action_fraction=float(np.mean(np.linalg.norm(a[:,:6],axis=1)<.02)),
                    demo_final_streak=int(streak))
    def save(self,path,metadata):
        if len(self.obs)!=len(self.actions)+1:raise ValueError('Expected T+1 observations for T actions')
        keys=set(self.obs[0]);
        if any(set(o)!=keys for o in self.obs):raise ValueError('Observation key set changed')
        arrays={}
        for k in sorted(keys):
            a=np.stack([o[k] for o in self.obs]);
            if not np.isfinite(a).all():raise ValueError('Nonfinite observation: '+k)
            arrays['obs/'+k]=a
        arrays.update(actions=np.asarray(self.actions,dtype=np.float32),stage=np.array([p[0] for p in self.phases]),phase=np.array([p[1] for p in self.phases]),success=np.array(self.success),iou=np.array(self.ious),task_target_xy_yaw=np.array(self.target),metadata_json=np.array(json.dumps(metadata,ensure_ascii=False)))
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,**arrays);tmp.replace(path)
        return str(path)
