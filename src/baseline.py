"""Staged expert orchestration and rotation-to-translation handoff."""
from __future__ import annotations

import numpy as np
from rotation_controller import EccentricRotation, HIGH_Z, ContinuousRotation, ArcRotation
from geometry import angular_pose, wrapped, tee_robot_contact


class LargeYawThenFineContact:
    """Stage a rotation before handing over to the translation controller.

    Small yaw errors go directly to translation; rotation handoff occurs
    at a lifted end-effector boundary.
    """
    def __init__(self, env, translation_factory, *, start_deg=15., **rotation_kwargs):
        self.env = env
        self.translation_factory = translation_factory
        self.start_deg = start_deg
        self.rotation_kwargs = rotation_kwargs
        self.reset()

    def reset(self):
        self.delegate = None
        self.rotator = None
        self.handoff = False
        self.handoff_step = None
        self.handoff_yaw_deg = None
        self.handoff_position_error_m = None

    def action(self, observation):
        if self.delegate is not None:
            return self.delegate.action(observation)
        _, yaw = angular_pose(self.env)
        angle = abs(np.rad2deg(wrapped(self.env.target_yaw-yaw)))
        if self.rotator is None and angle < self.start_deg:
            self.delegate = self.translation_factory(self.env)
            return self.delegate.action(observation)
        if self.rotator is None:
            self.rotator = EccentricRotation(self.env,**self.rotation_kwargs)
        if not self.rotator.done:
            return self.rotator.action(observation)
        eef = np.asarray(observation['robot0_eef_pos'],float)
        if eef[2]<.896:
            return self.rotator._move(eef,np.r_[eef[:2],HIGH_Z],gripper=1. if (self.rotator.closed or self.rotator.close_count) else -1.)
        self.delegate = self.translation_factory(self.env)
        # Fresh planner: no contact-specific open-gripper response is transferred.
        self.delegate.grip_mode = 'closed' if self.rotator.closed else 'open'
        self.delegate.mode = 'translate'
        self.delegate.rotate_completed = self.rotator.completed
        self.handoff = True
        self.handoff_step = int(getattr(self.env,'timestep',self.rotator.steps))
        m=self.env.task_metrics()
        self.handoff_yaw_deg=float(m['yaw_error_degrees'])
        self.handoff_position_error_m=float(m['position_error'])
        return self.delegate.action(observation)


class LargeYawWithRecovery(LargeYawThenFineContact):
    def __init__(self, env, translation_factory, *, episode_steps=400, correction_deg=4.,
                 near_m=.040, min_remaining=80, max_corrections=1, **kwargs):
        self.episode_steps=int(episode_steps)
        self.correction_deg=float(correction_deg)
        self.near_m=float(near_m)
        self.min_remaining=int(min_remaining)
        self.max_corrections=int(max_corrections)
        super().__init__(env,translation_factory,**kwargs)

    def reset(self):
        super().reset()
        self.repair=None
        self.repairs=[]
        self.repair_events=[]
        self.repair_start=None
        self._large_route_configured=False
        self.last_stage='initial'

    def _configure_delegate(self):
        if self.handoff and not self._large_route_configured:
            # Disable both translation re-entry paths.
            # Do not change rotation on the independent small-angle route.
            self.delegate.rotate_budget_steps=0
            self._large_route_configured=True

    def _safe_boundary(self, observation):
        d=self.delegate
        return (d.grip_mode=='closed' and d.mode=='translate'
                and d.phase in ('plan','four_lift','release')
                and d._four_active is None and d._four_planned is None
                and float(observation['robot0_eef_pos'][2])>=.896)

    def _should_repair(self, observation, metrics):
        if (not self.handoff or len(self.repairs)>=self.max_corrections
                or not self._safe_boundary(observation)):
            return False
        if not (metrics['upright'] and metrics['on_table']):
            return False
        if metrics['iou']>=self.env.overlap_threshold:
            return False
        remaining=self.episode_steps-int(self.env.timestep)
        yaw=metrics['yaw_error_degrees']; dist=metrics['position_error']
        return (remaining>=self.min_remaining and
                ((dist<=self.near_m and yaw>=self.correction_deg) or yaw>=8.))

    def _repair_action(self, observation):
        self.last_stage='closed_repair'
        r=self.repair
        if not r.done:
            return r.action(observation)
        eef=np.asarray(observation['robot0_eef_pos'],float)
        if eef[2]<.896:
            return r._move(eef,np.r_[eef[:2],.91])
        m=self.env.task_metrics()
        self.repair_events.append(dict(start_step=self.repair_start['step'],end_step=int(self.env.timestep),
            start_yaw_deg=self.repair_start['yaw'],end_yaw_deg=m['yaw_error_degrees'],
            start_position_m=self.repair_start['position'],end_position_m=m['position_error'],
            start_iou=self.repair_start['iou'],end_iou=m['iou'],completed=r.completed,exit_reason=r.exit_reason))
        self.repair=None
        d=self.delegate
        # The physical pose changed outside the translation FSM. Retain learned
        # response rates and recorded pushes, but discard stale contact blocks.
        d.phase='plan'; d.phase_steps=0
        d._four_blocked.clear()
        for k in d._four_stalls: d._four_stalls[k]=0
        d._fallback_pose=None; d._fallback_retries=0; d._v5_terminal=False
        d.last_contact=None; d.last_error=None
        self.last_stage='translation'
        return d.action(observation)

    def action(self, observation):
        if self.repair is not None:
            return self._repair_action(observation)
        if self.delegate is None:
            self.last_stage='initial_rotation'
            # Same initial rotator and exact handoff conditions as parent, but
            # configure the delegate BEFORE its first translation command.
            _, yaw=angular_pose(self.env)
            angle=abs(np.rad2deg(wrapped(self.env.target_yaw-yaw)))
            if self.rotator is None and angle<self.start_deg:
                self.delegate=self.translation_factory(self.env)
                self.last_stage='translation'
                return self.delegate.action(observation)
            if self.rotator is None:
                self.rotator=EccentricRotation(self.env,**self.rotation_kwargs)
            if not self.rotator.done:
                return self.rotator.action(observation)
            eef=np.asarray(observation['robot0_eef_pos'],float)
            if eef[2]<.896:
                return self.rotator._move(eef,np.r_[eef[:2],HIGH_Z],gripper=1. if (self.rotator.closed or self.rotator.close_count) else -1.)
            self.delegate=self.translation_factory(self.env)
            self.delegate.grip_mode='closed' if self.rotator.closed else 'open'
            self.delegate.mode='translate'
            self.delegate.rotate_completed=self.rotator.completed
            self.handoff=True
            self.handoff_step=int(getattr(self.env,'timestep',self.rotator.steps))
            m=self.env.task_metrics()
            self.handoff_yaw_deg=float(m['yaw_error_degrees'])
            self.handoff_position_error_m=float(m['position_error'])
            self._configure_delegate()
            self.last_stage='translation'
            return self.delegate.action(observation)
        self._configure_delegate()
        if self.handoff and self._should_repair(observation,self.env.task_metrics()):
            m=self.env.task_metrics(); remain=self.episode_steps-int(self.env.timestep)
            # Already closed and safely raised: do not spend another 24 steps
            # closing the fingers. Keep time for translation when possible.
            r=EccentricRotation(self.env,goal_deg=2.5,budget_steps=max(80,min(105,remain-20)),max_attempts=1)
            r.closed=True; r._phase('plan')
            self.repair=r; self.repairs.append(r)
            self.repair_start=dict(step=int(self.env.timestep),yaw=m['yaw_error_degrees'],position=m['position_error'],iou=m['iou'])
            return self._repair_action(observation)
        self.last_stage='translation'
        return self.delegate.action(observation)


class LatchedRotationRecovery(LargeYawWithRecovery):
    def __init__(self, env, translation_factory, **kwargs):
        kwargs.setdefault('correction_deg',3.5)
        kwargs.setdefault('near_m',.060)
        kwargs.setdefault('min_remaining',65)
        super().__init__(env,translation_factory,**kwargs)

    def reset(self):
        super().reset()
        self.pending_repair=None
        self.gate_events=[]
        self.gate_status='initial'
        self._last_gate_status=None
        self._retreat_target=None
        self._retreat_steps=0

    def _gate(self, status, m, obs):
        self.gate_status=status
        if status != self._last_gate_status:
            d=self.delegate
            self.gate_events.append(dict(step=int(self.env.timestep),status=status,
                phase=d.phase if d is not None else '',yaw_deg=m['yaw_error_degrees'],
                position_m=m['position_error'],iou=m['iou'],eef_z=float(obs['robot0_eef_pos'][2]),
                remaining=self.episode_steps-int(self.env.timestep)))
            self._last_gate_status=status

    def _request(self, obs, m):
        d=self.delegate; eef=np.asarray(obs['robot0_eef_pos'],float)
        self.pending_repair=dict(step=int(self.env.timestep),yaw=m['yaw_error_degrees'])
        self._retreat_target=None; self._retreat_steps=0
        # Own this disengagement boundary. The inherited four_lift advances
        # directly to approach at z>=.895, so waiting for PLAN/.896 is unsafe.
        p=d._four_active
        if p is not None:
            available,hit=tee_robot_contact(self.env)
            p['contact_api_available'] |= bool(available)
            p['contact_frames'] += int(hit)
            p['feedback_stop_reason']='closed_repair_request'
            d._four_log_push(eef)
        else:
            p=d._four_planned
        normal=None if p is None else np.asarray(p['normal'],float)
        if d.phase=='four_retreat' and d._four_post_target is not None:
            self._retreat_target=np.asarray(d._four_post_target,float).copy()
        elif eef[2]<.885 and normal is not None:
            self._retreat_target=np.r_[eef[:2]-d.retreat_m*normal,eef[2]]
        d._four_active=None; d._four_planned=None
        d._four_post_target=None; d._four_post_steps=0
        d.phase='plan'; d.phase_steps=0
        self._gate('requested',m,obs)

    def _motion(self, eef, target):
        return np.r_[np.clip((np.asarray(target)-eef)*18.,-.65,.65),np.zeros(3),1.]

    def _drain(self, obs, m):
        eef=np.asarray(obs['robot0_eef_pos'],float)
        self.last_stage='repair_disengage'
        if self._retreat_target is not None:
            if np.linalg.norm(eef-self._retreat_target)>=.006 and self._retreat_steps<12:
                self._retreat_steps+=1
                self._gate('retreating',m,obs)
                return self._motion(eef,self._retreat_target)
            self._retreat_target=None
        if eef[2]<.896:
            self._gate('lifting',m,obs)
            return self._motion(eef,np.r_[eef[:2],HIGH_Z])
        self.pending_repair=None
        remain=self.episode_steps-int(self.env.timestep)
        if (m['iou']>=self.env.overlap_threshold or not(m['upright'] and m['on_table'])
                or m['yaw_error_degrees']<self.correction_deg):
            self._gate('cancelled_pose_changed',m,obs)
        elif remain<50:
            self._gate('too_late_after_disengage',m,obs)
        else:
            # Same validated contact servo. Reserve its existing 20-step
            # disengage margin, with a budget no larger than actual time left.
            budget=min(105,remain)
            r=EccentricRotation(self.env,goal_deg=2.5,budget_steps=max(80,budget),max_attempts=1)
            r.budget_steps=budget
            r.closed=True; r._phase('plan')
            self.repair=r; self.repairs.append(r)
            self.repair_start=dict(step=int(self.env.timestep),yaw=m['yaw_error_degrees'],position=m['position_error'],iou=m['iou'])
            self._gate('started',m,obs)
            return self._repair_action(obs)
        self.last_stage='translation'
        return self.delegate.action(obs)

    def action(self, obs):
        if self.repair is not None:
            return self._repair_action(obs)
        if self.delegate is None or not self.handoff:
            return super().action(obs)
        self._configure_delegate()
        m=self.env.task_metrics(); d=self.delegate
        if self.pending_repair is not None:
            return self._drain(obs,m)
        remain=self.episode_steps-int(self.env.timestep)
        yaw=m['yaw_error_degrees']; dist=m['position_error']
        if len(self.repairs)>=self.max_corrections:
            status='correction_limit'
        elif not (m['upright'] and m['on_table']):
            status='invalid_pose'
        elif m['iou']>=self.env.overlap_threshold:
            status='already_success'
        elif not ((dist<=self.near_m and yaw>=self.correction_deg) or yaw>=8.):
            status='pose_below_trigger'
        elif d.grip_mode!='closed' or d.mode!='translate':
            status='not_closed_translation'
        elif remain<self.min_remaining:
            status='too_late_before_request'
        else:
            self._request(obs,m)
            return self._drain(obs,m)
        self._gate(status,m,obs)
        self.last_stage='translation'
        return d.action(obs)


# radians of object turn per metre of target-axis object motion.
# Bar rates and -Y stem rate are derived approximately from the two traces;
# +Y stem retains its empirical prior.


class BalancedTranslationRecovery(LatchedRotationRecovery):
    def __init__(self,env,translation_factory,**kwargs):
        kwargs.setdefault('near_m',.012)
        super().__init__(env,translation_factory,**kwargs)
    def _configure_delegate(self):
        super()._configure_delegate()
        if self.handoff: self.delegate.large_route=True
    def action(self,obs):
        # Do not interrupt an intentionally yaw-balanced pair mid-stroke;
        # if it exceeds the emergency bound, the delegate stops safely.
        d=self.delegate
        if self.handoff and d is not None and self.repair is None and self.pending_repair is None:
            p=d._four_active or d._four_planned or {}
            if p.get('balance_y') and d.phase in ('approach','descend','push','plan'):
                self._configure_delegate(); self.last_stage='translation'
                self._gate('balanced_y_in_progress',self.env.task_metrics(),obs)
                return d.action(obs)
        return super().action(obs)


class ContinuousRotationRecovery(BalancedTranslationRecovery):
    def action(self, obs):
        if self.delegate is None and self.rotator is None:
            _, yaw = angular_pose(self.env)
            err = abs(np.rad2deg(wrapped(self.env.target_yaw-yaw)))
            if err >= self.start_deg:
                self.rotator = ContinuousRotation(self.env, **self.rotation_kwargs)
        return super().action(obs)


class StagedBaselineExpert(ContinuousRotationRecovery):
    def __init__(self,env,factory,**kw):
        kw['max_corrections']=0 # strict three translation attempts, no hidden extra correction
        super().__init__(env,factory,**kw)

    def reset(self):
        self.arc_route=False
        self.handoff_blocked=False
        super().reset()

    def _configure_delegate(self):
        super()._configure_delegate()
        if self.handoff:
            self.delegate.wide_angle_route=self.arc_route
            self.handoff_blocked=False

    def action(self,obs):
        if self.delegate is None and self.rotator is None:
            _,yaw=angular_pose(self.env)
            err=abs(np.rad2deg(wrapped(self.env.target_yaw-yaw)))
            # Preserve the successful <=60-degree route byte for byte in its
            # actual planner/execution behavior. 0.05 deg reset tolerance.
            if err>60.05:
                self.arc_route=True
                self.rotator=ArcRotation(self.env,**self.rotation_kwargs)
        if self.arc_route and self.delegate is None and self.rotator.done:
            m=self.env.task_metrics()
            _,yaw=angular_pose(self.env)
            err=abs(np.rad2deg(wrapped(self.env.target_yaw-yaw)))
            # No translation on an unaligned/invalid pose. The rotator owns
            # its 240-step budget and all retry/disengagement logic.
            if not self.rotator.completed or err>self.rotator.goal_deg or not(m['upright'] and m['on_table']):
                self.handoff_blocked=True; self.last_stage='rotation_incomplete'
                eef=np.asarray(obs['robot0_eef_pos'],float)
                if eef[2]<.896:
                    return self.rotator._move(eef,np.r_[eef[:2],.91],gripper=1.)
                return np.r_[np.zeros(6),1.]
        return super().action(obs)
