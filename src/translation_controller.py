"""Closed-loop translation and material contact control."""
from __future__ import annotations

import numpy as np
from geometry import FACES, FACE_BY_NAME, DIRECTIONS, yaw_bin, wrapped, angular_pose, tee_robot_contact, read_face_model, rotation_2d, NORMALS, CAPS
from rotation_controller import RepositioningPushTExpert, CalibratedRotatePushTExpert, heading_bin
from scipy.spatial.transform import Rotation
from contact_planner import CalibratedPlanner, load_model, StagedPlanner


class FourWayPlanningMixin:
    """Insert translation-only four-face contact planning into parent Expert.

    Parent can be Release85PushTExpert (translation-only) or the calibrated
    rotate-first Expert. Rotation remains untouched; face controller only runs
    during TRANSLATE. Uses parent action for OSC approach/descend/push.
    """
    def __init__(self, environment, trial_environment=None, *,
                 face_model_path=None, axis_tolerance_m=.004,
                 max_stroke_m=.035, min_stroke_m=.005,
                 retreat_m=.016, allow_original_fallback=True, **kwargs):
        if trial_environment is not None:
            raise ValueError('Four-way controller requires trial_environment=None')
        if not 0 < min_stroke_m <= max_stroke_m:
            raise ValueError('Invalid stroke limits')
        if axis_tolerance_m <= 0 or retreat_m < 0:
            raise ValueError('Invalid axis tolerance / retreat')
        super().__init__(environment, trial_environment=None, **kwargs)
        self.face_model = read_face_model(face_model_path, environment)
        self.axis_tolerance_m = float(axis_tolerance_m)
        self.min_stroke_m = float(min_stroke_m)
        self.max_stroke_m = float(max_stroke_m)
        self.retreat_m = float(retreat_m)
        self.allow_original_fallback = bool(allow_original_fallback)
        self._reset_four_way_state()

    def _reset_four_way_state(self):
        self.four_way_plans = 0
        self.four_way_completed = 0
        self.four_way_failed_approaches = 0
        self.four_way_fallbacks = 0
        self.four_way_no_contact = 0
        self.four_way_bad_pushes = 0
        self.four_way_samples = []
        self._four_planned = None
        self._four_active = None
        self._four_blocked = set()
        self._four_stalls = {d: 0 for d in DIRECTIONS}
        self._four_response = {}
        self._four_post_target = None
        self._four_post_steps = 0
        self._four_last_normal = None

    def reset(self):
        parent_reset = getattr(super(), 'reset', None)
        if callable(parent_reset):
            parent_reset()
        else:
            RepositioningPushTExpert.__init__(self, self.environment, trial_environment=None)
        self._reset_four_way_state()

    def _four_local_error(self, position, target):
        return rotation_2d(float(self.environment.target_yaw)).T @ (target - position)

    def _four_direction(self, position, target):
        ex, ey = self._four_local_error(position, target)
        # Prefer X then Y, but switch rather than repeatedly retrying a
        # direction whose measured contact has stalled.
        if abs(ex) > self.axis_tolerance_m and self._four_stalls['+x' if ex > 0 else '-x'] < 3:
            return '+x' if ex > 0 else '-x', float(abs(ex))
        if abs(ey) > self.axis_tolerance_m and self._four_stalls['+y' if ey > 0 else '-y'] < 3:
            return '+y' if ey > 0 else '-y', float(abs(ey))
        if abs(ex) > self.axis_tolerance_m:
            return '+x' if ex > 0 else '-x', float(abs(ex))
        if abs(ey) > self.axis_tolerance_m:
            return '+y' if ey > 0 else '-y', float(abs(ey))
        return None, 0.0

    def _four_fallback(self, position, rotation, target, angle, eef):
        self.four_way_fallbacks += 1
        self._four_planned = None
        if not self.allow_original_fallback:
            return False
        # The original planner is called explicitly, NOT by recursion through
        # the calibrated rotation planner or this mixin.
        self.last_contact = None
        self.last_error = None
        return RepositioningPushTExpert.plan(self, position, rotation, target, angle, eef)

    def _plan_four_way(self, position, rotation, target, angle, eef):
        direction, remaining = self._four_direction(position, target)
        if direction is None:
            return self._four_fallback(position, rotation, target, angle, eef)
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        bin_id = yaw_bin(yaw, self.face_model['yaw_bins']) if self.face_model else None
        basis = rotation_2d(float(self.environment.target_yaw))
        axis_idx = 0 if 'x' in direction else 1
        forward = basis[:, axis_idx] * (1 if direction[0] == '+' else -1)
        old_error = self._four_local_error(position, target)
        options = []
        candidates = [f for f in FACES if f.direction == direction]
        for face in candidates:
            if face.name in self._four_blocked:
                continue
            point = position + rotation @ np.asarray(face.xy)
            normal = rotation @ np.asarray(face.normal)
            if float(normal @ forward) < .96:
                continue
            approach = point - .024 * normal
            if np.any(approach < [-.26, -.25]) or np.any(approach > [.21, .25]):
                continue
            empirical = self._four_response.get(face.name)
            record = None
            if self.face_model:
                record = next((r for r in self.face_model['entries']
                               if r['candidate'] == face.name and int(r['yaw_bin']) == bin_id), None)
                # Probe data present but this face was unreliable: do NOT use it.
                if record is None or not record['reliable']:
                    continue
            if empirical is not None:
                gain, yaw_rate, cross_gain = empirical
            elif record:
                gain, yaw_rate, cross_gain = (float(record['forward_gain']),
                                               float(record['yaw_rate_rad_per_m']),
                                               float(record['cross_gain']))
            else:
                # Geometry is a fallback preference, not a claim of physical contact.
                gain = .8
                arm = np.asarray(face.xy) - self.center_of_mass
                n = np.asarray(face.normal)
                yaw_rate = float((arm[0] * n[1] - arm[1] * n[0]) / .0024)
                cross_gain = 0.0
            if gain <= .10:
                continue
            desired = min(self.max_stroke_m, max(self.min_stroke_m, .85 * remaining / gain))
            for factor in (.65, 1.0):
                stroke = float(np.clip(desired * factor, self.min_stroke_m, self.max_stroke_m))
                predicted_progress = gain * stroke
                predicted_yaw = wrapped(angle - yaw_rate * stroke)
                # Favor task progress; penalize yaw drift / unwanted sideways movement.
                cost = (abs(remaining - predicted_progress)
                        + .085 * abs(predicted_yaw)
                        + .7 * abs(cross_gain * stroke)
                        + .007 * np.linalg.norm(eef[:2] - approach)
                        + (0 if (record or empirical is not None) else .003))
                options.append((float(cost), face, approach, point + stroke * normal,
                                stroke, normal, point, old_error.copy()))
        if not options:
            return self._four_fallback(position, rotation, target, angle, eef)
        options.sort(key=lambda v: v[0])
        _, face, self.approach, self.end, stroke, normal, point, err_before = options[0]
        self.rotation = self.wrist_rotation(normal)
        self._four_planned = dict(direction=direction, candidate=face.name,
                                  surface=face.surface, normal=normal.copy(),
                                  point=point.copy(), planned_stroke_m=stroke,
                                  target=target.copy(), error_before=err_before)
        self.last_contact = None
        self.last_error = None
        self.four_way_plans += 1
        return True

    def plan(self, position, rotation, target, angle, eef):
        if getattr(self, 'mode', 'translate') == 'rotate':
            # CalibratedRotatePushTExpert.plan() handles rotate unchanged.
            return super().plan(position, rotation, target, angle, eef)
        return self._plan_four_way(position, rotation, target, angle, eef)

    def _four_log_push(self, eef):
        pending = self._four_active
        if pending is None:
            return
        position, yaw = angular_pose(self.environment)
        d_world = position - pending['start_xy']
        d_local = rotation_2d(-pending['start_yaw']) @ d_world
        actual_axis = float(d_world @ pending['normal'])
        old_error = pending['error_before']
        new_error = self._four_local_error(position, pending['target'])
        axis_idx = 0 if 'x' in pending['direction'] else 1
        axis_progress = float(abs(old_error[axis_idx]) - abs(new_error[axis_idx]))
        turn = wrapped(yaw - pending['start_yaw'])
        travel = max(.003, min(float(pending['planned_stroke_m']) + .010,
                               float(max(0, (eef[:2] - pending['start_eef']) @ pending['normal']))))
        key = pending['candidate']
        contact_verified_or_unknown = (pending['contact_frames'] > 0
                                       or not pending['contact_api_available'])
        effective = axis_progress > .0008 and contact_verified_or_unknown
        moved = float(np.linalg.norm(d_world)) >= .001
        if moved and (pending['contact_frames'] > 0 or not pending['contact_api_available']):
            axis_gain = max(-1, min(2, actual_axis / travel))
            other_idx = 1 - axis_idx
            cross_gain = float((old_error[other_idx] - new_error[other_idx]) / travel)
            yaw_gain = float(np.clip(turn / travel, -45, 45))
            if key in self._four_response:
                g, y, c = self._four_response[key]
                axis_gain, yaw_gain, cross_gain = (
                    .6*g + .4*axis_gain, .6*y + .4*yaw_gain, .6*c + .4*cross_gain)
            self._four_response[key] = (axis_gain, yaw_gain, cross_gain)
        if not effective:
            self._four_stalls[pending['direction']] += 1
            self._four_blocked.add(key)
            self.four_way_bad_pushes += 1
        else:
            self._four_stalls[pending['direction']] = 0
            self._four_blocked.clear()
        if pending['contact_api_available'] and not pending['contact_frames']:
            self.four_way_no_contact += 1
        self.four_way_completed += 1
        self.four_way_samples.append(dict(
            direction=pending['direction'], candidate=key, surface=pending['surface'],
            planned_stroke_m=float(pending['planned_stroke_m']),
            eef_travel_m=float(travel), object_dx_local_m=float(d_local[0]),
            object_dy_local_m=float(d_local[1]),
            object_translation_m=float(np.linalg.norm(d_world)),
            actual_yaw_deg=float(np.rad2deg(turn)),
            axis_progress_m=axis_progress, useful=bool(effective),
            robot_tee_contact_frames=int(pending['contact_frames']),
            contact_api_available=bool(pending['contact_api_available']),
        ))
        self._four_last_normal = pending['normal'].copy()
        self._four_active = None
        self._four_planned = None

    @staticmethod
    def _direct_move(eef, desired):
        xyz = np.clip((np.asarray(desired) - np.asarray(eef)) * 18., -.7, .7)
        return np.r_[xyz, [0., 0., 0., -1.]]

    def action(self, observation):
        eef = np.asarray(observation['robot0_eef_pos'])
        metrics = self.environment.task_metrics()
        success = (metrics['iou'] >= self.environment.overlap_threshold
                   and metrics['upright'] and metrics['on_table'])
        if self._four_active is not None:
            available, contacting = tee_robot_contact(self.environment)
            self._four_active['contact_api_available'] |= bool(available)
            if self.phase == 'push' and contacting:
                self._four_active['contact_frames'] += 1
            if self.phase in ('plan', 'release') or success:
                self._four_log_push(eef)
                if not success and self.retreat_m > 0 and self._four_last_normal is not None:
                    self._four_post_target = np.r_[eef[:2] - self.retreat_m * self._four_last_normal,
                                                   float(eef[2])]
                    self._four_post_steps = 0
                    self.phase = 'four_retreat'
        # A face push must be physically disengaged before moving to a new face.
        if success:
            self._four_post_target = None
            self.phase = 'release'
            self.hold_steps += 1
            return self._direct_move(eef, np.r_[eef[:2], .95])
        if self.phase == 'four_retreat':
            self._four_post_steps += 1
            if (np.linalg.norm(eef - self._four_post_target) < .006
                    or self._four_post_steps >= 18):
                self.phase = 'four_lift'
                self._four_post_steps = 0
            else:
                return self._direct_move(eef, self._four_post_target)
        if self.phase == 'four_lift':
            self._four_post_steps += 1
            if eef[2] >= .895 or self._four_post_steps >= 24:
                self.phase = 'plan'
            else:
                return self._direct_move(eef, np.r_[eef[:2], .91])
        previous_phase = self.phase
        command = super().action(observation)
        if (self._four_planned is not None and self._four_active is None
                and previous_phase != 'push' and self.phase == 'push'):
            xy, yaw = angular_pose(self.environment)
            self._four_active = dict(
                **self._four_planned, start_xy=xy.copy(), start_yaw=yaw,
                start_eef=eef[:2].copy(), contact_frames=0,
                contact_api_available=False)
        if (self._four_planned is not None and self._four_active is None
                and previous_phase in ('approach', 'descend') and self.phase == 'plan'):
            self.four_way_failed_approaches += 1
            self._four_blocked.add(self._four_planned['candidate'])
            self._four_stalls[self._four_planned['direction']] += 1
            self._four_planned = None
        return command


class RotateThenFourDirectionExpert(FourWayPlanningMixin, CalibratedRotatePushTExpert):
    """Previously validated calibrated ROTATE, new four-direction TRANSLATE."""
    pass

class ClosedGripperFourWayMixin:
    """MRO wrapper around the original four-direction action/state machine.

    Parent owns face planning and contact feedback; this mixin only controls
    when gripper open/close commands are issued and guarantees that a change
    happens at safe height instead of inside a contact push.
    """

    def __init__(self, environment, trial_environment=None, *,
                 gripper_change_steps: int = 24, gripper_safe_z: float = 0.900,
                 **kwargs):
        if gripper_change_steps < 1:
            raise ValueError('gripper_change_steps must be >= 1')
        if gripper_safe_z < .885:
            raise ValueError('gripper_safe_z should clear the table and tee')
        # DO NOT load the old open-gripper face calibration by accident.
        if kwargs.get('face_model_path') is not None:
            raise ValueError('Prior face_model.json used an OPEN gripper; re-probe closed contacts before using any face model')
        super().__init__(environment, trial_environment=trial_environment, **kwargs)
        self.gripper_change_steps = int(gripper_change_steps)
        self.gripper_safe_z = float(gripper_safe_z)
        self._reset_closed_state()

    def _reset_closed_state(self):
        self.grip_mode = 'open'  # open, closing, closed, opening
        self.grip_change_progress = 0
        self.grip_close_events = 0
        self.grip_open_events = 0
        self.closed_translation_steps = 0

    def reset(self):
        super().reset()
        self._reset_closed_state()

    def _begin_change(self, target):
        if target == 'closed' and self.grip_mode in ('open', 'opening'):
            self.grip_mode = 'closing'
            self.grip_change_progress = 0
        elif target == 'open' and self.grip_mode in ('closed', 'closing'):
            self.grip_mode = 'opening'
            self.grip_change_progress = 0

    def _change_action(self, observation):
        """Pause parent controller without advancing its FSM until safe and actuated."""
        eef = np.asarray(observation['robot0_eef_pos'], dtype=float)
        closing = self.grip_mode == 'closing'
        gripper = 1.0 if closing else -1.0
        # For safety, do not close or open the fingers while the EEF is near T.
        if eef[2] < self.gripper_safe_z - .004:
            target = np.r_[eef[:2], self.gripper_safe_z + .016]
            xyz = np.clip((target - eef) * 18., -.65, .65)
            # Hold previous state while raising; avoid moving fingers near T.
            old_command = -1.0 if closing else +1.0
            return np.r_[xyz, 0., 0., 0., old_command]
        self.grip_change_progress += 1
        if self.grip_change_progress >= self.gripper_change_steps:
            if closing:
                self.grip_mode = 'closed'
                self.grip_close_events += 1
            else:
                self.grip_mode = 'open'
                self.grip_open_events += 1
            self.grip_change_progress = 0
            # The parent planner's approach timer must not include close time.
            if getattr(self, 'phase', None) == 'approach':
                self.phase_steps = 0
        return np.r_[0., 0., 0., 0., 0., 0., gripper]

    def _should_begin_closed(self):
        """Initial episode with no rotation, or existing translate mode."""
        if self.grip_mode != 'open':
            return False
        mode = getattr(self, 'mode', None)
        if mode == 'translate':
            return True
        if mode == 'rotate':
            return False
        # Initial (unset) mode: use the same start criterion as parent.
        _, _, _, error = self._pose()
        angle = abs(float(np.rad2deg(error)))
        return (angle <= self.rotate_start_deg or not self.rotate_budget_steps
                or not self.max_rotate_pushes)

    def action(self, observation):
        if self.grip_mode in ('closing', 'opening'):
            return self._change_action(observation)
        metrics = self.environment.task_metrics()
        success = (metrics['iou'] >= self.environment.overlap_threshold
                   and metrics['upright'] and metrics['on_table'])
        if self._should_begin_closed() and not success:
            self._begin_change('closed')
            return self._change_action(observation)

        # Parent may change mode at this very timestep (even during four_lift),
        # so inspect mode AFTER parent.action and intercept its outgoing motion.
        command = np.asarray(super().action(observation), dtype=float).copy()
        if command.shape != (7,):
            raise ValueError(f'Expected 7D Panda action, got {command.shape}')
        mode = getattr(self, 'mode', 'translate')

        if mode == 'rotate' and self.grip_mode == 'closed' and not success:
            self._begin_change('open')
            return self._change_action(observation)
        if mode == 'translate' and self.grip_mode == 'open' and not success:
            self._begin_change('closed')
            return self._change_action(observation)

        # Allow the gripper to close in TRANSLATE only; no changes to movement
        # axes, face selection, approach/descend/push strokes, or retreat/lift.
        if self.grip_mode == 'closed':
            command[-1] = 1.0
            if mode == 'translate':
                self.closed_translation_steps += 1
        else:
            command[-1] = -1.0
        return command


class RotateThenClosedFourDirectionExpert(ClosedGripperFourWayMixin,
                                         RotateThenFourDirectionExpert):
    """Calibrated rotation unchanged; four-direction translation uses closed fingers."""

def signed_axis_remaining(direction: str, object_xy, target_xy, target_yaw: float) -> float:
    """Signed remaining distance along the CURRENT stroke's intended T-local axis.

    Positive means the target is still ahead along the commanded push direction;
    zero/negative means that axis has reached or passed its target.
    This is deliberately independent of the object's own yaw drift.
    """
    if direction not in ('+x', '-x', '+y', '-y'):
        raise ValueError(f'Unknown four-way direction: {direction!r}')
    local = rotation_2d(float(target_yaw)).T @ (
        np.asarray(target_xy, dtype=float) - np.asarray(object_xy, dtype=float)
    )
    axis = 0 if direction[1] == 'x' else 1
    sign = 1.0 if direction[0] == '+' else -1.0
    return float(sign * local[axis])


def stop_buffer(progress_per_step_m: float, *, margin_m: float,
                lookahead_steps: float, maximum_m: float) -> float:
    """Short reaction buffer proportional to recent object motion, clipped.

    This is not a model of the robot's physical stopping distance; it only
    anticipates a small amount of movement during controller latency.
    """
    if margin_m < 0 or maximum_m < margin_m or lookahead_steps < 0:
        raise ValueError('Invalid feedback stop buffer configuration')
    return min(maximum_m, max(margin_m, max(0., progress_per_step_m) * lookahead_steps))


def speed_limit(remaining_m: float, *, slowdown_start_m: float,
                slow_command: float, fast_command: float) -> float:
    """Piecewise-linear normalized OSC translation cap near the goal."""
    if slowdown_start_m <= 0 or not 0 < slow_command <= fast_command <= 1:
        raise ValueError('Invalid OSC speed parameters')
    alpha = float(np.clip(max(0., remaining_m) / slowdown_start_m, 0., 1.))
    return float(slow_command + (fast_command - slow_command) * alpha)


class RotateThenClosedFeedbackExpert(RotateThenClosedFourDirectionExpert):
    """Old calibrated rotation + old four-face selection + position-servo stop."""

    def __init__(
        self, environment, trial_environment=None, *,
        feedback_stop_margin_m: float = 0.0020,
        feedback_max_buffer_m: float = 0.0060,
        feedback_lookahead_steps: float = 1.5,
        feedback_slowdown_start_m: float = 0.030,
        feedback_slow_command: float = 0.12,
        feedback_fast_command: float = 0.70,
        feedback_yaw_guard_deg: float = 0.0,
        **kwargs,
    ):
        if not 0 <= feedback_stop_margin_m <= feedback_max_buffer_m:
            raise ValueError('Require 0 <= feedback_stop_margin_m <= feedback_max_buffer_m')
        if feedback_lookahead_steps < 0:
            raise ValueError('feedback_lookahead_steps cannot be negative')
        if feedback_slowdown_start_m <= 0:
            raise ValueError('feedback_slowdown_start_m must be positive')
        if not 0 < feedback_slow_command <= feedback_fast_command <= 1.0:
            raise ValueError('Require 0 < feedback_slow_command <= feedback_fast_command <= 1')
        if feedback_yaw_guard_deg < 0:
            raise ValueError('feedback_yaw_guard_deg cannot be negative')
        super().__init__(environment, trial_environment=trial_environment, **kwargs)
        self.feedback_stop_margin_m = float(feedback_stop_margin_m)
        self.feedback_max_buffer_m = float(feedback_max_buffer_m)
        self.feedback_lookahead_steps = float(feedback_lookahead_steps)
        self.feedback_slowdown_start_m = float(feedback_slowdown_start_m)
        self.feedback_slow_command = float(feedback_slow_command)
        self.feedback_fast_command = float(feedback_fast_command)
        self.feedback_yaw_guard_deg = float(feedback_yaw_guard_deg)
        self._reset_feedback_state()

    def _reset_feedback_state(self):
        self.feedback_stops = 0
        self.feedback_goal_stops = 0
        self.feedback_yaw_stops = 0
        self.feedback_slow_steps = 0
        self.feedback_monitored_steps = 0
        self.feedback_pre_push_aborts = 0
        self._feedback_previous_xy = None
        self._feedback_active_id = None
        self._feedback_last_remaining = None

    def reset(self):
        super().reset()
        self._reset_feedback_state()

    def _monitored_stroke(self):
        """Return parent four-face stroke ONLY while closed and in translation."""
        if (getattr(self, 'grip_mode', None) != 'closed'
                or getattr(self, 'mode', None) != 'translate'
                or getattr(self, 'phase', None) != 'push'):
            return None
        return getattr(self, '_four_active', None)

    def _record_new_stroke(self, pending, object_xy):
        active_id = id(pending)
        if self._feedback_active_id != active_id:
            self._feedback_active_id = active_id
            self._feedback_previous_xy = np.asarray(object_xy, dtype=float).copy()
            self._feedback_last_remaining = None
            pending['feedback_stop_reason'] = 'normal'
            pending['feedback_remaining_at_stop_m'] = None
            pending['feedback_buffer_at_stop_m'] = None
            pending['feedback_slow_steps'] = 0
            pending['feedback_monitored_steps'] = 0

    def _four_log_push(self, eef):
        """Append feedback diagnostics to the parent's normal push sample."""
        pending = getattr(self, '_four_active', None)
        old_count = len(getattr(self, 'four_way_samples', []))
        super()._four_log_push(eef)
        if pending is not None and len(self.four_way_samples) > old_count:
            sample = self.four_way_samples[-1]
            for field in ('feedback_stop_reason', 'feedback_remaining_at_stop_m',
                          'feedback_buffer_at_stop_m', 'feedback_slow_steps',
                          'feedback_monitored_steps'):
                sample[field] = pending.get(field, None)
        self._feedback_previous_xy = None
        self._feedback_active_id = None
        self._feedback_last_remaining = None

    def action(self, observation):
        # Some object motion may occur during DESCEND as the closed fingers
        # approach the T; do not enter a new push if that motion has already
        # eliminated the requested axis error. Retreat without changing face.
        planned = getattr(self, '_four_planned', None)
        if (planned is not None and getattr(self, '_four_active', None) is None
                and getattr(self, 'grip_mode', None) == 'closed'
                and getattr(self, 'mode', None) == 'translate'
                and getattr(self, 'phase', None) in ('approach', 'descend')):
            position = np.asarray(
                self.environment.sim.data.body_xpos[self.environment.cube_body_id, :2],
                dtype=float,
            )
            early_remaining = signed_axis_remaining(
                planned['direction'], position, planned['target'],
                self.environment.target_yaw,
            )
            if early_remaining <= self.feedback_stop_margin_m:
                eef = np.asarray(observation['robot0_eef_pos'], dtype=float)
                normal = np.asarray(planned['normal'], dtype=float)
                self.feedback_pre_push_aborts += 1
                self._four_planned = None
                self.phase = 'four_retreat'
                self._four_post_steps = 0
                self._four_post_target = np.r_[
                    eef[:2] - self.retreat_m * normal, eef[2]
                ]
                return np.asarray(super().action(observation), dtype=float)

        pending = self._monitored_stroke()
        remaining = None
        if pending is not None:
            position = np.asarray(
                self.environment.sim.data.body_xpos[self.environment.cube_body_id, :2],
                dtype=float,
            ).copy()
            self._record_new_stroke(pending, position)
            remaining = signed_axis_remaining(
                pending['direction'], position, pending['target'],
                self.environment.target_yaw,
            )
            goal_basis = rotation_2d(float(self.environment.target_yaw))
            axis_idx = 0 if pending['direction'][1] == 'x' else 1
            sign = 1.0 if pending['direction'][0] == '+' else -1.0
            forward = sign * goal_basis[:, axis_idx]
            progress = max(0., float((position - self._feedback_previous_xy) @ forward))
            buffer_m = stop_buffer(
                progress, margin_m=self.feedback_stop_margin_m,
                lookahead_steps=self.feedback_lookahead_steps,
                maximum_m=self.feedback_max_buffer_m,
            )
            self._feedback_previous_xy = position.copy()
            self._feedback_last_remaining = remaining
            self.feedback_monitored_steps += 1
            pending['feedback_monitored_steps'] += 1

            # Optional emergency guard; disabled by default to avoid introducing
            # a new yaw-based decision into the verified rotation/translation FSM.
            yaw_exceeded = False
            if self.feedback_yaw_guard_deg > 0:
                mat = self.environment.sim.data.body_xmat[
                    self.environment.cube_body_id
                ].reshape(3, 3)
                yaw = np.arctan2(mat[1, 0], mat[0, 0])
                yaw_error = np.arctan2(
                    np.sin(self.environment.target_yaw - yaw),
                    np.cos(self.environment.target_yaw - yaw),
                )
                yaw_exceeded = abs(np.rad2deg(yaw_error)) > self.feedback_yaw_guard_deg

            if remaining <= buffer_m or yaw_exceeded:
                pending['feedback_stop_reason'] = 'yaw_guard' if yaw_exceeded else 'axis_goal'
                pending['feedback_remaining_at_stop_m'] = float(remaining)
                pending['feedback_buffer_at_stop_m'] = float(buffer_m)
                self.feedback_stops += 1
                if yaw_exceeded:
                    self.feedback_yaw_stops += 1
                else:
                    self.feedback_goal_stops += 1
                # FourWayPlanningMixin.action() sees 'plan', logs the stroke,
                # commands retreat away from this face, and subsequently lifts.
                # We do NOT jump to a different contact while the tee is touched.
                self.phase = 'plan'
                return np.asarray(super().action(observation), dtype=float)

        command = np.asarray(super().action(observation), dtype=float).copy()
        if (pending is not None and self._monitored_stroke() is pending
                and remaining is not None):
            cap = speed_limit(
                remaining,
                slowdown_start_m=self.feedback_slowdown_start_m,
                slow_command=self.feedback_slow_command,
                fast_command=self.feedback_fast_command,
            )
            # Uniform scaling preserves the world-frame XY push direction;
            # independent per-axis clipping could skew it at arbitrary T yaw.
            largest = float(np.max(np.abs(command[:2])))
            if largest > cap:
                command[:2] *= cap / largest
                self.feedback_slow_steps += 1
                pending['feedback_slow_steps'] += 1
        return command


def should_stop_y_push(start_error_deg: float, current_error_deg: float,
                       *, max_error_deg: float = 4.0,
                       worsening_deg: float = 1.25) -> bool:
    """Avoid stopping due to pre-existing yaw error alone.

    An already-large yaw error does not immediately prevent a Y translation;
    only NEW additional angular error that crosses the threshold does.
    """
    if max_error_deg <= 0 or worsening_deg <= 0:
        raise ValueError('Y guard thresholds must be positive')
    return (float(current_error_deg) >= float(max_error_deg)
            and float(current_error_deg) - float(start_error_deg)
            >= float(worsening_deg))


class YawGuardExpert(RotateThenClosedFeedbackExpert):
    """Translation controller with yaw guarding and contact-motion reliability checks."""

    def __init__(self, environment, trial_environment=None, *,
                 y_guard_enabled: bool = True,
                 y_guard_max_error_deg: float = 4.0,
                 y_guard_worsening_deg: float = 1.25,
                 motion_evidence_enabled: bool = True,
                 motion_evidence_min_m: float = .001,
                 motion_evidence_progress_m: float = .0008,
                 **kwargs):
        if y_guard_max_error_deg <= 0 or y_guard_worsening_deg <= 0:
            raise ValueError('Y guard thresholds must be positive')
        if motion_evidence_min_m <= 0 or motion_evidence_progress_m <= 0:
            raise ValueError('Motion evidence thresholds must be positive')
        # Disable the global yaw guard: this guard is Y-only and based on
        # ADDITIONAL angular error, not merely the absolute angle.
        if float(kwargs.get('feedback_yaw_guard_deg', 0.0)) != 0.0:
            raise ValueError('Do not enable the global yaw guard together with the Y-axis guard')
        kwargs['feedback_yaw_guard_deg'] = 0.0
        super().__init__(environment, trial_environment=trial_environment, **kwargs)
        self.y_guard_enabled = bool(y_guard_enabled)
        self.y_guard_max_error_deg = float(y_guard_max_error_deg)
        self.y_guard_worsening_deg = float(y_guard_worsening_deg)
        self.motion_evidence_enabled = bool(motion_evidence_enabled)
        self.motion_evidence_min_m = float(motion_evidence_min_m)
        self.motion_evidence_progress_m = float(motion_evidence_progress_m)
        self._reset_v4_diagnostics()

    def _reset_v4_diagnostics(self):
        self.y_guard_stops = 0
        self.y_guard_checked_steps = 0
        self.motion_evidence_overrides = 0
        self.pre_push_contact_observed = 0

    def reset(self):
        super().reset()
        self._reset_v4_diagnostics()

    def _yaw_target_error_deg(self, yaw: float) -> float:
        return float(abs(np.rad2deg(wrapped(float(self.environment.target_yaw) - yaw))))

    def _four_log_push(self, eef):
        """Reuse parent logger then correct motion-evidence false negatives.

        This only changes the learned contact-point efficacy when there is
        measurable forward progress. The raw MuJoCo contact count stays intact.
        """
        pending = getattr(self, '_four_active', None)
        before = len(self.four_way_samples)
        super()._four_log_push(eef)
        if pending is None or len(self.four_way_samples) == before:
            return
        sample = self.four_way_samples[-1]
        sample['pre_push_contact_frames'] = int(pending.get('pre_push_contact_frames', 0))
        sample['y_guard_start_error_deg'] = pending.get('y_guard_start_error_deg', None)
        sample['y_guard_stop_error_deg'] = pending.get('y_guard_stop_error_deg', None)
        sample['motion_evidence_override'] = False

        if (not self.motion_evidence_enabled
                or sample['useful']
                or sample['robot_tee_contact_frames'] != 0
                or not sample['contact_api_available']
                or float(sample['axis_progress_m']) <= self.motion_evidence_progress_m
                or float(sample['object_translation_m']) < self.motion_evidence_min_m):
            return

        # The object visibly progressed during the stroke; retain this contact
        # point even if a small window of contact sampling missed the collision.
        sample['useful'] = True
        sample['motion_evidence_override'] = True
        self.motion_evidence_overrides += 1
        self.four_way_bad_pushes = max(0, self.four_way_bad_pushes - 1)
        self._four_stalls[pending['direction']] = 0
        self._four_blocked.clear()

        # Update the SAME empirical response model as the parent would when it
        # sees a verified (or unobservable) contact. This prevents a stale
        # response from biasing the next stroke in the same direction.
        position, yaw = angular_pose(self.environment)
        travel = max(.003, float(sample['eef_travel_m']))
        delta = position - np.asarray(pending['start_xy'], dtype=float)
        normal = np.asarray(pending['normal'], dtype=float)
        axis_gain = float(np.clip(float(delta @ normal) / travel, -1, 2))
        axis_idx = 0 if 'x' in pending['direction'] else 1
        old_error = np.asarray(pending['error_before'], dtype=float)
        new_error = self._four_local_error(position, pending['target'])
        cross_gain = float((old_error[1 - axis_idx] - new_error[1 - axis_idx]) / travel)
        yaw_gain = float(np.clip(wrapped(yaw - float(pending['start_yaw'])) / travel, -45, 45))
        key = pending['candidate']
        if key in self._four_response:
            old_g, old_y, old_c = self._four_response[key]
            axis_gain, yaw_gain, cross_gain = (
                .6 * old_g + .4 * axis_gain,
                .6 * old_y + .4 * yaw_gain,
                .6 * old_c + .4 * cross_gain,
            )
        self._four_response[key] = (axis_gain, yaw_gain, cross_gain)

    def action(self, observation):
        # Record contacts during approach and descend without reclassifying
        # them as push-phase contact. The original contact counter is unchanged.
        planned = getattr(self, '_four_planned', None)
        if (planned is not None and getattr(self, '_four_active', None) is None
                and getattr(self, 'phase', None) in ('approach', 'descend')
                and getattr(self, 'mode', None) == 'translate'):
            available, hit = tee_robot_contact(self.environment)
            if available and hit:
                planned['pre_push_contact_frames'] = int(
                    planned.get('pre_push_contact_frames', 0)) + 1
                self.pre_push_contact_observed += 1

        pending = self._monitored_stroke()
        if (self.y_guard_enabled and pending is not None
                and pending.get('direction') in ('+y', '-y')):
            _, current_yaw = angular_pose(self.environment)
            start_err = self._yaw_target_error_deg(float(pending['start_yaw']))
            current_err = self._yaw_target_error_deg(current_yaw)
            self.y_guard_checked_steps += 1
            if should_stop_y_push(
                    start_err, current_err,
                    max_error_deg=self.y_guard_max_error_deg,
                    worsening_deg=self.y_guard_worsening_deg):
                pending['feedback_stop_reason'] = 'y_yaw_guard'
                pending['feedback_remaining_at_stop_m'] = signed_axis_remaining(
                    pending['direction'],
                    self.environment.sim.data.body_xpos[self.environment.cube_body_id, :2],
                    pending['target'], self.environment.target_yaw)
                pending['feedback_buffer_at_stop_m'] = None
                pending['y_guard_start_error_deg'] = start_err
                pending['y_guard_stop_error_deg'] = current_err
                self.feedback_stops += 1
                self.feedback_yaw_stops += 1
                self.y_guard_stops += 1
                # The inherited action() detects PLAN with an active stroke,
                # logs it, then retreats and lifts before planning a new face.
                self.phase = 'plan'
                return np.asarray(super().action(observation), dtype=float)

        return np.asarray(super().action(observation), dtype=float)


# Weak priors (rad per metre of OBJECT movement) estimated from the supplied
# Empirical motion measurements, not an independent calibration; online motion
# supersedes them. Other faces retain the geometry prior.
YAW_PRIORS = dict(yp_bar_near=-3.24, yn_bar_near=2.61,
                  yp_stem_mid=5.81, yn_stem_mid=-6.18)


def rotation_cost(position, target, angle, turn, delta, travel, consistency):
    """Metre-equivalent yaw loss + goal progress + lateral/away-motion cost."""
    d = target-position
    after = target-position-delta
    forward = d / max(np.linalg.norm(d), 1e-9)
    axial = float(delta @ forward)
    lateral = float(np.linalg.norm(delta-axial*forward))
    # Strongly prefer a forward correcting stroke over one that drives away.
    outside = np.linalg.norm(position+delta-np.clip(position+delta,[-.19,-.19],[.15,.19]))
    return float(.17*abs(wrapped(angle-turn)) + .65*max(0.,np.linalg.norm(after)-np.linalg.norm(d))
                 + .32*lateral - .20*min(max(axial,0.),.045)
                 + 2.0*outside + .008*travel + .006*(1.-consistency))


class TargetAwareExpert(YawGuardExpert):
    """Use calibrated rotation in open mode; retain closed XY feedback.

    Changes: favor rotation strokes toward the goal; measure full-stroke net
    pose (incl. approach/retreat); retreat before changing rotation contact;
    bound damaging strokes; fine closed-loop XY and yaw-aware Y selection;
    bounded recovery instead of retrying an impossible fallback indefinitely.
    """
    def __init__(self, environment, trial_environment=None, **kwargs):
        kwargs.setdefault('rotate_budget_steps', 185)  # internal; episode still 400
        kwargs.setdefault('max_rotate_pushes', 5)
        kwargs.setdefault('min_rotate_stroke_m', .006)
        kwargs.setdefault('max_rotate_stroke_m', .045)
        super().__init__(environment, trial_environment=trial_environment, **kwargs)
        self._reset_v5()

    def _reset_v5(self):
        self._rotation_snapshot = None
        self._rotation_post = None
        self._rotation_net = {}
        self._v5_yaw_bad = set()
        self._fallback_pose = None
        self._fallback_retries = 0
        self._fine_reentries = 0
        self._v5_terminal = False
        self.v5_rotation_retreats = 0
        self.v5_drift_stops = 0
        self.v5_fine_plans = 0
        self.v5_recovery_plans = 0
        self.v5_unreachable = False
        self.v5_rotation_net_samples = []

    def reset(self):
        super().reset()
        self._reset_v5()

    def _goal(self):
        return self.environment.sim.data.body_xpos[self.environment.target_body_id,:2].copy()

    def _plan_rotation(self, position, rotation, angle, eef):
        yaw = float(np.arctan2(rotation[1,0],rotation[0,0]))
        b = heading_bin(yaw,self.model_bins)
        self.last_calibration_bin = b
        target = self._goal()
        options = []
        for i, (local_point,local_normal) in enumerate(self.contacts):
            r = self.calibration.get((i,b))
            if r is None or i in self._rotate_blocked:
                continue
            rate = float(r['rate_rad_per_m'])
            normal = rotation @ local_normal
            point = position + rotation @ local_point
            approach = point-.024*normal
            if np.any(approach<[-.26,-.25]) or np.any(approach>[.21,.25]):
                continue
            gain = float(np.clip(r['translation_per_stroke'],.15,2.0))
            velocity = gain*local_normal
            if i in self._rotation_net:
                actual_velocity,actual_rate = self._rotation_net[i]
                if actual_rate*rate>0:
                    rate=.55*rate+.45*actual_rate
                    velocity=.55*velocity+.45*actual_velocity
            if rate*angle<=0 or abs(rate)<.35:
                continue
            ideal=abs(angle/rate)
            # Near the goal avoid a full 3cm rotation stroke for a few degrees.
            cap=min(self.max_rotate_stroke_m,.020 if np.linalg.norm(target-position)<.025 else self.max_rotate_stroke_m)
            for s in np.unique(np.clip([.55*ideal,.8*ideal,ideal],self.min_rotate_stroke_m,cap)):
                turn=rate*s
                delta=rotation@velocity*s
                cost=rotation_cost(position,target,angle,turn,delta,np.linalg.norm(eef[:2]-approach),float(r['sign_consistency']))
                options.append((cost,i,approach,point+s*normal,float(s),normal,point,turn))
        if not options:
            self.no_calibrated_contact=True
            self.rotate_unreachable_exit=True
            self.rotate_steps=self.rotate_budget_steps
            return False
        _,i,self.approach,self.end,s,n,p,turn=min(options,key=lambda x:x[0])
        self.rotation=self.wrist_rotation(n)
        self._planned_rotation_push=dict(contact=i,stroke=s,normal=n.copy(),point=p.copy(),predicted_turn=turn)
        self._rotation_snapshot=dict(contact=i,xy=position.copy(),yaw=yaw,angle=angle,stroke=s,normal=n.copy(),distance=float(np.linalg.norm(target-position)))
        self._last_selected_rotation_contact=i
        self.last_contact=None
        self.rotate_plans+=1
        self.calibrated_plans+=1
        self.calibrated_contacts_used.add(i)
        return True

    def _finish_net_rotation(self):
        snap=self._rotation_snapshot
        if snap is None: return
        pos,yaw=angular_pose(self.environment)
        angle=wrapped(self.environment.target_yaw-yaw)
        turn=wrapped(yaw-snap['yaw'])
        delta=pos-snap['xy']
        progress=abs(snap['angle'])-abs(angle)
        i=snap['contact']; s=max(snap['stroke'],.006)
        if abs(turn)>np.deg2rad(.2):
            vel=np.clip(rotation_2d(-snap['yaw'])@delta/s,-2,2)
            rate=float(np.clip(turn/s,-45,45))
            if i in self._rotation_net:
                v,r=self._rotation_net[i]; vel=.5*v+.5*vel; rate=.5*r+.5*rate
            self._rotation_net[i]=(vel,rate)
        # Do not repeatedly select a contact that undoes its own correction.
        away=float(np.linalg.norm(self._goal()-pos))-snap['distance']
        if progress<np.deg2rad(.25) or away>.022:
            self._rotate_blocked.add(i)
        elif progress>=np.deg2rad(.5):
            # Full-stroke improvement is stronger evidence than the parent's
            # geometric reached-contact test.
            self._rotate_blocked.discard(i)
        self.v5_rotation_net_samples.append(dict(contact=i,planned_stroke_m=s,
            net_turn_deg=float(np.rad2deg(turn)),net_progress_deg=float(np.rad2deg(progress)),
            net_translation_m=float(np.linalg.norm(delta)),
            goal_distance_before_m=snap['distance'],goal_distance_after_m=float(np.linalg.norm(self._goal()-pos))))
        self._rotation_snapshot=None

    def _start_rotation_retreat(self,eef):
        snap=self._rotation_snapshot
        if self._active_rotation_push is not None:
            self._finish_push_feedback(eef)
        self._planned_rotation_push=None
        self._rotation_post=dict(stage='retreat',steps=0,target=np.r_[eef[:2]-.012*snap['normal'],eef[2]])
        self.phase='plan'
        self.phase_steps=0
        self.v5_rotation_retreats+=1

    def _rotation_post_action(self,eef):
        post=self._rotation_post
        self.rotate_steps+=1
        post['steps']+=1
        if post['stage']=='retreat':
            if np.linalg.norm(eef-post['target'])<.005 or post['steps']>=12:
                post.update(stage='lift',steps=0)
            else:
                return np.r_[np.clip((post['target']-eef)*18.,-.5,.5),0.,0.,0.,-1.]
        if eef[2]<.895 and post['steps']<22:
            return np.r_[0.,0.,np.clip((.91-eef[2])*18.,0.,.7),0.,0.,0.,-1.]
        # Never move laterally at low height after a timeout. Parent plan will
        # continue raising, and a gripper transition also checks safe height.
        self._finish_net_rotation()
        self._rotation_post=None
        self.phase='plan'
        return np.r_[0.,0.,.4 if eef[2]<.895 else 0.,0.,0.,0.,-1.]

    def _axis_plan(self,position,rotation,target,angle,eef,direction,remaining,*,relaxed=False):
        """Closed Y/precision/recovery planner with actual yaw-response priors."""
        basis=rotation_2d(self.environment.target_yaw)
        idx=0 if 'x' in direction else 1
        forward=basis[:,idx]*(1 if direction[0]=='+' else -1)
        old=self._four_local_error(position,target)
        opts=[]
        fine=np.linalg.norm(target-position)<.012
        for face in FACES:
            if face.direction!=direction or face.name in self._four_blocked: continue
            normal=rotation@np.asarray(face.normal)
            if normal@forward<.94: continue
            point=position+rotation@np.asarray(face.xy)
            approach=point-(.032 if fine else .024)*normal
            low,high=([-.30,-.28],[.25,.28]) if relaxed else ([-.26,-.25],[.21,.25])
            if np.any(approach<low) or np.any(approach>high): continue
            # Servo stops on actual position, so predict angle per object motion.
            arm=np.asarray(face.xy)-self.center_of_mass
            n=np.asarray(face.normal)
            rate=YAW_PRIORS.get(face.name,float((arm[0]*n[1]-arm[1]*n[0])/.0024/.8))
            gain=.8
            if face.name in self._four_response:
                g,r,_=self._four_response[face.name]
                if g>.10: gain=g; rate=.25*rate+.75*r/g
                else: continue
            motion=min(remaining,.035)
            turn=rate*motion
            # Favor a face that corrects yaw, not merely a small geometric arm.
            cost=.14*abs(wrapped(angle-turn))+.010*np.linalg.norm(eef[:2]-approach)
            if face.name in self._v5_yaw_bad: cost+=.010
            if fine: cost+=.025*abs(turn)
            stroke=float(np.clip(.85*remaining/gain,self.min_stroke_m,self.max_stroke_m))
            if fine: stroke=min(stroke,.012)
            opts.append((cost,face,approach,point+stroke*normal,stroke,normal,point))
        if not opts: return False
        _,face,self.approach,self.end,stroke,n,p=min(opts,key=lambda v:v[0])
        self.rotation=self.wrist_rotation(n)
        self._four_planned=dict(direction=direction,candidate=face.name,surface=face.surface,normal=n.copy(),point=p.copy(),planned_stroke_m=stroke,target=target.copy(),error_before=old.copy())
        self.last_contact=None; self.last_error=None; self.four_way_plans+=1
        if fine: self.v5_fine_plans+=1
        return True

    def _plan_four_way(self,position,rotation,target,angle,eef):
        direction,remaining=self._four_direction(position,target)
        fine=np.linalg.norm(target-position)<.012
        if fine:
            err=self._four_local_error(position,target)
            # Correct the larger residual even when BOTH axes are below the old 4mm gate.
            i=int(np.argmax(np.abs(err)))
            direction=('+' if err[i]>=0 else '-')+'xy'[i]
            remaining=float(abs(err[i]))
        if direction is not None and (fine or 'y' in direction):
            if self._axis_plan(position,rotation,target,angle,eef,direction,remaining): return True
        if not fine:
            return super()._plan_four_way(position,rotation,target,angle,eef)
        return self._four_fallback(position,rotation,target,angle,eef)

    def _four_fallback(self,position,rotation,target,angle,eef):
        # First try the other axis and, only as recovery, a modestly wider
        # approach envelope. Never override the actual physical table state.
        self._four_planned=None
        self.four_way_fallbacks+=1
        pose=np.r_[position,angle]
        if self._fallback_pose is None or np.linalg.norm(pose[:2]-self._fallback_pose[:2])>.002 or abs(wrapped(pose[2]-self._fallback_pose[2]))>np.deg2rad(1):
            self._fallback_retries=0
        self._fallback_pose=pose.copy(); self._fallback_retries+=1
        err=self._four_local_error(position,target)
        for i in np.argsort(-np.abs(err)):
            if abs(err[i])<.001: continue
            direction=('+' if err[i]>0 else '-')+'xy'[i]
            if self._axis_plan(position,rotation,target,angle,eef,direction,float(abs(err[i])),relaxed=True):
                self.v5_recovery_plans+=1; return True
        if self._fallback_retries==1:
            self._four_blocked.clear()
            for i in np.argsort(-np.abs(err)):
                if abs(err[i])<.001: continue
                direction=('+' if err[i]>0 else '-')+'xy'[i]
                if self._axis_plan(position,rotation,target,angle,eef,direction,float(abs(err[i])),relaxed=True):
                    self.v5_recovery_plans+=1; return True
        # One bounded original fallback. Its success is still checked by the env.
        if self._fallback_retries<=2 and self.allow_original_fallback and np.linalg.norm(target-position)>=.012:
            from rotation_controller import RepositioningPushTExpert
            self.last_contact=None; self.last_error=None
            if RepositioningPushTExpert.plan(self,position,rotation,target,angle,eef): return True
        if self._fallback_retries>=3:
            self.v5_unreachable=True; self._v5_terminal=True
        return False

    def _four_log_push(self,eef):
        pending=self._four_active
        if pending is not None and pending.get('feedback_stop_reason')=='y_yaw_guard':
            self._v5_yaw_bad.add(pending['candidate'])
        super()._four_log_push(eef)

    def action(self,observation):
        eef=np.asarray(observation['robot0_eef_pos'],float)
        pos,yaw=angular_pose(self.environment)
        goal=self._goal(); distance=float(np.linalg.norm(goal-pos))
        angle=wrapped(self.environment.target_yaw-yaw); err=abs(np.rad2deg(angle))
        metrics=self.environment.task_metrics()
        success=metrics['iou']>=self.environment.overlap_threshold and metrics['upright'] and metrics['on_table']
        if success:
            if self._active_rotation_push is not None: self._finish_push_feedback(eef)
            if self._rotation_snapshot is not None: self._finish_net_rotation()
            self._rotation_post=None
        elif self._rotation_post is not None:
            return self._rotation_post_action(eef)
        elif self._v5_terminal:
            return np.r_[0.,0.,np.clip((.95-eef[2])*18.,0.,.7),0.,0.,0.,1. if self.grip_mode=='closed' else -1.]
        snap=self._rotation_snapshot
        if not success and self.mode=='rotate' and snap is not None and self.grip_mode=='open':
            if self.phase in ('descend','push') and (err<=self.rotate_goal_deg or distance>snap['distance']+.022):
                if distance>snap['distance']+.022: self.v5_drift_stops+=1
                self._start_rotation_retreat(eef); return self._rotation_post_action(eef)
            if self.phase=='plan':
                # Natural completion, including an unsuccessful approach.
                if self._active_rotation_push is not None or eef[2]<.885:
                    self._start_rotation_retreat(eef); return self._rotation_post_action(eef)
                self._finish_net_rotation()
        # One additional near-goal yaw correction if the ordinary reentry
        # threshold (8deg) would otherwise leave an unsatisfiable pose.
        if (not success and self.mode=='translate' and self.grip_mode=='closed' and self.phase=='plan'
                and eef[2]>=.895 and distance<.015 and err>=3.5 and self._fine_reentries<1
                and self.rotate_steps<self.rotate_budget_steps-20 and self.rotate_pushes<self.max_rotate_pushes
                and getattr(self.environment,'timestep',400)<getattr(self.environment,'expert_step_limit',400)-100):
            self._fine_reentries+=1
            self._set_mode('rotate',err)
        # Use finer speed and margin near the target; preserve coarse XY control.
        fine=(distance<.012 and self.mode=='translate' and self.grip_mode=='closed' and not success)
        attrs=('feedback_stop_margin_m','feedback_max_buffer_m','feedback_slow_command','feedback_fast_command','axis_tolerance_m')
        old=[getattr(self,k) for k in attrs]
        if fine:
            for k,v in zip(attrs,(.0006,.002,.04,.20,.0012)): setattr(self,k,v)
        try:
            return np.asarray(super().action(observation),float)
        finally:
            for k,v in zip(attrs,old): setattr(self,k,v)


class FineContactExpert(TargetAwareExpert):
    def _axis_plan(self, position, rotation, target, angle, eef, direction, remaining, *, relaxed=False):
        ok = super()._axis_plan(position, rotation, target, angle, eef, direction, remaining, relaxed=relaxed)
        if ok and np.linalg.norm(target-position) < .012:
            pending = self._four_planned
            # Use the standard contact approach gap.
            self.approach = pending['point'] - .024 * pending['normal']
            pending['v51_precision'] = True
        return ok

    def _four_log_push(self, eef):
        pending = self._four_active
        before = len(self.four_way_samples)
        super()._four_log_push(eef)
        if pending is not None and len(self.four_way_samples) > before:
            sample = self.four_way_samples[-1]
            for key in ('v51_precision', 'v51_seek_steps', 'v51_servo_steps', 'v51_contact_acquired', 'v51_measured_eef_travel_m'):
                sample[key] = pending.get(key, None)

    def _precision_stop(self, observation, p, reason, remaining, buffer=None):
        p['feedback_stop_reason'] = reason
        p['feedback_remaining_at_stop_m'] = float(remaining)
        p['feedback_buffer_at_stop_m'] = buffer
        self.feedback_stops += 1
        if reason == 'axis_goal': self.feedback_goal_stops += 1
        if reason == 'y_yaw_guard':
            self.feedback_yaw_stops += 1
            self.y_guard_stops += 1
        self.phase = 'plan'
        # Existing FSM logs, disengages and lifts, with closed gripper.
        return super().action(observation)

    def action(self, observation):
        p = self._monitored_stroke()
        if p is None or not p.get('v51_precision'):
            return super().action(observation)
        metrics = self.environment.task_metrics()
        if metrics['iou'] >= self.environment.overlap_threshold and metrics['upright'] and metrics['on_table']:
            return super().action(observation)
        xy, yaw = angular_pose(self.environment)
        eef = np.asarray(observation['robot0_eef_pos'], float)
        self._record_new_stroke(p, xy)
        p.setdefault('v51_seek_steps', 0)
        p.setdefault('v51_servo_steps', 0)
        p.setdefault('v51_contact_acquired', False)
        available, hit = tee_robot_contact(self.environment)
        p['contact_api_available'] |= bool(available)
        p['contact_frames'] += int(hit)
        local = rotation_2d(self.environment.target_yaw)
        idx = 0 if p['direction'][1] == 'x' else 1
        forward = local[:,idx] * (1 if p['direction'][0] == '+' else -1)
        remaining = signed_axis_remaining(p['direction'], xy, p['target'], self.environment.target_yaw)
        progress = max(0., float((xy-self._feedback_previous_xy) @ forward))
        self._feedback_previous_xy = xy.copy()
        self._feedback_last_remaining = remaining
        self.feedback_monitored_steps += 1
        p['feedback_monitored_steps'] += 1
        normal = np.asarray(p['normal'], float)
        travel = float((eef[:2]-p['start_eef']) @ normal)
        p['v51_measured_eef_travel_m'] = travel
        # Contact API may miss between-control-step collisions; motion is a separate signal.
        moved = float((xy-p['start_xy']) @ forward) >= .00035
        p['v51_contact_acquired'] |= bool(hit or moved or p.get('pre_push_contact_frames',0)>0)
        if self.y_guard_enabled and idx == 1:
            start_err = self._yaw_target_error_deg(p['start_yaw'])
            err = self._yaw_target_error_deg(yaw)
            self.y_guard_checked_steps += 1
            if should_stop_y_push(start_err,err,max_error_deg=self.y_guard_max_error_deg,worsening_deg=self.y_guard_worsening_deg):
                p['y_guard_start_error_deg'],p['y_guard_stop_error_deg'] = start_err,err
                return self._precision_stop(observation,p,'y_yaw_guard',remaining)
        if not (metrics['upright'] and metrics['on_table']):
            return self._precision_stop(observation,p,'invalid_pose',remaining)
        buffer = stop_buffer(progress,margin_m=.0006,lookahead_steps=1.,maximum_m=.002)
        if remaining <= buffer:
            return self._precision_stop(observation,p,'axis_goal',remaining,buffer)
        # Do not mistake the geometric EEF endpoint for physical contact.
        # Bound every attempt by time and actual EEF travel, including seek.
        if travel >= .048 or p['v51_seek_steps'] >= 24 or p['v51_servo_steps'] >= 30:
            reason = 'precision_servo_limit' if p['v51_contact_acquired'] else 'precision_no_contact'
            return self._precision_stop(observation,p,reason,remaining)
        if p['v51_contact_acquired']:
            cap = float(np.clip(remaining*12.,.035,.11))
            p['v51_servo_steps'] += 1
            p['feedback_slow_steps'] += 1
            self.feedback_slow_steps += 1
        else:
            cap = .16
            p['v51_seek_steps'] += 1
        # Preserve the same wrist pose and pushing height; drive only along face normal.
        site = self.environment.robots[0].eef_site_id['right']
        current = self.environment.sim.data.site_xmat[site].reshape(3,3)
        rot = np.clip(Rotation.from_matrix(self.rotation @ current.T).as_rotvec()/.5,-.5,.5)
        self.closed_translation_steps += 1
        return np.r_[normal*cap,np.clip((.824-eef[2])*18.,-.25,.25),rot,1.]


# radians of object turn per metre of target-axis object motion.
# Bar rates and -Y stem rate are derived approximately from the two traces;
# +Y stem retains its empirical prior.
RATES={'yp_bar_near':-2.9,'yp_stem_mid':5.81,
       'yn_bar_near':2.7,'yn_stem_mid':-6.7}

def split_motion(error, distance, r1, r2):
    """Solve r1*d1+r2*(distance-d1)=error, with nonnegative travel."""
    if distance<=0 or abs(r1-r2)<1e-9: return 0.
    return float(np.clip((error-r2*distance)/(r1-r2),0.,distance))

class LargeRouteTranslation(FineContactExpert):
    large_route=False
    def reset(self):
        super().reset()
        self.large_route=False
        self.balance_rates={}

    def _four_direction(self, position, target):
        if self.large_route:
            x,y=self._four_local_error(position,target)
            if abs(x)<.007 and abs(y)>.012:
                return ('+' if y>0 else '-')+'y',float(abs(y))
        return super()._four_direction(position,target)

    def _balanced_y(self,position,rotation,target,angle,eef,direction,remaining):
        prefix='yp' if direction=='+y' else 'yn'
        names=[prefix+'_bar_near',prefix+'_stem_mid']
        rates=getattr(self,'balance_rates',{})
        viable={}
        for name in names:
            if name in self._four_blocked: continue
            f=FACE_BY_NAME[name]; n=rotation@np.asarray(f.normal); p=position+rotation@np.asarray(f.xy)
            approach=p-.024*n
            forward=rotation_2d(self.environment.target_yaw)[:,1]*(1 if direction=='+y' else -1)
            if n@forward<.94 or np.any(approach<[-.26,-.25]) or np.any(approach>[.21,.25]): continue
            viable[name]=(f,n,p,approach,rates.get(name,RATES[name]))
        if not viable: return False
        opts=[]
        # Prefer a single stroke if it is predicted to align both Y and yaw.
        for name,(f,n,p,a,r) in viable.items():
            enderr=abs(wrapped(angle-r*remaining))
            opts.append((enderr+.025*np.linalg.norm(eef[:2]-a),name,remaining,None))
        if len(viable)==2:
            for first,second in (names,names[::-1]):
                r1=viable[first][-1]; r2=viable[second][-1]
                d1=split_motion(angle,remaining,r1,r2); d2=remaining-d1
                if min(d1,d2)<.007: continue
                intermediate=abs(wrapped(angle-r1*d1))
                # Intermediate yaw can be nonzero, but must remain small enough
                # for ordinary translation; prefer first contact that initially corrects.
                if intermediate>np.deg2rad(8.): continue
                score=(abs(wrapped(angle-r1*d1-r2*d2))+.035+.20*intermediate
                       +(.035 if angle*r1<0 else 0.)+.025*np.linalg.norm(eef[:2]-viable[first][3]))
                opts.append((score,first,d1,second))
        _,name,motion,second=min(opts,key=lambda o:o[0])
        f,n,p,approach,rate=viable[name]
        # Allow EEF enough geometric travel; actual object feedback stops it.
        stroke=float(np.clip(motion+.018,.020,.095))
        self.approach=approach; self.end=p+stroke*n; self.rotation=self.wrist_rotation(n)
        self._four_planned=dict(direction=direction,candidate=name,surface=f.surface,normal=n.copy(),point=p.copy(),
            planned_stroke_m=stroke,target=target.copy(),error_before=self._four_local_error(position,target),
            balance_y=True,balance_floor_m=max(0.,remaining-motion),balance_planned_motion_m=motion,
            balance_next=second or '',balance_rate=rate)
        self.last_contact=None; self.last_error=None; self.four_way_plans+=1
        return True

    def _plan_four_way(self,position,rotation,target,angle,eef):
        if not self.large_route: return super()._plan_four_way(position,rotation,target,angle,eef)
        direction,remaining=self._four_direction(position,target)
        if direction and 'y' in direction and remaining>=.012 and abs(angle)<=np.deg2rad(9):
            if self._balanced_y(position,rotation,target,angle,eef,direction,remaining): return True
        old=self.max_stroke_m
        if direction and 'x' in direction and remaining>.045: self.max_stroke_m=.095
        try: return super()._plan_four_way(position,rotation,target,angle,eef)
        finally: self.max_stroke_m=old

    def _four_log_push(self,eef):
        p=self._four_active; before=len(self.four_way_samples)
        if p and p.get('balance_y'):
            pos,yaw=angular_pose(self.environment)
            f=rotation_2d(self.environment.target_yaw)[:,1]*(1 if p['direction']=='+y' else -1)
            motion=float((pos-p['start_xy'])@f)
            if motion>.006:
                measured=float(np.clip(wrapped(yaw-p['start_yaw'])/motion,-15.,15.))
                self.balance_rates=getattr(self,'balance_rates',{})
                self.balance_rates[p['candidate']]=measured
                p['balance_measured_rate']=measured
        super()._four_log_push(eef)
        if p and len(self.four_way_samples)>before:
            self.four_way_samples[-1].update({k:v for k,v in p.items() if k.startswith('balance_')})

    def action(self, observation):
        p=self._monitored_stroke() if self.large_route else None
        if p is None or not p.get('balance_y'): return super().action(observation)
        env=self.environment; pos,yaw=angular_pose(env); m=env.task_metrics()
        if m['iou']>=env.overlap_threshold and m['upright'] and m['on_table']: return super().action(observation)
        self._record_new_stroke(p,pos)
        remaining=signed_axis_remaining(p['direction'],pos,p['target'],env.target_yaw)
        gap=remaining-p['balance_floor_m']
        prev=p.get('balance_previous_remaining',remaining); p['balance_previous_remaining']=remaining
        progress=max(0.,prev-remaining)
        buffer=stop_buffer(progress,margin_m=.0015,lookahead_steps=1.5,maximum_m=.004)
        err=abs(np.rad2deg(wrapped(env.target_yaw-yaw)))
        start=abs(np.rad2deg(wrapped(env.target_yaw-p['start_yaw'])))
        reason=None
        if not(m['upright'] and m['on_table']): reason='balance_invalid_pose'
        elif err>9. and err>start+1.: reason='balance_yaw_limit'
        elif p['balance_floor_m']>.001 and gap<=buffer: reason='balance_split'
        if reason:
            available,hit=tee_robot_contact(env); p['contact_api_available'] |= bool(available); p['contact_frames']+=int(hit)
            p['feedback_stop_reason']=reason; p['feedback_remaining_at_stop_m']=remaining; p['feedback_buffer_at_stop_m']=buffer
            self.phase='plan'; self.feedback_stops+=1
            return super().action(observation)
        # Planned temporary yaw excursion is necessary for opposite-torque pair.
        # This paired stroke bypasses the 4-degree guard; the 9-degree limit remains.
        old=self.y_guard_enabled; self.y_guard_enabled=False
        try: a=np.asarray(super().action(observation),float).copy()
        finally: self.y_guard_enabled=old
        if self._monitored_stroke() is p and p['balance_floor_m']>.001:
            cap=speed_limit(gap,slowdown_start_m=.018,slow_command=.10,fast_command=.70)
            peak=float(np.max(np.abs(a[:2])))
            if peak>cap: a[:2]*=cap/peak
        return a


class WideAngleTranslationController(LargeRouteTranslation):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw); self._reset_joint()
    wide_angle_route=False
    def _reset_joint(self):
        self.wide_angle_route=False
        self.joint_attempts=0; self.joint_events=[]; self.joint_rates={}; self.joint_snapshot=None
        self.joint_blocked=set(); self.joint_terminal=False
    def reset(self):
        super().reset(); self._reset_joint()
    def _finish_joint(self):
        snap=self.joint_snapshot
        if snap is None: return
        p,y=angular_pose(self.environment); delta=rotation_2d(-snap['yaw'])@(p-snap['xy']); turn=wrapped(y-snap['yaw'])
        motion=float(delta@NORMALS[snap['name']]); ev=dict(event='net',attempt=snap['attempt'],step=int(self.environment.timestep),candidate=snap['name'],actual_motion_m=motion,actual_dx_m=float(delta[0]),actual_dy_m=float(delta[1]),actual_yaw_deg=float(np.rad2deg(turn)),iou=self.environment.task_metrics()['iou'])
        self.joint_events.append(ev)
        if motion>.003:
            old=np.asarray(self.joint_rates.get(snap['name'],self.planner.PRIORS[snap['name']]))
            measured=np.r_[np.clip(delta/motion,-2,2),np.clip(turn/motion,-12,12)]
            self.joint_rates[snap['name']]=.35*old+.65*measured
        elif motion<.0008: self.joint_blocked.add(snap['name'])
        self.joint_snapshot=None
    def _plan_four_way(self,position,rotation,target,angle,eef):
        if not self.large_route: return super()._plan_four_way(position,rotation,target,angle,eef)
        self._finish_joint()
        if self.joint_attempts>=3 or self.joint_terminal: return False
        env=self.environment; remaining=getattr(env,"expert_step_limit",400)-int(env.timestep)
        pose=np.r_[rotation_2d(-env.target_yaw)@(position-target),-angle]
        plan=self.planner.solve(pose,target,env.target_yaw,eef,remaining,3-self.joint_attempts,self.joint_rates,self.joint_blocked)
        if plan is None:
            self.joint_terminal=True; self.joint_events.append(dict(event='no_feasible_plan',step=int(env.timestep),attempt=self.joint_attempts)); return False
        name=plan.names[0]; dist=float(plan.distances[0]); face=FACE_BY_NAME[name]; normal=rotation@np.asarray(face.normal); point=position+rotation@np.asarray(face.xy)
        # The intermediate target is the intended object displacement along this
        # stroke, not the overall task goal. Parent approach guard uses target-frame axes.
        goal_basis=rotation_2d(env.target_yaw); idx=0 if face.direction[1]=='x' else 1; forward=goal_basis[:,idx]*(1 if face.direction[0]=='+' else -1)
        projected=dist*float(normal@forward); intermediate=position+projected*forward
        self.approach=point-.024*normal; stroke=dist+.018; self.end=point+stroke*normal; self.rotation=self.wrist_rotation(normal)
        self.joint_attempts+=1; attempt=self.joint_attempts
        self._four_planned=dict(direction=face.direction,candidate=name,surface=face.surface,normal=normal.copy(),point=point.copy(),planned_stroke_m=stroke,target=intermediate.copy(),error_before=self._four_local_error(position,intermediate),joint=True,joint_attempt=attempt,joint_motion_m=dist,joint_origin=position.copy(),joint_goal=target.copy(),joint_sequence=tuple(plan.names),joint_distances=plan.distances.copy(),joint_original_motion=dist)
        self.joint_snapshot=dict(name=name,attempt=attempt,xy=position.copy(),yaw=env.target_yaw-angle)
        self.joint_events.append(dict(event='plan',attempt=attempt,step=int(env.timestep),candidate=name,sequence='>'.join(plan.names),distances_mm=';'.join(f'{1000*d:.2f}' for d in plan.distances),predicted_iou=plan.predicted_iou,variant_min_iou=plan.variant_min_iou,estimated_steps=plan.estimated_steps,remaining_steps=remaining,predicted_x_error_mm=-1000*plan.residual[0],predicted_y_error_mm=-1000*plan.residual[1],predicted_yaw_error_deg=-np.rad2deg(plan.residual[2])))
        self.last_contact=None; self.last_error=None; self.four_way_plans+=1
        return True
    def _four_log_push(self,eef):
        p=self._four_active; before=len(self.four_way_samples)
        super()._four_log_push(eef)
        if p and p.get('joint') and len(self.four_way_samples)>before:
            self.four_way_samples[-1].update(joint_attempt=p['joint_attempt'],joint_motion_m=p['joint_motion_m'],joint_actual_motion_m=p.get('joint_actual_motion_m'),joint_seek_steps=p.get('joint_seek_steps',0),joint_servo_steps=p.get('joint_servo_steps',0))

    def action(self,obs):
        p=self._monitored_stroke() if self.large_route else None
        if p is None or not p.get('joint'):
            # Prevent any new plan from being launched below safe clearance after
            # the inherited lift timeout. Retain actual lift and gripper logic.
            if self.large_route and self.phase=='four_lift' and float(obs['robot0_eef_pos'][2])<.895:
                return np.r_[0.,0.,np.clip((.91-float(obs['robot0_eef_pos'][2]))*18.,0.,.7),0.,0.,0.,1.]
            planned=self._four_planned
            if planned and planned.get('joint'):
                old=self.feedback_stop_margin_m; self.feedback_stop_margin_m=.00065
                try: return super().action(obs)
                finally: self.feedback_stop_margin_m=old
            return super().action(obs)
        env=self.environment; m=env.task_metrics(); xy,yaw=angular_pose(env); eef=np.asarray(obs['robot0_eef_pos'],float); n=p['normal']
        self._record_new_stroke(p,xy)
        available,hit=tee_robot_contact(env); p['contact_api_available'] |= bool(available); p['contact_frames']+=int(hit)
        motion=float((xy-p['joint_origin'])@n)
        self._adapt_joint(p,xy,yaw,eef)
        gap=p['joint_motion_m']-motion
        progress=max(0.,motion-p.get('joint_actual_motion_m',motion)); p['joint_actual_motion_m']=motion
        p['joint_contact']=bool(p.get('joint_contact',False) or hit or motion>.00035 or p.get('pre_push_contact_frames',0)>0)
        p.setdefault('joint_seek_steps',0); p.setdefault('joint_servo_steps',0)
        buffer=stop_buffer(progress,margin_m=.00065,lookahead_steps=1.2,maximum_m=.0035)
        success=m['iou']>=env.overlap_threshold and m['upright'] and m['on_table']
        travel=float((eef[:2]-p['start_eef'])@n)
        reason=None
        if success: reason='joint_success'
        elif not(m['upright'] and m['on_table']): reason='joint_invalid_pose'
        elif abs(wrapped(env.target_yaw-yaw))>np.deg2rad(30 if self.wide_angle_route else 16): reason='joint_yaw_limit'
        elif gap<=buffer: reason='joint_distance'
        elif travel>p['joint_motion_m']+.035 or p['joint_seek_steps']>=24 or p['joint_servo_steps']>=60: reason='joint_travel_or_time_limit'
        elif int(env.timestep)>=getattr(env,"expert_step_limit",400)-3: reason='joint_episode_end'
        if reason:
            p['feedback_stop_reason']=reason; p['feedback_remaining_at_stop_m']=gap; p['feedback_buffer_at_stop_m']=buffer
            self.phase='plan'; self.feedback_stops+=1
            return super().action(obs)
        # Same closed wrist, height, and face normal as inherited translation.
        # No 28-step geometric-end cutoff, and no Y guard for planned yaw excursion.
        if not p['joint_contact']:
            cap=.24; p['joint_seek_steps']+=1
        else:
            cap=float(np.clip(gap*20,.055,.70)); p['joint_servo_steps']+=1
        site=env.robots[0].eef_site_id['right']; current=env.sim.data.site_xmat[site].reshape(3,3)
        r=np.clip(Rotation.from_matrix(self.rotation@current.T).as_rotvec()/.5,-.5,.5)
        self.closed_translation_steps+=1
        return np.r_[n*cap,np.clip((.824-eef[2])*18,-.25,.25),r,1.]

class ContactRecoveryController(WideAngleTranslationController):
    STALL_STEPS = 3
    RECOVERY_STEPS = 8  # cumulative per stroke, never another approach/descend
    MIN_PROGRESS = .00025

    def _four_log_push(self, eef):
        p = self._four_active
        before = len(self.four_way_samples)
        super()._four_log_push(eef)
        if p is not None and p.get('joint') and len(self.four_way_samples) > before:
            self.four_way_samples[-1].update({k: p.get(k, 0) for k in (
                'joint_recovery_steps', 'joint_recovery_entries', 'joint_recovered',
                'joint_max_stall_steps', 'joint_no_contact_steps', 'joint_recovery_failure')})

    def _recover_command(self, p, xy, yaw, eef, gap):
        """Follow the current face, nudging toward the last verified contact.
        Anchor is measured EEF position in object coordinates, not assumed finger
        geometry. Keep the original wrist and height, and do not slide sideways
        by more than a small bounded correction. Distances remain in the original
        stroke frame for the unchanged planner and live-tail model.
        """
        face = FACE_BY_NAME[p['candidate']]
        R = rotation_2d(yaw)
        n = R @ np.asarray(face.normal, float)
        if n @ p['normal'] < .85:
            return None, 'face_changed'
        tangent = np.array([-n[1], n[0]])
        lateral = 0.
        anchor = p.get('joint_contact_anchor')
        if anchor is not None:
            desired = xy + R @ anchor
            error = desired - eef[:2]
            # Beyond this envelope, low-height recontact is not safe to guess.
            if np.linalg.norm(error) > .025 or abs(error @ tangent) > .012:
                return None, 'anchor_outside_envelope'
            lateral = float(np.clip(12. * (error @ tangent), -.08, .08))
        # No blind sideways searching. Unknown contact API can use the current
        # normal and measured progress, but never a fabricated contact anchor.
        forward = float(np.clip(gap * 20., .16, .26))
        command = n * forward + tangent * lateral
        if command @ p['normal'] <= 0.:
            return None, 'reverse_command'
        # Apply the contact workspace envelope with a one-step margin.
        next_xy = eef[:2] + .05 * command
        if np.any(next_xy < [-.29, -.28]) or np.any(next_xy > [.25, .28]):
            return None, 'workspace'
        return command, ''

    def action(self, obs):
        p = self._monitored_stroke() if self.large_route and self.wide_angle_route else None
        if p is None or not p.get('joint'):
            return super().action(obs)
        env = self.environment
        m = env.task_metrics(); xy, yaw = angular_pose(env)
        eef = np.asarray(obs['robot0_eef_pos'], float); n = p['normal']
        self._record_new_stroke(p, xy)
        available, hit = tee_robot_contact(env)
        p['contact_api_available'] |= bool(available); p['contact_frames'] += int(hit)
        motion = float((xy-p['joint_origin']) @ n)
        progress = max(0., motion-p.get('joint_actual_motion_m', motion))
        p['joint_actual_motion_m'] = motion
        # Separate current evidence from the historical acquisition flag.
        had_contact = p.get('joint_ever_contact', False)
        current = bool(hit or progress >= self.MIN_PROGRESS)
        p['joint_contact'] = current
        p['joint_ever_contact'] = bool(had_contact or current)
        p['joint_no_contact_steps'] = p.get('joint_no_contact_steps', 0) + int(available and not hit and had_contact)
        # Refresh the anchor only on a verified contact with motion, or first hit.
        if hit and (progress >= self.MIN_PROGRESS or 'joint_contact_anchor' not in p):
            p['joint_contact_anchor'] = rotation_2d(-yaw) @ (eef[:2] - xy)
        if p['joint_ever_contact']:
            p['joint_stall_steps'] = 0 if progress >= self.MIN_PROGRESS else p.get('joint_stall_steps', 0) + 1
        p['joint_max_stall_steps'] = max(p.get('joint_max_stall_steps', 0), p.get('joint_stall_steps', 0))
        if current:
            self._adapt_joint(p, xy, yaw, eef)
        gap = p['joint_motion_m'] - motion
        p.setdefault('joint_seek_steps', 0); p.setdefault('joint_servo_steps', 0)
        p.setdefault('joint_recovery_steps', 0); p.setdefault('joint_recovery_entries', 0)
        buffer = stop_buffer(progress, margin_m=.00065, lookahead_steps=1.2, maximum_m=.0035)
        success = m['iou'] >= env.overlap_threshold and m['upright'] and m['on_table']
        travel = float((eef[:2]-p['start_eef']) @ n)
        reason = None
        if success: reason = 'joint_success'
        elif not (m['upright'] and m['on_table']): reason = 'joint_invalid_pose'
        elif abs(wrapped(env.target_yaw-yaw)) > np.deg2rad(30): reason = 'joint_yaw_limit'
        elif gap <= buffer: reason = 'joint_distance'
        elif travel > p['joint_motion_m']+.035 or p['joint_seek_steps'] >= 24 or p['joint_servo_steps'] >= 60: reason = 'joint_travel_or_time_limit'
        elif int(env.timestep) >= getattr(env,"expert_step_limit",400)-3: reason = 'joint_episode_end'
        xy_command = None
        if reason is None and p['joint_ever_contact']:
            if p.get('joint_recovering') and progress >= self.MIN_PROGRESS:
                p['joint_recovered'] = p.get('joint_recovered', 0) + 1
                p['joint_recovering'] = False
                p['joint_follow_face'] = True
                self.joint_events.append(dict(event='contact_recovered', step=int(env.timestep), attempt=p['joint_attempt'], candidate=p['candidate'], actual_motion_m=motion))
            if p.get('joint_stall_steps', 0) >= self.STALL_STEPS:
                if p['joint_recovery_steps'] >= self.RECOVERY_STEPS:
                    reason = 'joint_contact_recovery_exhausted'
                else:
                    if not p.get('joint_recovering'):
                        p['joint_recovery_entries'] += 1; p['joint_recovering'] = True
                        self.joint_events.append(dict(event='contact_recovery', step=int(env.timestep), attempt=p['joint_attempt'], candidate=p['candidate'], actual_motion_m=motion, remaining_motion_m=gap, verified_contact=bool(hit)))
                    xy_command, failure = self._recover_command(p, xy, yaw, eef, gap)
                    if failure:
                        p['joint_recovery_failure'] = failure
                        reason = 'joint_contact_recovery_unsafe'
                    else:
                        p['joint_recovery_steps'] += 1
            elif p.get('joint_follow_face'):
                # After successful recovery, follow the current face normal with
                # the ordinary speed schedule; don't revert to the stale direction.
                face = FACE_BY_NAME[p['candidate']]
                live_n = rotation_2d(yaw) @ np.asarray(face.normal, float)
                if live_n @ n < .85:
                    reason = 'joint_contact_recovery_unsafe'
                    p['joint_recovery_failure'] = 'face_changed'
                else:
                    xy_command = live_n * float(np.clip(gap*20, .055, .70))
        if reason:
            p['feedback_stop_reason'] = reason; p['feedback_remaining_at_stop_m'] = gap; p['feedback_buffer_at_stop_m'] = buffer
            self.phase = 'plan'; self.feedback_stops += 1
            return super().action(obs)
        if not p['joint_ever_contact']:
            cap = .24; p['joint_seek_steps'] += 1
        else:
            cap = float(np.clip(gap*20, .055, .70)); p['joint_servo_steps'] += 1
        site = env.robots[0].eef_site_id['right']; current_rot = env.sim.data.site_xmat[site].reshape(3,3)
        r = np.clip(Rotation.from_matrix(self.rotation @ current_rot.T).as_rotvec()/.5, -.5, .5)
        self.closed_translation_steps += 1
        return np.r_[n*cap if xy_command is None else xy_command, np.clip((.824-eef[2])*18,-.25,.25), r, 1.]


# Exactly the original rotation and handoff implementation.

def material_command(p, xy, yaw, eef, gap, recovery=False):
    """Track the FIRST measured contact in body coordinates, not the last slip.
    All distances used by the planner remain projections on its original normal.
    """
    R = rotation_2d(yaw)
    face = FACE_BY_NAME[p['candidate']]
    n = R @ np.asarray(face.normal, float)
    t = np.array([-n[1], n[0]])
    if n @ p['normal'] < .85:
        return None, 'face_changed', {}
    anchor = xy + R @ p['v5_material_anchor']
    error = anchor - eef[:2]
    lateral, normal = float(error @ t), float(error @ n)
    diag = dict(v5_tangent_error_m=lateral, v5_normal_error_m=normal,
                v5_live_normal_x=float(n[0]), v5_live_normal_y=float(n[1]))
    if abs(lateral) > .018 or abs(normal) > .025:
        return None, 'anchor_outside_envelope', diag
    # Positive normal command only: don't drag the T backwards. Modest preload
    # when behind the measured anchor; reduce forward speed if too far ahead.
    base = float(np.clip(gap*18., .055, .34))
    if recovery: base = float(np.clip(gap*18., .12, .24))
    forward = float(np.clip(base + 14.*normal, .035, .40))
    side = float(np.clip(24.*lateral, -.20, .20))
    command = n*forward + t*side
    if command @ p['normal'] <= 0:
        return None, 'reverse_command', diag
    next_xy = eef[:2] + .05*command
    if np.any(next_xy < [-.29,-.28]) or np.any(next_xy > [.25,.28]):
        return None, 'workspace', diag
    return command, '', diag


class MaterialTrackingController(ContactRecoveryController):
    TRACK_TURN_DEG = 4.

    def _track_enabled(self, p, yaw):
        return (p.get('candidate') in ('yp_stem_mid','yn_stem_mid')
                and 'v5_material_anchor' in p
                and (p.get('v5_tracking', False) or
                     abs(np.rad2deg(wrapped(yaw-p['start_yaw']))) >= self.TRACK_TURN_DEG))

    def _recover_command(self, p, xy, yaw, eef, gap):
        if self._track_enabled(p, yaw):
            cmd, why, diag = material_command(p, xy, yaw, eef, gap, recovery=True)
            p.update(diag)
            return cmd, why
        return super()._recover_command(p, xy, yaw, eef, gap)

    def _four_log_push(self, eef):
        p = self._four_active
        count = len(self.four_way_samples)
        super()._four_log_push(eef)
        if p is not None and p.get('joint') and len(self.four_way_samples)>count:
            self.four_way_samples[-1].update({k:p.get(k,0) for k in (
                'v5_tracking_steps','v5_tracking_start_step','v5_max_tangent_error_m',
                'v5_max_normal_error_m','v5_tracking_exit')})

    def action(self, obs):
        p = self._monitored_stroke() if self.large_route and self.wide_angle_route else None
        if p is None or not p.get('joint') or p['candidate'] not in ('yp_stem_mid','yn_stem_mid'):
            return super().action(obs)
        env = self.environment
        xy, yaw = angular_pose(env)
        eef = np.asarray(obs['robot0_eef_pos'],float)
        available,hit = tee_robot_contact(env)
        if available and hit and 'v5_material_anchor' not in p:
            # Keep the first verified material-frame anchor; never refresh it
            # while a finger slides along the edge. Freeze initial wrist target.
            p['v5_material_anchor'] = rotation_2d(-yaw) @ (eef[:2]-xy)
            p['v5_wrist_start'] = self.rotation.copy()
            p['v5_wrist_yaw_start'] = p['start_yaw']
        enabled = self._track_enabled(p,yaw)
        if enabled and not p.get('v5_tracking'):
            p['v5_tracking'] = True
            p['v5_tracking_start_step'] = int(env.timestep)
            self.joint_events.append(dict(event='material_tracking_start',step=int(env.timestep),attempt=p['joint_attempt'],candidate=p['candidate'],actual_yaw_deg=float(np.rad2deg(wrapped(yaw-p['start_yaw'])))))
        # The base controller owns stopping rules, joint replanning and contact
        # recovery. Only replace the command if it is STILL the same stroke.
        a = np.asarray(super().action(obs),float)
        if not enabled or self.phase!='push' or self._four_active is not p:
            return a
        gap = p['joint_motion_m'] - p.get('joint_actual_motion_m',0.)
        cmd, why, diag = material_command(p,xy,yaw,eef,gap,bool(p.get('joint_recovering')))
        p.update(diag)
        p['v5_max_tangent_error_m'] = max(p.get('v5_max_tangent_error_m',0.),abs(diag.get('v5_tangent_error_m',0.)))
        p['v5_max_normal_error_m'] = max(p.get('v5_max_normal_error_m',0.),abs(diag.get('v5_normal_error_m',0.)))
        if cmd is None:
            p['v5_tracking_exit'] = why
            p['feedback_stop_reason'] = 'joint_material_tracking_unsafe'
            p['feedback_remaining_at_stop_m'] = gap
            self.phase='plan'; self.feedback_stops+=1
            return np.asarray(super().action(obs),float)
        # Follow the material face orientation, preserving wrist branch rather
        # than asking wrist_rotation to choose a new 180-degree branch.
        rz = np.eye(3); rz[:2,:2] = rotation_2d(wrapped(yaw-p['v5_wrist_yaw_start']))
        desired = rz @ p['v5_wrist_start']
        site = env.robots[0].eef_site_id['right']
        current = env.sim.data.site_xmat[site].reshape(3,3)
        r = np.clip(Rotation.from_matrix(desired @ current.T).as_rotvec()/.5,-.5,.5)
        p['v5_tracking_steps'] = p.get('v5_tracking_steps',0)+1
        a[:2]=cmd; a[3:6]=r
        return a


class AdaptiveTranslationController(MaterialTrackingController):
    def __init__(self,*a,response_model=None,**kw):
        model=load_model(response_model)
        self._calibrated_planners={k:CalibratedPlanner(k,model) for k in ('standard','wide')}
        super().__init__(*a,**kw)
    @property
    def planner(self):return self._calibrated_planners['wide' if self.wide_angle_route else 'standard']
    def _adapt_joint(self,p,xy,yaw,eef):
        env=self.environment;snap=self.joint_snapshot
        if snap is None or not p.get('joint_contact') or int(env.timestep)>getattr(env,"expert_step_limit",400)-20:return
        seq=p['joint_sequence'];name=seq[0];n=p['normal'];motion=float((xy-p['joint_origin'])@n)
        if motion<.003 or motion-p.get('joint_last_adapt_m',0.)<.0025:return
        p['joint_last_adapt_m']=motion
        delta=rotation_2d(-snap['yaw'])@(xy-snap['xy']);turn=wrapped(yaw-snap['yaw'])
        prior=np.asarray(self.joint_rates.get(name,self.planner.PRIORS[name]))
        measured=np.r_[np.clip(delta/motion,-2.,2.),np.clip(turn/motion,-12.,12.)]
        # Equivalent prior travel: regularize short-transient yaw slopes.
        # X in particular must not sacrifice 80mm of unmet translation to a
        # momentary angular disturbance at the first few mm of a long stroke.
        trust=motion/(motion+(.025 if name[0]=='x' else .009))
        v=(1-trust)*prior+trust*measured
        rates={**self.planner.PRIORS,**self.joint_rates,name:v}
        q=np.r_[rotation_2d(-env.target_yaw)@(xy-p['joint_goal']),wrapped(yaw-env.target_yaw)]
        heading=wrapped(snap['yaw']-env.target_yaw)
        limit=min(CAPS[name],p['joint_original_motion']+.020);point=p['point'];end=point+(limit+.018)*n
        while limit>motion and (np.any(end<[-.29,-.28]) or np.any(end>[.25,.28])):
            limit-=.002;end=point+(limit+.018)*n
        upper=limit-motion
        if upper<.00015:return
        old=float(p['joint_motion_m']);guess=np.r_[max(0.,old-motion),p['joint_distances'][1:]]
        d=self.planner.live_tail(q,seq,upper,rates,heading,guess)
        if d is None:return
        new=float(np.clip(motion+d[0],max(motion,old-.008),max(motion,old+.008)))
        # For a terminal X contact, preserve forward travel while the remaining
        # X error is large. Apply only where the frozen normal actually reduces it.
        if len(seq)==1 and name[0]=='x':
            target_n=rotation_2d(-env.target_yaw)@n
            required=-q[0]*target_n[0]
            if required>.015 and abs(target_n[0])>.85:
                new=max(new,motion+min(upper,max(0.,required-.004)))
        new=float(np.clip(new,motion,limit))
        if abs(new-old)<.0006:return
        p['joint_motion_m']=new;p['joint_distances']=d.copy();p['joint_distances'][0]=new
        endpose,_=self.planner.rollout(q,seq,np.r_[new-motion,d[1:]],rates,heading)
        self.joint_events.append(dict(event='live_replan',attempt=p['joint_attempt'],step=int(env.timestep),candidate=name,sequence='>'.join(seq),actual_motion_m=motion,old_distance_mm=1000*old,new_distance_mm=1000*new,actual_yaw_deg=float(np.rad2deg(turn)),estimated_yaw_rate_deg_per_10mm=float(np.rad2deg(v[2]*.010)),predicted_iou=self.planner.predict_iou(endpose),distances_mm=';'.join(f'{1000*x:.2f}' for x in np.r_[new,d[1:]]),v6_response_trust=trust))


class StagedTranslationController(AdaptiveTranslationController):
    def __init__(self,*a,response_model=None,**kw):
        super().__init__(*a,response_model=response_model,**kw)
        model=load_model(response_model)
        self._calibrated_planners={k:StagedPlanner(k,model) for k in ('standard','wide')}
