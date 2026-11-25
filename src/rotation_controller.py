"""Rotation and shared robot-motion primitives."""
from __future__ import annotations

import numpy as np
import json
from copy import copy
from geometry import rotation_2d, angular_pose, wrapped, tee_robot_contact
from pathlib import Path
from dataclasses import dataclass
from scipy.spatial.transform import Rotation


class RepositioningPushTExpert:
    """Replan short pushes from the measured pose after each contact."""

    # Local contact points on exterior edges and their inward push normals.
    # The gripper is wider than the stem, so leave room at the corners.
    contacts = np.array([
        [[-0.060, -0.037], [1, 0]], [[-0.060, 0.000], [1, 0]],
        [[-0.060, 0.037], [1, 0]], [[0.060, 0.000], [-1, 0]],
        [[-0.043, -0.060], [0, 1]], [[-0.043, 0.060], [0, -1]],
        [[0.026, -0.0175], [0, 1]], [[0.026, 0.0175], [0, -1]],
        [[-0.025, -0.043], [-1, 0]], [[-0.025, 0.043], [-1, 0]],
    ])
    center_of_mass = np.array([-0.0176, 0.0])

    def __init__(self, environment, trial_environment=None):
        self.environment = environment
        self.trial_environment = trial_environment
        self.phase = "plan"
        self.hold_steps = 0
        self.phase_steps = 0
        self.last_contact = None
        self.last_error = None
        self.blocked = set()
        self.motion_by_contact = {}

    @staticmethod
    def pose_error(translation, angle):
        return np.linalg.norm(np.r_[translation, 0.10 * angle])

    def plan(self, position, rotation, target, angle, eef):
        error = self.pose_error(target - position, angle)
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        if self.last_contact is not None:
            old_position, old_yaw, stroke = self.start
            actual_local = rotation_2d(-old_yaw) @ (position - old_position) / stroke
            actual_turn = np.arctan2(np.sin(yaw - old_yaw), np.cos(yaw - old_yaw)) / stroke
            self.motion_by_contact[self.last_contact] = (
                np.clip(actual_local, -2, 2), np.clip(actual_turn, -45, 45)
            )
            if error >= self.last_error - 0.0015:
                self.blocked.add(self.last_contact)
            else:
                self.blocked.clear()
        if len(self.blocked) == len(self.contacts):
            self.blocked.clear()

        points = position + self.contacts[:, 0] @ rotation.T
        normals = self.contacts[:, 1] @ rotation.T
        # Approximate planar friction to rank feasible pushes; observed motion
        # after each stroke determines the next contact.
        arms = self.contacts[:, 0] - self.center_of_mass
        local_normals = self.contacts[:, 1]
        torque = arms[:, 0] * local_normals[:, 1] - arms[:, 1] * local_normals[:, 0]
        strokes = np.clip(error * np.array([0.32, 0.68]), 0.003, 0.050)
        if self.trial_environment is not None:
            strokes = np.unique(np.clip(error * np.array([0.20, 0.6, 1.1]), 0.003, 0.075))
        choices = []
        for index, (point, normal) in enumerate(zip(points, normals)):
            if index in self.blocked:
                continue
            approach = point - 0.024 * normal
            # Very far approach poses can be outside Panda's working region.
            if np.any(approach < [-0.26, -0.25]) or np.any(approach > [0.21, 0.25]):
                continue
            for stroke in strokes:
                if index in self.motion_by_contact:
                    local_motion, angular_rate = self.motion_by_contact[index]
                    turn = stroke * angular_rate
                    estimated = position + stroke * (rotation @ local_motion)
                else:
                    turn = stroke * torque[index] / 0.0024
                    center = self.center_of_mass @ rotation.T
                    estimated = position + 0.80 * stroke * normal + center - rotation_2d(turn) @ center
                next_angle = np.arctan2(np.sin(angle - turn), np.cos(angle - turn))
                cost = self.pose_error(target - estimated, next_angle)
                cost += 0.012 * np.linalg.norm(eef[:2] - approach)
                choices.append((cost, index, approach, point + stroke * normal, stroke))
        if not choices:
            self.blocked.clear()
            self.last_contact = None
            return False
        choices.sort(key=lambda choice: choice[0])
        if self.trial_environment is not None:
            chosen = self.choose_simulated_push(choices, normals)
        else:
            chosen = choices[0]
        _, index, self.approach, self.end, stroke = chosen
        self.start = position.copy(), yaw, stroke
        self.rotation = self.wrist_rotation(normals[index])
        self.last_error, self.last_contact = error, index
        return True

    def wrist_rotation(self, normal):
        from scipy.spatial.transform import Rotation

        heading = np.arctan2(normal[1], normal[0])
        site = self.environment.robots[0].eef_site_id["right"]
        current = self.environment.sim.data.site_xmat[site].reshape(3, 3)
        orientations = []
        for candidate in (heading, heading + np.pi):
            desired = np.eye(3)
            desired[:2, :2] = rotation_2d(candidate + np.pi / 2)
            desired = desired @ np.diag([1, -1, -1])
            # Keep Panda's wrist near the same branch as the original expert.
            wrist = abs(np.arctan2(np.sin(candidate), np.cos(candidate)))
            change = Rotation.from_matrix(desired @ current.T).magnitude()
            orientations.append((change + 1.2 * wrist, desired))
        return min(orientations, key=lambda option: option[0])[1]

    def choose_simulated_push(self, choices, normals):
        import mujoco

        from lift_common import synchronized_observation

        real = self.environment
        trial = self.trial_environment
        state_type = mujoco.mjtState.mjSTATE_INTEGRATION
        real_model = real.sim.model._model
        trial_model = trial.sim.model._model
        trial_data = trial.sim.data._data
        state = np.empty(mujoco.mj_stateSize(real_model, state_type))
        mujoco.mj_getState(real_model, real.sim.data._data, state, state_type)
        trial_model.body_pos[trial.target_body_id] = real_model.body_pos[real.target_body_id]
        trial_model.body_quat[trial.target_body_id] = real_model.body_quat[real.target_body_id]
        trial.target_yaw = real.target_yaw
        real_gripper = real.robots[0].gripper["right"]
        trial_gripper = trial.robots[0].gripper["right"]
        # Try the best estimated stroke for each face, plus short strokes near
        # the goal. The simulated outcomes determine which push to execute.
        per_contact = {}
        for choice in choices:
            per_contact.setdefault(choice[1], choice)
        candidates = list(per_contact.values())
        if self.last_error is None or self.last_error < 0.065:
            candidates.extend(choice for choice in choices[:4] if per_contact[choice[1]] is not choice)
        remaining_steps = getattr(real,"expert_step_limit",400) - real.timestep
        if remaining_steps <= 0:
            return choices[0]
        best_score, best_choice = float("inf"), choices[0]
        for choice in candidates:
            mujoco.mj_setState(trial_model, trial_data, state, state_type)
            trial_gripper.current_action = real_gripper.current_action.copy()
            trial.cur_time, trial.timestep, trial.done = real.cur_time, real.timestep, False
            rollout = copy(self)
            rollout.environment = trial
            rollout.trial_environment = None
            rollout.phase, rollout.phase_steps, rollout.hold_steps = "approach", 0, 0
            rollout.approach, rollout.end = choice[2:4]
            rollout.rotation = self.wrist_rotation(normals[choice[1]])
            observation = synchronized_observation(trial)
            stable = 0
            for step in range(min(remaining_steps, 115)):
                trial.step(rollout.action(observation))
                observation = synchronized_observation(trial)
                metrics = trial.task_metrics()
                valid = metrics["upright"] and metrics["on_table"]
                stable = stable + 1 if valid and metrics["iou"] >= 0.85 else 0
                if stable >= 20:
                    return choice
                if rollout.phase == "plan" or not valid:
                    break
            if not valid:
                continue
            # Penalize time and favor alignment near the target, where small
            # contact-model errors can matter more than translations.
            angle = np.deg2rad(metrics["yaw_error_degrees"])
            score = self.pose_error(metrics["position_error"], angle)
            score += 0.00030 * (step + 1) - 0.015 * metrics["iou"]
            if score < best_score:
                best_score, best_choice = score, choice
        return best_choice

    def action(self, observation):
        from scipy.spatial.transform import Rotation

        env = self.environment
        sim = env.sim.data
        eef = observation["robot0_eef_pos"]
        position = sim.body_xpos[env.cube_body_id, :2].copy()
        rotation = sim.body_xmat[env.cube_body_id].reshape(3, 3)[:2, :2]
        target = sim.body_xpos[env.target_body_id, :2]
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        angle = np.arctan2(np.sin(env.target_yaw - yaw), np.cos(env.target_yaw - yaw))
        site = env.robots[0].eef_site_id["right"]
        current = sim.site_xmat[site].reshape(3, 3)
        metrics = env.task_metrics()
        release_iou = 0.85 if self.phase == "release" else 0.875
        if metrics["iou"] >= release_iou and metrics["on_table"] and metrics["upright"]:
            self.phase = "release"
        elif self.phase == "release":
            self.hold_steps = 0
            self.phase = "plan"

        if self.phase == "plan":
            if eef[2] < 0.885:
                return np.array([0, 0, 0.75, 0, 0, 0, -1.0])
            if not self.plan(position, rotation, target, angle, eef):
                return np.array([0, 0, 0, 0, 0, 0, -1.0])
            self.phase, self.phase_steps = "approach", 0

        if self.phase == "release":
            self.hold_steps += 1
            desired = np.r_[eef[:2], 0.95]
            rotation_action = np.zeros(3)
        else:
            self.phase_steps += 1
            rotation_action = Rotation.from_matrix(self.rotation @ current.T).as_rotvec() / 0.5
            if self.phase == "approach":
                desired = np.r_[self.approach, 0.90]
                if np.linalg.norm(eef - desired) < 0.010 and np.linalg.norm(rotation_action) < 0.25:
                    self.phase, self.phase_steps = "descend", 0
                elif self.phase_steps >= 45:
                    self.phase = "plan"
            elif self.phase == "descend":
                desired = np.r_[self.approach, 0.824]
                if np.linalg.norm(eef - desired) < 0.007:
                    self.phase, self.phase_steps = "push", 0
                elif self.phase_steps >= 30:
                    self.phase = "plan"
            else:
                desired = np.r_[self.end, 0.824]
                if np.linalg.norm(eef - desired) < 0.004 or self.phase_steps >= 28:
                    self.phase = "plan"
        translation = np.clip((desired - eef) * 18, -0.70, 0.70)
        return np.r_[translation, np.clip(rotation_action, -0.5, 0.5), -1.0]

def wrap_angle(angle):
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


class Release85PushTExpert(RepositioningPushTExpert):
    """Original planner with only release-entry threshold corrected."""

    def action(self, observation):
        metrics = self.environment.task_metrics()
        if (metrics["iou"] >= self.environment.overlap_threshold
                and metrics["upright"] and metrics["on_table"]):
            self.phase = "release"
        return super().action(observation)


class RotateFeedbackPushTExpert(Release85PushTExpert):
    """First correct yaw by eccentric pushes; then use original translation.

    The rotation planner only runs when mode == 'rotate'.  No hard yaw guards,
    rotation penalties, or fixed short-strokes are added to original translation.
    A limited number of rotation attempts and time budget avoid endless loops.
    """

    def __init__(
        self, environment, trial_environment=None, *,
        rotate_start_deg=4.0,
        rotate_goal_deg=2.5,
        reenter_deg=8.0,
        rotate_budget_steps=170,
        max_rotate_pushes=4,
        min_rotate_stroke_m=0.012,
        max_rotate_stroke_m=0.045,
        no_progress_deg=0.25,
        max_consecutive_failures=2,
    ):
        if trial_environment is not None:
            raise ValueError("Feedback rotation currently requires trial_environment=None")
        if not (0 < rotate_goal_deg < rotate_start_deg < reenter_deg < 180):
            raise ValueError("Require 0 < rotate_goal_deg < rotate_start_deg < reenter_deg < 180")
        if rotate_budget_steps < 0 or max_rotate_pushes < 0:
            raise ValueError("Rotation budgets must be nonnegative")
        if not (0 < min_rotate_stroke_m <= max_rotate_stroke_m):
            raise ValueError("Invalid rotation stroke limits")
        if no_progress_deg < 0 or max_consecutive_failures < 1:
            raise ValueError("Invalid feedback thresholds")
        super().__init__(environment, trial_environment=None)
        self.rotate_start_deg = float(rotate_start_deg)
        self.rotate_goal_deg = float(rotate_goal_deg)
        self.reenter_deg = float(reenter_deg)
        self.rotate_budget_steps = int(rotate_budget_steps)
        self.max_rotate_pushes = int(max_rotate_pushes)
        self.min_rotate_stroke_m = float(min_rotate_stroke_m)
        self.max_rotate_stroke_m = float(max_rotate_stroke_m)
        self.no_progress_deg = float(no_progress_deg)
        self.max_consecutive_failures = int(max_consecutive_failures)
        self._reset_rotation_state()

    def _reset_rotation_state(self):
        self.mode = None                       # rotate | translate
        self.initial_yaw_error_deg = None
        self.rotate_entry_yaw_deg = None
        self.rotate_exit_yaw_deg = None
        self.rotate_best_yaw_deg = float("inf")
        self.rotate_completed = False
        self.rotate_budget_exhausted = False
        self.rotate_no_progress_exit = False
        self.rotate_unreachable_exit = False
        self.rotate_steps = 0
        self.rotate_pushes = 0                 # completed rotation attempts
        self.rotate_plans = 0
        self.translate_plans = 0
        self.mode_switches = 0
        self.rotate_effective_pushes = 0
        self.rotate_wrong_direction_pushes = 0
        self.rotate_no_motion_pushes = 0
        self.rotate_approach_failures = 0
        self.rotate_goal_reached_during_push = 0
        self.translate_yaw_max_deg = 0.0
        self.first_translate_yaw_deg = None
        self._rotate_consecutive_failures = 0
        self._rotate_blocked = set()
        self._rotate_motion_rate = {}         # contact index -> observed radians/m
        self._rotate_motion_count = {}
        self._planned_rotation_push = None
        self._active_rotation_push = None
        self.rotation_samples = []
        self._last_selected_rotation_contact = None

    def reset(self):
        """Reset at each episode (a new instance per episode also works)."""
        RepositioningPushTExpert.__init__(self, self.environment, trial_environment=None)
        self._reset_rotation_state()

    def _pose(self):
        data = self.environment.sim.data
        position = data.body_xpos[self.environment.cube_body_id, :2].copy()
        rot3 = data.body_xmat[self.environment.cube_body_id].reshape(3, 3)
        yaw = float(np.arctan2(rot3[1, 0], rot3[0, 0]))
        error = wrap_angle(self.environment.target_yaw - yaw)
        return position, rot3[:2, :2].copy(), yaw, error

    def _set_mode(self, new_mode, yaw_deg):
        if self.mode == new_mode:
            return
        if self.mode is not None:
            self.mode_switches += 1
        self.mode = new_mode
        # The original planner's error/block list is only meaningful for its
        # translation objective, never transfer a rotation attempt into it.
        self.last_contact = None
        self.last_error = None
        self.blocked.clear()
        self.phase_steps = 0
        if new_mode == "rotate":
            self.rotate_entry_yaw_deg = float(yaw_deg)
            self._rotate_consecutive_failures = 0
            self._rotate_blocked.clear()
        elif new_mode == "translate":
            self.rotate_exit_yaw_deg = float(yaw_deg)
            if self.first_translate_yaw_deg is None:
                self.first_translate_yaw_deg = float(yaw_deg)

    def _finish_push_feedback(self, obs_eef):
        """Record measured motion ONLY after a rotation push has actually ended.

        This runs before the next plan, with the previous env.step() completed.
        """
        pending = self._active_rotation_push
        if pending is None:
            return
        position, _, yaw, _ = self._pose()
        turn = wrap_angle(yaw - pending["start_yaw"])
        requested = pending["target_angle"]
        before = abs(requested)
        after = abs(wrap_angle(requested - turn))
        progress = before - after
        contact = pending["contact"]
        normal = pending["normal"]
        point = pending["point"]
        # Estimate the actual gripper motion *past* the geometric contact
        # plane. A planned stroke may not be physically completed.
        passed_contact = max(0.0, float(np.dot(obs_eef[:2] - point, normal)))
        effective_stroke = min(passed_contact, pending["stroke"] + 0.012)
        reached = effective_stroke >= 0.003
        if reached and abs(turn) >= np.deg2rad(0.15):
            measured_rate = float(np.clip(turn / max(effective_stroke, 0.003), -45, 45))
            n = self._rotate_motion_count.get(contact, 0)
            if n:
                self._rotate_motion_rate[contact] = (
                    0.55 * self._rotate_motion_rate[contact] + 0.45 * measured_rate
                )
            else:
                self._rotate_motion_rate[contact] = measured_rate
            self._rotate_motion_count[contact] = n + 1
        wrong_direction = turn * requested < -np.deg2rad(0.15) * abs(requested)
        useful = progress >= np.deg2rad(self.no_progress_deg)
        if useful:
            self.rotate_effective_pushes += 1
            self._rotate_consecutive_failures = 0
            self._rotate_blocked.clear()
        else:
            self._rotate_consecutive_failures += 1
            self._rotate_blocked.add(contact)
            if wrong_direction:
                self.rotate_wrong_direction_pushes += 1
            if abs(turn) < np.deg2rad(0.15):
                self.rotate_no_motion_pushes += 1
        self.rotate_pushes += 1
        self.rotation_samples.append(dict(
            attempt=self.rotate_pushes,
            contact=contact,
            planned_stroke_m=float(pending["stroke"]),
            effective_stroke_m=effective_stroke,
            predicted_turn_deg=float(np.rad2deg(pending["predicted_turn"])),
            actual_turn_deg=float(np.rad2deg(turn)),
            yaw_error_before_deg=float(np.rad2deg(before)),
            yaw_error_after_deg=float(np.rad2deg(after)),
            progress_deg=float(np.rad2deg(progress)),
            object_translation_m=float(np.linalg.norm(position - pending["start_xy"])),
            useful=bool(useful),
            wrong_direction=bool(wrong_direction),
            reached_contact=bool(reached),
        ))
        self.rotate_best_yaw_deg = min(self.rotate_best_yaw_deg, np.rad2deg(after))
        self._active_rotation_push = None
        self._planned_rotation_push = None

    def _plan_rotation(self, position, rotation, angle, eef):
        """Pick eccentric contact with correct *predicted* rotation direction.

        Once a contact has produced a measurable turn, its empirical rate is
        used for ranking and stroke selection instead of the torque heuristic.
        """
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        arms = self.contacts[:, 0] - self.center_of_mass
        local_normals = self.contacts[:, 1]
        torque = arms[:, 0] * local_normals[:, 1] - arms[:, 1] * local_normals[:, 0]
        points = position + self.contacts[:, 0] @ rotation.T
        normals = self.contacts[:, 1] @ rotation.T
        candidates = []
        for index, (point, normal) in enumerate(zip(points, normals)):
            if index in self._rotate_blocked:
                continue
            approach = point - 0.024 * normal
            if np.any(approach < [-0.26, -0.25]) or np.any(approach > [0.21, 0.25]):
                continue
            geometric_rate = float(torque[index] / 0.0024)
            angular_rate = self._rotate_motion_rate.get(index, geometric_rate)
            # Direction must agree with the requested rotation.
            if angular_rate * angle <= 0 or abs(angular_rate) < 0.5:
                continue
            rough_stroke = abs(angle) / abs(angular_rate)
            lengths = np.unique(np.clip(
                [0.65 * rough_stroke, rough_stroke, 1.35 * rough_stroke],
                self.min_rotate_stroke_m, self.max_rotate_stroke_m
            ))
            for stroke in lengths:
                turn = float(angular_rate * stroke)
                predicted_error = abs(wrap_angle(angle - turn))
                # Conservative estimate of displacement/remaining reachability.
                center = self.center_of_mass @ rotation.T
                estimated = (position + 0.80 * stroke * normal
                             + center - rotation_2d(turn) @ center)
                # Strongly avoid pushing T outside the ordinary workspace.
                out_of_bounds = float(np.linalg.norm(
                    estimated - np.clip(estimated, [-0.19, -0.19], [0.15, 0.19])
                ))
                # Avoid large position drift, but yaw is the primary objective.
                drift = float(np.linalg.norm(estimated - position))
                wrist_cost = float(np.linalg.norm(eef[:2] - approach))
                cost = (predicted_error + 1.5 * out_of_bounds
                        + 0.22 * drift + 0.025 * wrist_cost
                        + 0.0005 * stroke)
                # Unverified contacts can be tried, but prefer measured ones
                # when predicted effects are otherwise similar.
                if index not in self._rotate_motion_rate:
                    cost += 0.004
                candidates.append((cost, index, approach.copy(),
                                   (point + stroke * normal).copy(),
                                   float(stroke), normal.copy(), point.copy(), turn))
        if not candidates:
            return False
        candidates.sort(key=lambda candidate: candidate[0])
        _, index, approach, end, stroke, normal, point, turn = candidates[0]
        self.approach = approach
        self.end = end
        self.rotation = self.wrist_rotation(normal)
        self._planned_rotation_push = dict(
            contact=int(index), stroke=stroke, normal=normal,
            point=point, predicted_turn=turn,
        )
        self._last_selected_rotation_contact = index
        # Important: no cross-contamination of the parent translation learner.
        self.last_contact = None
        self.rotate_plans += 1
        return True

    def plan(self, position, rotation, target, angle, eef):
        if self.mode == "rotate":
            return self._plan_rotation(position, rotation, angle, eef)
        self.translate_plans += 1
        # Deliberately UNCHANGED original translation planning.
        return super().plan(position, rotation, target, angle, eef)

    def action(self, observation):
        _, _, _, error = self._pose()
        yaw_error_deg = abs(float(np.rad2deg(error)))
        if self.mode is None:
            self.initial_yaw_error_deg = yaw_error_deg
            if (yaw_error_deg > self.rotate_start_deg
                    and self.rotate_budget_steps and self.max_rotate_pushes):
                self._set_mode("rotate", yaw_error_deg)
            else:
                self._set_mode("translate", yaw_error_deg)
                if yaw_error_deg <= self.rotate_goal_deg:
                    self.rotate_completed = True
        if self.mode == "rotate":
            self.rotate_best_yaw_deg = min(self.rotate_best_yaw_deg, yaw_error_deg)

        metrics = self.environment.task_metrics()
        success = bool(metrics["iou"] >= self.environment.overlap_threshold
                       and metrics["upright"] and metrics["on_table"])

        # A live rotation push may reach the goal before the endpoint. End it
        # gracefully by raising the arm; finish measuring on the next call.
        if (self.mode == "rotate" and self.phase == "push" and not success
                and yaw_error_deg <= self.rotate_goal_deg):
            self.rotate_goal_reached_during_push += 1
            self.phase = "plan"
            self.phase_steps = 0

        # A rotation push ends when the inherited low-level controller returns
        # to 'plan' (or when success triggers release). Read AFTER the env.step.
        if (self._active_rotation_push is not None
                and (self.phase == "plan" or success or self.phase == "release")):
            self._finish_push_feedback(observation["robot0_eef_pos"])

        if self.phase == "plan" and not success:
            if self.mode == "rotate":
                if yaw_error_deg <= self.rotate_goal_deg:
                    self.rotate_completed = True
                    self._set_mode("translate", yaw_error_deg)
                elif (self.rotate_steps >= self.rotate_budget_steps
                      or self.rotate_pushes >= self.max_rotate_pushes):
                    self.rotate_budget_exhausted = True
                    self._set_mode("translate", yaw_error_deg)
                elif self._rotate_consecutive_failures >= self.max_consecutive_failures:
                    self.rotate_no_progress_exit = True
                    self._set_mode("translate", yaw_error_deg)
                else:
                    # If all feasible opposite-sign faces were blocked by
                    # failures, allow a fresh choice from other learned faces.
                    available = set(range(len(self.contacts))) - self._rotate_blocked
                    if not available:
                        self._rotate_blocked.clear()
            elif (self.mode == "translate" and yaw_error_deg >= self.reenter_deg
                  and self.rotate_steps < self.rotate_budget_steps
                  and self.rotate_pushes < self.max_rotate_pushes
                  and self._rotate_consecutive_failures < self.max_consecutive_failures):
                self._set_mode("rotate", yaw_error_deg)

        if self.mode == "rotate" and not success:
            self.rotate_steps += 1
        if self.mode == "translate":
            self.translate_yaw_max_deg = max(self.translate_yaw_max_deg, yaw_error_deg)

        previous_phase = self.phase
        # This calls the inherited motion controller. In translate mode, its
        # plan() dispatches to super().plan unchanged (see above).
        commanded = super().action(observation)
        if (self.mode == "rotate" and self._planned_rotation_push is not None
                and self._active_rotation_push is None
                and previous_phase != "push" and self.phase == "push"):
            # Only NOW store the object's pose (after approach/descend).
            position, _, yaw, requested = self._pose()
            self._active_rotation_push = dict(
                **self._planned_rotation_push,
                start_xy=position.copy(),
                start_yaw=yaw,
                target_angle=requested,
            )
        # Reaching the approach/descend timeout without entering push is a
        # failed contact attempt; try another face next time rather than
        # retrying the same unreachable wrist pose indefinitely.
        if (self.mode == "rotate" and previous_phase in ("approach", "descend")
                and self.phase == "plan" and self._active_rotation_push is None
                and self._planned_rotation_push is not None):
            self.rotate_approach_failures += 1
            self._rotate_consecutive_failures += 1
            self._rotate_blocked.add(self._planned_rotation_push["contact"])
            self._planned_rotation_push = None
        # If no rotation contact is feasible, parent's plan() returns False.
        # Avoid idling in rotate forever; switch on next safe planning step.
        if (self.mode == "rotate" and self.phase == "plan"
                and self._planned_rotation_push is None
                and self._active_rotation_push is None
                and self.rotate_plans == 0 and self.rotate_steps > 3):
            self.rotate_unreachable_exit = True
            self._set_mode("translate", yaw_error_deg)
        return commanded


def heading_bin(angle, bins):
    return min(bins-1, int(((float(angle) + np.pi) % (2*np.pi)) * bins / (2*np.pi)))


class CalibratedRotatePushTExpert(RotateFeedbackPushTExpert):
    def __init__(self, environment, trial_environment=None, *,
                 model_path="contact_model.json", **kwargs):
        super().__init__(environment, trial_environment=trial_environment,
                         max_consecutive_failures=3, **kwargs)
        path = Path(model_path)
        with path.open("r", encoding="utf-8") as f:
            model = json.load(f)
        if model.get("version") != 1 or not isinstance(model.get("yaw_bins"), int):
            raise ValueError("Unsupported contact model: use the supplied compatible contact_model.json")
        env_name = ("random" if type(environment).__name__ == "PandaPushTRandom"
                    else "independent" if getattr(environment, "independent_poses", False)
                    else "baseline")
        if model.get("environment") not in (None, env_name):
            raise ValueError(f"Contact model calibrated for {model['environment']}, "
                             f"but environment is {env_name}")
        self.model_path = str(path)
        self.model_bins = int(model["yaw_bins"])
        self.calibration = {}
        for record in model["entries"]:
            if record.get("reliable"):
                key = int(record["contact"]), int(record["yaw_bin"])
                self.calibration[key] = record
        self.no_calibrated_contact = False
        self.calibrated_plans = 0
        self.calibrated_contacts_used = set()
        self.last_calibration_bin = None

    def _plan_rotation(self, position, rotation, angle, eef):
        # Rotation selection ONLY. The inherited translation planner is untouched.
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        bin_id = heading_bin(yaw, self.model_bins)
        self.last_calibration_bin = bin_id
        points = position + self.contacts[:, 0] @ rotation.T
        normals = self.contacts[:, 1] @ rotation.T
        options = []
        for index, (point, normal) in enumerate(zip(points, normals)):
            if index in self._rotate_blocked:
                continue
            calibration = self.calibration.get((index, bin_id))
            if calibration is None:
                continue
            rate = float(calibration["rate_rad_per_m"])
            if abs(rate) < 0.35 or rate * angle <= 0:
                continue
            # Blend only *this episode's* observed rate if compatible with
            # calibration. Rejected/contradictory pushes are blocked separately.
            online_n = self._rotate_motion_count.get(index, 0)
            if online_n:
                online = float(self._rotate_motion_rate[index])
                if online * rate < 0:
                    continue
                rate = 0.65 * rate + 0.35 * online
            approach = point - 0.024 * normal
            if np.any(approach < [-0.26, -0.25]) or np.any(approach > [0.21, 0.25]):
                continue
            ideal = abs(angle / rate)
            strokes = np.unique(np.clip([0.55 * ideal, 0.80 * ideal, ideal],
                                        self.min_rotate_stroke_m, self.max_rotate_stroke_m))
            for stroke in strokes:
                expected_turn = float(rate * stroke)
                remaining = abs(wrap_angle(angle - expected_turn))
                drift = float(calibration["translation_per_stroke"]) * float(stroke)
                estimated = position + 0.8 * stroke * normal
                outside = float(np.linalg.norm(
                    estimated - np.clip(estimated, [-0.19, -0.19], [0.15, 0.19])
                ))
                # Main objective = yaw error; allow some translation for rotation.
                cost = (remaining + 0.25 * drift + 1.5 * outside
                        + 0.018 * np.linalg.norm(eef[:2] - approach)
                        + 0.02 * (1.0 - float(calibration["sign_consistency"])))
                options.append((cost, int(index), approach.copy(),
                                (point + stroke * normal).copy(), float(stroke),
                                normal.copy(), point.copy(), expected_turn))
        if not options:
            # Do NOT guess a torque direction if no calibrated face is valid.
            # Fall back promptly to the proven original translation planner.
            self.no_calibrated_contact = True
            self.rotate_unreachable_exit = True
            self.rotate_steps = self.rotate_budget_steps
            return False
        options.sort(key=lambda x: x[0])
        _, index, self.approach, self.end, stroke, normal, point, turn = options[0]
        self.rotation = self.wrist_rotation(normal)
        self._planned_rotation_push = dict(contact=index, stroke=stroke,
                                           normal=normal, point=point,
                                           predicted_turn=turn)
        self._last_selected_rotation_contact = index
        self.last_contact = None
        self.rotate_plans += 1
        self.calibrated_plans += 1
        self.calibrated_contacts_used.add(index)
        return True

    def _finish_push_feedback(self, obs_eef):
        old_n = len(self.rotation_samples)
        super()._finish_push_feedback(obs_eef)
        if len(self.rotation_samples) == old_n:
            return
        last = self.rotation_samples[-1]
        # Wrong sign and ineffective pushes are disabled for this episode.
        # A calibration from other episodes is evidence, not a guarantee.
        if (last["wrong_direction"] or not last["reached_contact"]
                or abs(last["actual_turn_deg"]) < 0.35):
            self._rotate_blocked.add(int(last["contact"]))


@dataclass(frozen=True)
class Contact:
    name: str
    xy: tuple
    normal: tuple


# Long lever first; also allow an opposite side of the crossbar to produce
# the same torque but a different translation. All points lie on the outline.
CONTACTS = (
    Contact('stem_tip_bottom', (.049, -.0175), (0., 1.)),
    Contact('stem_tip_top', (.049, .0175), (0., -1.)),
    Contact('stem_side_bottom', (.035, -.0175), (0., 1.)),
    Contact('stem_side_top', (.035, .0175), (0., -1.)),
    Contact('bar_top_far', (-.053, .060), (0., -1.)),
    Contact('bar_bottom_far', (-.053, -.060), (0., 1.)),
    Contact('bar_left_lower', (-.060, -.043), (1., 0.)),
    Contact('bar_left_upper', (-.060, .043), (1., 0.)),
)
COM = np.array([-.0176, 0.])
APPROACH_GAP = .024
PUSH_Z = .824
HIGH_Z = .91
LOW = np.array([-.26, -.25])
HIGH = np.array([.21, .25])


def local_lever(contact):
    r = np.asarray(contact.xy) - COM
    n = np.asarray(contact.normal)
    return float(r[0]*n[1] - r[1]*n[0])


def face_world(contact, pos, yaw):
    r = rotation_2d(yaw)
    return pos + r @ contact.xy, r @ contact.normal


def pick_contact(pos, yaw, error, eef, target, blocked, response):
    """Rank candidates by yaw efficiency and projected drift, not only torque.

    Prior rate is a *heuristic*; measured response is rad/metre of actual
    object translation. Negative measured signs are not ignored.
    """
    options = []
    basis = rotation_2d(yaw)
    for contact in CONTACTS:
        if contact.name in blocked:
            continue
        p, n = face_world(contact, pos, yaw)
        approach = p - APPROACH_GAP*n
        if np.any(approach < LOW) or np.any(approach > HIGH):
            continue
        prior = local_lever(contact)/.0035
        rate, local_dir = response.get(contact.name, (prior, np.asarray(contact.normal)))
        if rate*error <= 0 or abs(rate) < 1.:
            continue
        # Evaluate a short, hypothetical 20 mm object motion. Never use this
        # estimate as the stopping condition. Measured yaw decides when to stop.
        move = min(.020, abs(error/rate))
        delta = basis @ local_dir * move
        gain = abs(error) - abs(wrapped(error-rate*move))
        away = max(0., np.linalg.norm(target-pos-delta)-np.linalg.norm(target-pos))
        projected = pos+delta
        outside = np.linalg.norm(projected-np.clip(projected,[-.20,-.21],[.16,.21]))
        # High angular gain per displacement, with drift and reach as penalties.
        score = -gain + 3.*away + 8.*outside + .12*np.linalg.norm(eef[:2]-approach)
        options.append((float(score), contact, approach))
    return min(options, key=lambda x:x[0]) if options else None


class EccentricRotation:
    """Standalone closed-gripper rotation, with feedback through all phases.

    Contact is not switched in the middle of a stroke. During servo, the local
    material contact target and push normal rotate with the measured Tee pose.
    If the current contact is ineffective, retreat/lift before trying a new one.
    """
    def __init__(self, environment, *, goal_deg=2.5, budget_steps=240, max_attempts=5,
                 close_steps=24):
        if not 0 < goal_deg < 10 or budget_steps < 80 or max_attempts < 1 or close_steps < 1:
            raise ValueError('Invalid rotation limits')
        self.environment = environment
        self.goal_deg = float(goal_deg)
        self.budget_steps = int(budget_steps)
        self.max_attempts = int(max_attempts)
        self.close_steps = int(close_steps)
        self.reset()

    def reset(self):
        self.phase = 'raise'
        self.phase_steps = 0
        self.steps = 0
        self.closed = False
        self.close_count = 0
        self.done = False
        self.completed = False
        self.exit_reason = ''
        self.attempts = 0
        self.samples = []
        self.blocked = set()
        self.response = {}
        self.active = None
        self.initial_xy = None
        self.best_error_deg = float('inf')
        self._abort = ''

    def _phase(self, phase):
        self.phase, self.phase_steps = phase, 0

    def _goal(self):
        return self.environment.sim.data.body_xpos[self.environment.target_body_id, :2].copy()

    def _error(self, yaw):
        return wrapped(self.environment.target_yaw-yaw)

    def _move(self, eef, desired, wrist=None, cap=.65, gripper=1.):
        xyz = np.clip((np.asarray(desired)-eef)*18., -cap, cap)
        rot = np.zeros(3)
        if wrist is not None:
            site = self.environment.robots[0].eef_site_id['right']
            current = self.environment.sim.data.site_xmat[site].reshape(3,3)
            rot = np.clip(Rotation.from_matrix(wrist @ current.T).as_rotvec()/.5, -.5, .5)
        return np.r_[xyz, rot, gripper]

    def _finish(self, reason, error, valid):
        self.completed = bool(valid and abs(np.rad2deg(error)) <= self.goal_deg)
        self.exit_reason = 'aligned' if self.completed else reason
        self.done = True
        self._phase('done')

    def _retreat(self, eef, yaw, reason):
        a = self.active
        if a is not None:
            a['stop_reason'] = reason
            _, n = face_world(a['contact'], a['start_xy'], yaw)
            a['retreat_target'] = np.r_[eef[:2]-.015*n, eef[2]]
        self._phase('retreat' if a is not None and eef[2]<.885 else 'lift')

    def _finalize_attempt(self, pos, yaw, eef):
        a = self.active
        if a is None:
            return
        angle = self._error(yaw)
        turn = wrapped(yaw-a['start_yaw'])
        delta = pos-a['start_xy']
        dist = float(np.linalg.norm(delta))
        progress = abs(a['start_error'])-abs(angle)
        goal = self._goal()
        before, after = np.linalg.norm(goal-a['start_xy']), np.linalg.norm(goal-pos)
        name = a['contact'].name
        # Only learn when there was measurable motion; no-contact is not a
        # zero-rate calibration. Use complete stroke incl. retreat for ranking.
        if dist >= .001 and abs(turn) >= np.deg2rad(.25):
            local_dir = rotation_2d(-a['start_yaw']) @ delta / dist
            rate = float(np.clip(turn/dist, -80, 80))
            if name in self.response:
                old_rate, old_dir = self.response[name]
                rate, local_dir = .25*old_rate+.75*rate, .25*old_dir+.75*local_dir
            self.response[name] = (rate, local_dir)
        if progress < np.deg2rad(.75) or after-before > .035:
            self.blocked.add(name)
        else:
            # Keep the same effective contact eligible on the next stroke.
            self.blocked.discard(name)
        self.samples.append(dict(attempt=self.attempts, contact=name,
            geometric_lever_m=local_lever(a['contact']),
            initial_signed_error_deg=float(np.rad2deg(a['start_error'])),
            final_signed_error_deg=float(np.rad2deg(angle)),
            net_turn_deg=float(np.rad2deg(turn)),
            net_progress_deg=float(np.rad2deg(progress)),
            net_translation_m=dist, goal_distance_before_m=float(before),
            goal_distance_after_m=float(after),
            contact_api_available=a['contact_api_available'],
            contact_frames=a['contact_frames'], motion_evidence=a['motion_evidence'],
            seek_steps=a['seek_steps'], servo_steps=a['servo_steps'],
            contact_acquired=a['contact_acquired'], stop_reason=a.get('stop_reason',''),
            steps_used=self.steps-a['start_step'],
            object_path_m=a['object_path_m'], eef_path_m=a['eef_path_m']))
        self.active = None

    def action(self, observation):
        env = self.environment
        eef = np.asarray(observation['robot0_eef_pos'], float)
        pos, yaw = angular_pose(env)
        angle = self._error(yaw)
        err = abs(float(np.rad2deg(angle)))
        m = env.task_metrics()
        valid = bool(m['upright'] and m['on_table'])
        if self.initial_xy is None:
            self.initial_xy = pos.copy()
        self.best_error_deg = min(self.best_error_deg,err)
        if self.done:
            return np.r_[np.zeros(6), 1. if self.closed else -1.]
        self.steps += 1
        self.phase_steps += 1
        a = self.active
        if a is not None:
            a['object_path_m'] += float(np.linalg.norm(pos-a['last_xy']))
            a['eef_path_m'] += float(np.linalg.norm(eef[:2]-a['last_eef']))
            a['last_xy'], a['last_eef'] = pos.copy(), eef[:2].copy()
            if self.phase in ('descend','seek','servo'):
                available, hit = tee_robot_contact(env)
                a['contact_api_available'] |= bool(available)
                a['contact_frames'] += int(hit)
            else:
                hit = False
        else:
            hit = False
        if not valid:
            self._abort = 'invalid_pose'
        # Reserve time to disengage rather than stop while loaded against T.
        if self.steps >= self.budget_steps-20 and not self._abort:
            self._abort = 'rotation_budget'
        if self.phase in ('approach','descend','seek','servo'):
            if self._abort or err <= self.goal_deg-0.5:
                self._retreat(eef,yaw,self._abort or 'yaw_goal')
        if self.phase in ('raise','close','plan') and self._abort and self.active is None:
            if eef[2] >= .896 or self.steps >= self.budget_steps:
                self._finish(self._abort,angle,valid and eef[2]>=.896)
                return np.r_[np.zeros(6),1. if (self.closed or self.close_count) else -1.]
            return self._move(eef,np.r_[eef[:2],HIGH_Z],gripper=1. if (self.closed or self.close_count) else -1.)
        if self.phase == 'raise':
            if eef[2] < .900:
                return self._move(eef,np.r_[eef[:2],HIGH_Z],gripper=-1.)
            self._phase('close')
        if self.phase == 'close':
            if eef[2] < .896:
                return self._move(eef,np.r_[eef[:2],HIGH_Z],gripper=1. if self.close_count else -1.)
            self.close_count += 1
            if self.close_count >= self.close_steps:
                self.closed = True
                self._phase('plan')
            return np.r_[np.zeros(6),1.]
        if self.phase == 'plan':
            if eef[2] < .896:
                return self._move(eef,np.r_[eef[:2],HIGH_Z])
            if err <= self.goal_deg or self._abort or self.attempts >= self.max_attempts:
                self._finish(self._abort or 'attempt_limit',angle,valid)
                return np.r_[np.zeros(6),1.]
            choice = pick_contact(pos,yaw,angle,eef,self._goal(),self.blocked,self.response)
            if choice is None:
                self._finish('no_feasible_contact',angle,valid)
                return np.r_[np.zeros(6),1.]
            _, c, approach = choice
            _, normal = face_world(c,pos,yaw)
            self.attempts += 1
            self.active = a = dict(contact=c, start_xy=pos.copy(),start_yaw=yaw,start_error=angle,
                start_step=self.steps, approach=approach.copy(),
                wrist=RepositioningPushTExpert.wrist_rotation(self,normal),
                last_xy=pos.copy(),last_eef=eef[:2].copy(),object_path_m=0.,eef_path_m=0.,
                contact_frames=0,contact_api_available=False,motion_evidence=False,
                contact_acquired=False,seek_steps=0,servo_steps=0)
            self._phase('approach')
        if self.phase == 'approach':
            a = self.active
            desired = np.r_[a['approach'],HIGH_Z]
            site = env.robots[0].eef_site_id['right']
            current = env.sim.data.site_xmat[site].reshape(3,3)
            orient_error = Rotation.from_matrix(a['wrist'] @ current.T).magnitude()
            if np.linalg.norm(eef-desired)<.009 and orient_error<.12:
                self._phase('descend')
            elif self.phase_steps>=45:
                self._retreat(eef,yaw,'approach_timeout')
            else:
                return self._move(eef,desired,a['wrist'])
        if self.phase == 'descend':
            a = self.active
            desired = np.r_[a['approach'],PUSH_Z]
            if np.linalg.norm(eef-desired)<.006 or hit:
                a['seek_start_xy'], a['seek_start_yaw'] = pos.copy(), yaw
                a['seek_start_eef'] = eef[:2].copy()
                self._phase('seek')
            elif self.phase_steps>=30:
                self._retreat(eef,yaw,'descend_timeout')
            else:
                return self._move(eef,desired,a['wrist'],cap=.45)
        if self.phase in ('seek','servo'):
            a = self.active
            p, n = face_world(a['contact'],pos,yaw)
            tangent = np.array([-n[1],n[0]])
            offset = eef[:2]-p
            # Contact is physical collision or separately labelled motion
            # evidence after seeking, not merely crossing a geometric plane.
            moved = (np.linalg.norm(pos-a['seek_start_xy'])>.0006 or
                     abs(wrapped(yaw-a['seek_start_yaw']))>np.deg2rad(.3))
            a['motion_evidence'] |= bool(moved)
            if self.phase=='seek':
                a['seek_steps'] += 1
                if hit or moved or a['contact_frames']>0:
                    a['contact_acquired'] = True
                    a['tangent_offset'] = float(np.clip(offset@tangent,-.012,.012))
                    a['servo_start_error'] = err
                    a['best_servo_error'] = err
                    a['last_progress_step'] = self.steps
                    a['servo_start_path'] = a['eef_path_m']
                    self._phase('servo')
                elif a['seek_steps']>=26 or np.linalg.norm(eef[:2]-a['seek_start_eef'])>.052:
                    self._retreat(eef,yaw,'no_contact')
                else:
                    xy = .24*n + np.clip(-8.*float(offset@tangent),-.15,.15)*tangent
                    return np.r_[xy,np.clip((PUSH_Z-eef[2])*18.,-.25,.25),self._move(eef,eef,a['wrist'])[3:6],1.]
            if self.phase=='servo':
                a['servo_steps'] += 1
                if err<a['best_servo_error']-.35:
                    a['best_servo_error'] = err
                    a['last_progress_step'] = self.steps
                overshoot = angle*a['start_error'] < 0 and err>self.goal_deg
                wrong = err>a['best_servo_error']+2.0
                stalled = self.steps-a['last_progress_step']>=10
                drift = np.linalg.norm(pos-self._goal())-np.linalg.norm(a['start_xy']-self._goal())>.035
                outside = np.any(pos<[-.22,-.23]) or np.any(pos>[.18,.23])
                too_far = a['eef_path_m']-a['servo_start_path']>.105 or a['servo_steps']>=65
                if overshoot or wrong or stalled or drift or outside or too_far:
                    reason = ('yaw_crossed' if overshoot else 'wrong_turn' if wrong else 'no_yaw_progress' if stalled else 'away_drift' if drift else 'workspace' if outside else 'stroke_limit')
                    self._retreat(eef,yaw,reason)
                else:
                    # Update the normal AND material contact's tangential position
                    # with measured yaw. Keep the gripper closed and rotate wrist
                    # with the object within one continuous contact segment.
                    speed = float(np.clip(.018*err,.08,.36))
                    correction = np.clip(12.*(a['tangent_offset']-float(offset@tangent)),-.20,.20)
                    xy = speed*n + correction*tangent
                    rz = np.eye(3); rz[:2,:2] = rotation_2d(wrapped(yaw-a['start_yaw']))
                    wrist = rz@a['wrist']
                    return np.r_[np.clip(xy,-.55,.55),np.clip((PUSH_Z-eef[2])*18.,-.25,.25),self._move(eef,eef,wrist)[3:6],1.]
        if self.phase == 'retreat':
            a = self.active
            if np.linalg.norm(eef-a['retreat_target'])<.005 or self.phase_steps>=12:
                self._phase('lift')
            else:
                return self._move(eef,a['retreat_target'],cap=.45)
        if self.phase == 'lift':
            if eef[2]<.896:
                # No sideways repositioning or opening after a lift timeout.
                if self.phase_steps>=24:
                    self._abort='lift_timeout'
                    self._finalize_attempt(pos,yaw,eef)
                    self._finish(self._abort,angle,False)
                return self._move(eef,np.r_[eef[:2],HIGH_Z])
            self._phase('settle')
        if self.phase == 'settle':
            if self.phase_steps>=4:
                self._finalize_attempt(pos,yaw,eef)
                if err<=self.goal_deg or self._abort:
                    self._finish(self._abort or 'aligned',angle,valid)
                else:
                    self._phase('plan')
            return np.r_[np.zeros(6),1.]
        return np.r_[np.zeros(6),1. if self.closed else -1.]

# Long lever first; also allow an opposite side of the crossbar to produce
# the same torque but a different translation. All points lie on the outline.


class ContinuousRotation:
    """Standalone closed-gripper rotation, with feedback through all phases.

    Contact is not switched in the middle of a stroke. During servo, the local
    material contact target and push normal rotate with the measured Tee pose.
    If the current contact is ineffective, retreat/lift before trying a new one.
    """
    def __init__(self, environment, *, goal_deg=2.5, budget_steps=240, max_attempts=5,
                 close_steps=24):
        if not 0 < goal_deg < 10 or budget_steps < 80 or max_attempts < 1 or close_steps < 1:
            raise ValueError('Invalid rotation limits')
        self.environment = environment
        self.goal_deg = float(goal_deg)
        self.budget_steps = int(budget_steps)
        self.max_attempts = int(max_attempts)
        self.close_steps = int(close_steps)
        self.reset()

    def reset(self):
        self.phase = 'raise'
        self.phase_steps = 0
        self.steps = 0
        self.closed = False
        self.close_count = 0
        self.done = False
        self.completed = False
        self.exit_reason = ''
        self.attempts = 0
        self.samples = []
        self.blocked = set()
        self.response = {}
        self.active = None
        self.initial_xy = None
        self.best_error_deg = float('inf')
        self._abort = ''

    def _phase(self, phase):
        self.phase, self.phase_steps = phase, 0

    def _goal(self):
        return self.environment.sim.data.body_xpos[self.environment.target_body_id, :2].copy()

    def _error(self, yaw):
        return wrapped(self.environment.target_yaw-yaw)

    def _move(self, eef, desired, wrist=None, cap=.65, gripper=1.):
        xyz = np.clip((np.asarray(desired)-eef)*18., -cap, cap)
        rot = np.zeros(3)
        if wrist is not None:
            site = self.environment.robots[0].eef_site_id['right']
            current = self.environment.sim.data.site_xmat[site].reshape(3,3)
            rot = np.clip(Rotation.from_matrix(wrist @ current.T).as_rotvec()/.5, -.5, .5)
        return np.r_[xyz, rot, gripper]

    def _finish(self, reason, error, valid):
        self.completed = bool(valid and abs(np.rad2deg(error)) <= self.goal_deg)
        self.exit_reason = 'aligned' if self.completed else reason
        self.done = True
        self._phase('done')

    def _retreat(self, eef, yaw, reason):
        a = self.active
        if a is not None:
            a['stop_reason'] = reason
            if 'servo_start_path' in a:
                a['servo_path_at_stop_m'] = float(a['eef_path_m']-a['servo_start_path'])
            _, n = face_world(a['contact'], a['start_xy'], yaw)
            a['retreat_target'] = np.r_[eef[:2]-.015*n, eef[2]]
        self._phase('retreat' if a is not None and eef[2]<.885 else 'lift')

    def _finalize_attempt(self, pos, yaw, eef):
        a = self.active
        if a is None:
            return
        angle = self._error(yaw)
        turn = wrapped(yaw-a['start_yaw'])
        delta = pos-a['start_xy']
        dist = float(np.linalg.norm(delta))
        progress = abs(a['start_error'])-abs(angle)
        goal = self._goal()
        before, after = np.linalg.norm(goal-a['start_xy']), np.linalg.norm(goal-pos)
        name = a['contact'].name
        # Only learn when there was measurable motion; no-contact is not a
        # zero-rate calibration. Use complete stroke incl. retreat for ranking.
        if dist >= .001 and abs(turn) >= np.deg2rad(.25):
            local_dir = rotation_2d(-a['start_yaw']) @ delta / dist
            rate = float(np.clip(turn/dist, -80, 80))
            if name in self.response:
                old_rate, old_dir = self.response[name]
                rate, local_dir = .25*old_rate+.75*rate, .25*old_dir+.75*local_dir
            self.response[name] = (rate, local_dir)
        if progress < np.deg2rad(.75) or after-before > .035:
            self.blocked.add(name)
        else:
            # Keep the same effective contact eligible on the next stroke.
            self.blocked.discard(name)
        self.samples.append(dict(attempt=self.attempts, contact=name,
            geometric_lever_m=local_lever(a['contact']),
            initial_signed_error_deg=float(np.rad2deg(a['start_error'])),
            final_signed_error_deg=float(np.rad2deg(angle)),
            net_turn_deg=float(np.rad2deg(turn)),
            net_progress_deg=float(np.rad2deg(progress)),
            net_translation_m=dist, goal_distance_before_m=float(before),
            goal_distance_after_m=float(after),
            contact_api_available=a['contact_api_available'],
            contact_frames=a['contact_frames'], motion_evidence=a['motion_evidence'],
            seek_steps=a['seek_steps'], servo_steps=a['servo_steps'],
            contact_acquired=a['contact_acquired'], stop_reason=a.get('stop_reason',''),
            steps_used=self.steps-a['start_step'],
            object_path_m=a['object_path_m'], eef_path_m=a['eef_path_m'],
            servo_path_at_stop_m=float(a.get('servo_path_at_stop_m',0.)),
            continuation_used=bool(a.get('continuation_used',False)),
            servo_path_cap_m=float(a.get('servo_path_cap_m',.105))))
        self.active = None

    def action(self, observation):
        env = self.environment
        eef = np.asarray(observation['robot0_eef_pos'], float)
        pos, yaw = angular_pose(env)
        angle = self._error(yaw)
        err = abs(float(np.rad2deg(angle)))
        m = env.task_metrics()
        valid = bool(m['upright'] and m['on_table'])
        if self.initial_xy is None:
            self.initial_xy = pos.copy()
        self.best_error_deg = min(self.best_error_deg,err)
        if self.done:
            return np.r_[np.zeros(6), 1. if self.closed else -1.]
        self.steps += 1
        self.phase_steps += 1
        a = self.active
        if a is not None:
            a['object_path_m'] += float(np.linalg.norm(pos-a['last_xy']))
            a['eef_path_m'] += float(np.linalg.norm(eef[:2]-a['last_eef']))
            a['last_xy'], a['last_eef'] = pos.copy(), eef[:2].copy()
            if self.phase in ('descend','seek','servo'):
                available, hit = tee_robot_contact(env)
                a['contact_api_available'] |= bool(available)
                a['contact_frames'] += int(hit)
            else:
                hit = False
        else:
            hit = False
        if not valid:
            self._abort = 'invalid_pose'
        # Reserve time to disengage rather than stop while loaded against T.
        if self.steps >= self.budget_steps-20 and not self._abort:
            self._abort = 'rotation_budget'
        if self.phase in ('approach','descend','seek','servo'):
            if self._abort or err <= self.goal_deg-0.5:
                self._retreat(eef,yaw,self._abort or 'yaw_goal')
        if self.phase in ('raise','close','plan') and self._abort and self.active is None:
            if eef[2] >= .896 or self.steps >= self.budget_steps:
                self._finish(self._abort,angle,valid and eef[2]>=.896)
                return np.r_[np.zeros(6),1. if (self.closed or self.close_count) else -1.]
            return self._move(eef,np.r_[eef[:2],HIGH_Z],gripper=1. if (self.closed or self.close_count) else -1.)
        if self.phase == 'raise':
            if eef[2] < .900:
                return self._move(eef,np.r_[eef[:2],HIGH_Z],gripper=-1.)
            self._phase('close')
        if self.phase == 'close':
            if eef[2] < .896:
                return self._move(eef,np.r_[eef[:2],HIGH_Z],gripper=1. if self.close_count else -1.)
            self.close_count += 1
            if self.close_count >= self.close_steps:
                self.closed = True
                self._phase('plan')
            return np.r_[np.zeros(6),1.]
        if self.phase == 'plan':
            if eef[2] < .896:
                return self._move(eef,np.r_[eef[:2],HIGH_Z])
            if err <= self.goal_deg or self._abort or self.attempts >= self.max_attempts:
                self._finish(self._abort or 'attempt_limit',angle,valid)
                return np.r_[np.zeros(6),1.]
            # Optional one-attempt forced contact used only by adaptive expert rescue.
            blocked = self.blocked
            preferred = getattr(self, 'preferred_contact', None)
            if preferred is not None and self.attempts == 0:
                blocked = blocked | {c.name for c in CONTACTS if c.name != preferred}
            choice = pick_contact(pos,yaw,angle,eef,self._goal(),blocked,self.response)
            if choice is None:
                self._finish('no_feasible_contact',angle,valid)
                return np.r_[np.zeros(6),1.]
            _, c, approach = choice
            _, normal = face_world(c,pos,yaw)
            self.attempts += 1
            self.active = a = dict(contact=c, start_xy=pos.copy(),start_yaw=yaw,start_error=angle,
                start_step=self.steps, approach=approach.copy(),
                wrist=RepositioningPushTExpert.wrist_rotation(self,normal),
                last_xy=pos.copy(),last_eef=eef[:2].copy(),object_path_m=0.,eef_path_m=0.,
                contact_frames=0,contact_api_available=False,motion_evidence=False,
                contact_acquired=False,seek_steps=0,servo_steps=0)
            self._phase('approach')
        if self.phase == 'approach':
            a = self.active
            desired = np.r_[a['approach'],HIGH_Z]
            site = env.robots[0].eef_site_id['right']
            current = env.sim.data.site_xmat[site].reshape(3,3)
            orient_error = Rotation.from_matrix(a['wrist'] @ current.T).magnitude()
            if np.linalg.norm(eef-desired)<.009 and orient_error<.12:
                self._phase('descend')
            elif self.phase_steps>=45:
                self._retreat(eef,yaw,'approach_timeout')
            else:
                return self._move(eef,desired,a['wrist'])
        if self.phase == 'descend':
            a = self.active
            desired = np.r_[a['approach'],PUSH_Z]
            if np.linalg.norm(eef-desired)<.006 or hit:
                a['seek_start_xy'], a['seek_start_yaw'] = pos.copy(), yaw
                a['seek_start_eef'] = eef[:2].copy()
                self._phase('seek')
            elif self.phase_steps>=30:
                self._retreat(eef,yaw,'descend_timeout')
            else:
                return self._move(eef,desired,a['wrist'],cap=.45)
        if self.phase in ('seek','servo'):
            a = self.active
            p, n = face_world(a['contact'],pos,yaw)
            tangent = np.array([-n[1],n[0]])
            offset = eef[:2]-p
            # Contact is physical collision or separately labelled motion
            # evidence after seeking, not merely crossing a geometric plane.
            moved = (np.linalg.norm(pos-a['seek_start_xy'])>.0006 or
                     abs(wrapped(yaw-a['seek_start_yaw']))>np.deg2rad(.3))
            a['motion_evidence'] |= bool(moved)
            if self.phase=='seek':
                a['seek_steps'] += 1
                if hit or moved or a['contact_frames']>0:
                    a['contact_acquired'] = True
                    a['tangent_offset'] = float(np.clip(offset@tangent,-.012,.012))
                    a['servo_start_error'] = err
                    a['best_servo_error'] = err
                    a['last_progress_step'] = self.steps
                    a['servo_start_path'] = a['eef_path_m']
                    self._phase('servo')
                elif a['seek_steps']>=26 or np.linalg.norm(eef[:2]-a['seek_start_eef'])>.052:
                    self._retreat(eef,yaw,'no_contact')
                else:
                    xy = .24*n + np.clip(-8.*float(offset@tangent),-.15,.15)*tangent
                    return np.r_[xy,np.clip((PUSH_Z-eef[2])*18.,-.25,.25),self._move(eef,eef,a['wrist'])[3:6],1.]
            if self.phase=='servo':
                a['servo_steps'] += 1
                if err<a['best_servo_error']-.35:
                    a['best_servo_error'] = err
                    a['last_progress_step'] = self.steps
                overshoot = angle*a['start_error'] < 0 and err>self.goal_deg
                wrong = err>a['best_servo_error']+2.0
                stalled = self.steps-a['last_progress_step']>=10
                drift = np.linalg.norm(pos-self._goal())-np.linalg.norm(a['start_xy']-self._goal())>.035
                outside = np.any(pos<[-.22,-.23]) or np.any(pos>[.18,.23])
                # Ordinary strokes retain the established 105 mm cap. A large
                # first stroke that already corrected >35 deg, with <=15 deg
                # remaining, can finish without another approach/descent.
                # All original wrong-turn, stall, drift and workspace guards
                # remain independent vetoes. Only a bounded extension is allowed.
                path = a['eef_path_m']-a['servo_start_path']
                if (path>=.105 and abs(np.rad2deg(a['start_error']))>=45.
                        and abs(np.rad2deg(a['start_error']))-err>=35.
                        and self.goal_deg<err<=15.
                        and not (overshoot or wrong or stalled or drift or outside)
                        and self.steps < self.budget_steps-20):
                    a['continuation_used'] = True
                cap = .155 if a.get('continuation_used',False) else .105
                a['servo_path_cap_m'] = cap
                too_far = path>cap or a['servo_steps']>=65
                if overshoot or wrong or stalled or drift or outside or too_far:
                    reason = ('yaw_crossed' if overshoot else 'wrong_turn' if wrong else 'no_yaw_progress' if stalled else 'away_drift' if drift else 'workspace' if outside else 'stroke_limit')
                    self._retreat(eef,yaw,reason)
                else:
                    # Update the normal AND material contact's tangential position
                    # with measured yaw. Keep the gripper closed and rotate wrist
                    # with the object within one continuous contact segment.
                    speed = float(np.clip(.018*err,.08,.36))
                    correction = np.clip(12.*(a['tangent_offset']-float(offset@tangent)),-.20,.20)
                    xy = speed*n + correction*tangent
                    rz = np.eye(3); rz[:2,:2] = rotation_2d(wrapped(yaw-a['start_yaw']))
                    wrist = rz@a['wrist']
                    return np.r_[np.clip(xy,-.55,.55),np.clip((PUSH_Z-eef[2])*18.,-.25,.25),self._move(eef,eef,wrist)[3:6],1.]
        if self.phase == 'retreat':
            a = self.active
            if np.linalg.norm(eef-a['retreat_target'])<.005 or self.phase_steps>=12:
                self._phase('lift')
            else:
                return self._move(eef,a['retreat_target'],cap=.45)
        if self.phase == 'lift':
            if eef[2]<.896:
                # No sideways repositioning or opening after a lift timeout.
                if self.phase_steps>=24:
                    self._abort='lift_timeout'
                    self._finalize_attempt(pos,yaw,eef)
                    self._finish(self._abort,angle,False)
                return self._move(eef,np.r_[eef[:2],HIGH_Z])
            self._phase('settle')
        if self.phase == 'settle':
            if self.phase_steps>=4:
                self._finalize_attempt(pos,yaw,eef)
                if err<=self.goal_deg or self._abort:
                    self._finish(self._abort or 'aligned',angle,valid)
                else:
                    self._phase('plan')
            return np.r_[np.zeros(6),1.]
        return np.r_[np.zeros(6),1. if self.closed else -1.]

# Long lever first; also allow an opposite side of the crossbar to produce
# the same torque but a different translation. All points lie on the outline.


def arc_forecast(pos, yaw, turn, local_per_rad, contact, tool_offset=.012):
    """Integrate measured/estimated body-frame motion over a turning arc.
    local_per_rad is translation per POSITIVE magnitude of yaw change.
    Returns endpoint and sampled object/tool paths (world coordinates).
    A kinematic forecast, not a contact-physics simulation.
    """
    samples=max(2,int(np.ceil(abs(np.rad2deg(turn))/5.)))
    headings=yaw+np.linspace(0.,turn,samples+1)
    ds=abs(turn)/samples
    increments=np.array([rotation_2d(h)@local_per_rad*ds for h in (headings[:-1]+headings[1:])/2])
    centers=np.vstack([pos,pos+np.cumsum(increments,axis=0)])
    tools=np.array([c+rotation_2d(h)@(np.asarray(contact.xy)-tool_offset*np.asarray(contact.normal)) for c,h in zip(centers,headings)])
    return centers[-1],centers,tools


def arc_safe(centers,tools):
    return bool(np.all(centers>=[-.22,-.23]) and np.all(centers<=[.18,.23])
                and np.all(tools>=[-.29,-.28]) and np.all(tools<=[.25,.28]))


def pick_arc_contact(pos,yaw,error,eef,target,blocked,response,target_yaw):
    """Prefer proven tip contact unless another route forecasts a substantial
    reduction in residual XY work. Crossbar motions are uncalibrated priors.
    """
    opts=[]
    for c in CONTACTS:
        if c.name in blocked: continue
        point,n=face_world(c,pos,yaw); approach=point-APPROACH_GAP*n
        if np.any(approach<LOW) or np.any(approach>HIGH): continue
        prior=local_lever(c)/.0035
        if c.name.startswith('stem_tip'): prior=np.sign(prior)/.080
        elif c.name.startswith('stem_side'): prior=np.sign(prior)/.095
        rate,direction=response.get(c.name,(prior,np.asarray(c.normal,float)))
        if rate*error<=0 or abs(rate)<1.: continue
        v=np.asarray(direction)/abs(rate)
        # Evaluate full yaw goal, rather than the old 20 mm lookahead.
        end,centers,tools=arc_forecast(pos,yaw,error,v,c)
        residual=rotation_2d(-target_yaw)@(target-end)
        feasible=arc_safe(centers,tools)
        # Y is expensive: often requires a pair of opposing-torque contacts.
        # Express cost in approximate steps; add penalty for untested bar faces.
        cost=(abs(residual[0])/.004+abs(residual[1])/.003
              +abs(error)/abs(rate)/.0035+np.linalg.norm(eef[:2]-approach)/.007
              +(30 if not c.name.startswith('stem') else 0)+(65 if not feasible else 0))
        # Prefer the empirically efficient, long lever over a nearby short one.
        opts.append((float(cost),c,approach,dict(forecast_x_error_m=float(residual[0]),forecast_y_error_m=float(residual[1]),forecast_workspace_ok=feasible,forecast_cost=float(cost))))
    return min(opts,key=lambda x:x[0]) if opts else None


class ArcRotation:
    """Standalone closed-gripper rotation, with feedback through all phases.

    Contact is not switched in the middle of a stroke. During servo, the local
    material contact target and push normal rotate with the measured Tee pose.
    If the current contact is ineffective, retreat/lift before trying a new one.
    """
    def __init__(self, environment, *, goal_deg=2.5, budget_steps=240, max_attempts=5,
                 close_steps=24):
        if not 0 < goal_deg < 10 or budget_steps < 80 or max_attempts < 1 or close_steps < 1:
            raise ValueError('Invalid rotation limits')
        self.environment = environment
        self.goal_deg = float(goal_deg)
        self.budget_steps = int(budget_steps)
        self.max_attempts = int(max_attempts)
        self.close_steps = int(close_steps)
        self.reset()

    def reset(self):
        self.phase = 'raise'
        self.phase_steps = 0
        self.steps = 0
        self.closed = False
        self.close_count = 0
        self.done = False
        self.completed = False
        self.exit_reason = ''
        self.attempts = 0
        self.samples = []
        self.blocked = set()
        self.response = {}
        self.active = None
        self.initial_xy = None
        self.best_error_deg = float('inf')
        self._abort = ''

    def _phase(self, phase):
        self.phase, self.phase_steps = phase, 0

    def _goal(self):
        return self.environment.sim.data.body_xpos[self.environment.target_body_id, :2].copy()

    def _error(self, yaw):
        return wrapped(self.environment.target_yaw-yaw)

    def _move(self, eef, desired, wrist=None, cap=.65, gripper=1.):
        xyz = np.clip((np.asarray(desired)-eef)*18., -cap, cap)
        rot = np.zeros(3)
        if wrist is not None:
            site = self.environment.robots[0].eef_site_id['right']
            current = self.environment.sim.data.site_xmat[site].reshape(3,3)
            rot = np.clip(Rotation.from_matrix(wrist @ current.T).as_rotvec()/.5, -.5, .5)
        return np.r_[xyz, rot, gripper]

    def _finish(self, reason, error, valid):
        self.completed = bool(valid and abs(np.rad2deg(error)) <= self.goal_deg)
        self.exit_reason = 'aligned' if self.completed else reason
        self.done = True
        self._phase('done')

    def _retreat(self, eef, yaw, reason):
        a = self.active
        if a is not None:
            a['stop_reason'] = reason
            if 'servo_start_path' in a:
                a['servo_path_at_stop_m'] = float(a['eef_path_m']-a['servo_start_path'])
            _, n = face_world(a['contact'], a['start_xy'], yaw)
            a['retreat_target'] = np.r_[eef[:2]-.015*n, eef[2]]
        self._phase('retreat' if a is not None and eef[2]<.885 else 'lift')

    def _finalize_attempt(self, pos, yaw, eef):
        a = self.active
        if a is None:
            return
        angle = self._error(yaw)
        turn = wrapped(yaw-a['start_yaw'])
        delta = pos-a['start_xy']
        dist = float(np.linalg.norm(delta))
        progress = abs(a['start_error'])-abs(angle)
        goal = self._goal()
        before, after = np.linalg.norm(goal-a['start_xy']), np.linalg.norm(goal-pos)
        name = a['contact'].name
        # Only learn when there was measurable motion; no-contact is not a
        # zero-rate calibration. Use complete stroke incl. retreat for ranking.
        if dist >= .001 and abs(turn) >= np.deg2rad(.25):
            local_dir = rotation_2d(-a['start_yaw']) @ delta / dist
            rate = float(np.clip(turn/dist, -80, 80))
            if name in self.response:
                old_rate, old_dir = self.response[name]
                rate, local_dir = .25*old_rate+.75*rate, .25*old_dir+.75*local_dir
            self.response[name] = (rate, local_dir)
        if progress < np.deg2rad(.75) or after-before > .035:
            self.blocked.add(name)
        else:
            # Keep the same effective contact eligible on the next stroke.
            self.blocked.discard(name)
        self.samples.append(dict(attempt=self.attempts, contact=name,
            geometric_lever_m=local_lever(a['contact']),
            initial_signed_error_deg=float(np.rad2deg(a['start_error'])),
            final_signed_error_deg=float(np.rad2deg(angle)),
            net_turn_deg=float(np.rad2deg(turn)),
            net_progress_deg=float(np.rad2deg(progress)),
            net_translation_m=dist, goal_distance_before_m=float(before),
            goal_distance_after_m=float(after),
            contact_api_available=a['contact_api_available'],
            contact_frames=a['contact_frames'], motion_evidence=a['motion_evidence'],
            seek_steps=a['seek_steps'], servo_steps=a['servo_steps'],
            contact_acquired=a['contact_acquired'], stop_reason=a.get('stop_reason',''),
            steps_used=self.steps-a['start_step'],
            object_path_m=a['object_path_m'], eef_path_m=a['eef_path_m'],
            servo_path_at_stop_m=float(a.get('servo_path_at_stop_m',0.)),
            continuation_used=bool(a.get('continuation_used',False)),
            servo_path_cap_m=float(a.get('servo_path_cap_m',.105)),
            forecast_x_error_m=a.get('forecast_x_error_m'),forecast_y_error_m=a.get('forecast_y_error_m'),
            forecast_workspace_ok=a.get('forecast_workspace_ok'),forecast_cost=a.get('forecast_cost'),
            continuation_forecast_ok=a.get('continuation_forecast_ok'),
            continuation_y_error_m=a.get('continuation_y_error_m'),
            continuation_path_needed_m=a.get('continuation_path_needed_m')))
        self.active = None

    def action(self, observation):
        env = self.environment
        eef = np.asarray(observation['robot0_eef_pos'], float)
        pos, yaw = angular_pose(env)
        angle = self._error(yaw)
        err = abs(float(np.rad2deg(angle)))
        m = env.task_metrics()
        valid = bool(m['upright'] and m['on_table'])
        if self.initial_xy is None:
            self.initial_xy = pos.copy()
        self.best_error_deg = min(self.best_error_deg,err)
        if self.done:
            return np.r_[np.zeros(6), 1. if self.closed else -1.]
        self.steps += 1
        self.phase_steps += 1
        a = self.active
        if a is not None:
            a['object_path_m'] += float(np.linalg.norm(pos-a['last_xy']))
            a['eef_path_m'] += float(np.linalg.norm(eef[:2]-a['last_eef']))
            a['last_xy'], a['last_eef'] = pos.copy(), eef[:2].copy()
            if self.phase in ('descend','seek','servo'):
                available, hit = tee_robot_contact(env)
                a['contact_api_available'] |= bool(available)
                a['contact_frames'] += int(hit)
            else:
                hit = False
        else:
            hit = False
        if not valid:
            self._abort = 'invalid_pose'
        # Reserve time to disengage rather than stop while loaded against T.
        if self.steps >= self.budget_steps-20 and not self._abort:
            self._abort = 'rotation_budget'
        if self.phase in ('approach','descend','seek','servo'):
            if self._abort or err <= self.goal_deg-0.5:
                self._retreat(eef,yaw,self._abort or 'yaw_goal')
        if self.phase in ('raise','close','plan') and self._abort and self.active is None:
            if eef[2] >= .896 or self.steps >= self.budget_steps:
                self._finish(self._abort,angle,valid and eef[2]>=.896)
                return np.r_[np.zeros(6),1. if (self.closed or self.close_count) else -1.]
            return self._move(eef,np.r_[eef[:2],HIGH_Z],gripper=1. if (self.closed or self.close_count) else -1.)
        if self.phase == 'raise':
            if eef[2] < .900:
                return self._move(eef,np.r_[eef[:2],HIGH_Z],gripper=-1.)
            self._phase('close')
        if self.phase == 'close':
            if eef[2] < .896:
                return self._move(eef,np.r_[eef[:2],HIGH_Z],gripper=1. if self.close_count else -1.)
            self.close_count += 1
            if self.close_count >= self.close_steps:
                self.closed = True
                self._phase('plan')
            return np.r_[np.zeros(6),1.]
        if self.phase == 'plan':
            if eef[2] < .896:
                return self._move(eef,np.r_[eef[:2],HIGH_Z])
            if err <= self.goal_deg or self._abort or self.attempts >= self.max_attempts:
                self._finish(self._abort or 'attempt_limit',angle,valid)
                return np.r_[np.zeros(6),1.]
            choice = pick_arc_contact(pos,yaw,angle,eef,self._goal(),self.blocked,self.response,env.target_yaw)
            if choice is None:
                self._finish('no_feasible_contact',angle,valid)
                return np.r_[np.zeros(6),1.]
            _, c, approach, forecast = choice
            _, normal = face_world(c,pos,yaw)
            self.attempts += 1
            self.active = a = dict(contact=c, start_xy=pos.copy(),start_yaw=yaw,start_error=angle,
                start_step=self.steps, approach=approach.copy(),
                wrist=RepositioningPushTExpert.wrist_rotation(self,normal),
                last_xy=pos.copy(),last_eef=eef[:2].copy(),object_path_m=0.,eef_path_m=0.,
                contact_frames=0,contact_api_available=False,motion_evidence=False,
                contact_acquired=False,seek_steps=0,servo_steps=0,**forecast)
            self._phase('approach')
        if self.phase == 'approach':
            a = self.active
            desired = np.r_[a['approach'],HIGH_Z]
            site = env.robots[0].eef_site_id['right']
            current = env.sim.data.site_xmat[site].reshape(3,3)
            orient_error = Rotation.from_matrix(a['wrist'] @ current.T).magnitude()
            if np.linalg.norm(eef-desired)<.009 and orient_error<.12:
                self._phase('descend')
            elif self.phase_steps>=45:
                self._retreat(eef,yaw,'approach_timeout')
            else:
                return self._move(eef,desired,a['wrist'])
        if self.phase == 'descend':
            a = self.active
            desired = np.r_[a['approach'],PUSH_Z]
            if np.linalg.norm(eef-desired)<.006 or hit:
                a['seek_start_xy'], a['seek_start_yaw'] = pos.copy(), yaw
                a['seek_start_eef'] = eef[:2].copy()
                self._phase('seek')
            elif self.phase_steps>=30:
                self._retreat(eef,yaw,'descend_timeout')
            else:
                return self._move(eef,desired,a['wrist'],cap=.45)
        if self.phase in ('seek','servo'):
            a = self.active
            p, n = face_world(a['contact'],pos,yaw)
            tangent = np.array([-n[1],n[0]])
            offset = eef[:2]-p
            # Contact is physical collision or separately labelled motion
            # evidence after seeking, not merely crossing a geometric plane.
            moved = (np.linalg.norm(pos-a['seek_start_xy'])>.0006 or
                     abs(wrapped(yaw-a['seek_start_yaw']))>np.deg2rad(.3))
            a['motion_evidence'] |= bool(moved)
            if self.phase=='seek':
                a['seek_steps'] += 1
                if hit or moved or a['contact_frames']>0:
                    a['contact_acquired'] = True
                    a['tangent_offset'] = float(np.clip(offset@tangent,-.012,.012))
                    a['servo_start_error'] = err
                    a['best_servo_error'] = err
                    a['last_progress_step'] = self.steps
                    a['servo_start_path'] = a['eef_path_m']
                    self._phase('servo')
                elif a['seek_steps']>=26 or np.linalg.norm(eef[:2]-a['seek_start_eef'])>.052:
                    self._retreat(eef,yaw,'no_contact')
                else:
                    xy = .24*n + np.clip(-8.*float(offset@tangent),-.15,.15)*tangent
                    return np.r_[xy,np.clip((PUSH_Z-eef[2])*18.,-.25,.25),self._move(eef,eef,a['wrist'])[3:6],1.]
            if self.phase=='servo':
                a['servo_steps'] += 1
                if err<a['best_servo_error']-.35:
                    a['best_servo_error'] = err
                    a['last_progress_step'] = self.steps
                overshoot = angle*a['start_error'] < 0 and err>self.goal_deg
                wrong = err>a['best_servo_error']+2.0
                stalled = self.steps-a['last_progress_step']>=10
                drift = np.linalg.norm(pos-self._goal())-np.linalg.norm(a['start_xy']-self._goal())>.035
                outside = np.any(pos<[-.22,-.23]) or np.any(pos>[.18,.23])
                # At the old 105mm cutoff, estimate the remaining arc from
                # the current measured progress and path. Continue if it fits
                # within the bounded continuation/workspace and time bounds. No seed/angle
                # lookup, and the original physical vetoes remain active.
                path = a['eef_path_m']-a['servo_start_path']
                progress = a['servo_start_error']-err
                # Integrate body-frame displacement per radian, using recent
                # observed pose increments rather than a fixed world direction.
                prev=a.get('arc_prev')
                if prev is not None:
                    prev_pos,prev_yaw=prev
                    dtheta=wrapped(yaw-prev_yaw)
                    if dtheta*a['start_error']>0 and abs(dtheta)>np.deg2rad(.15):
                        local=rotation_2d(-prev_yaw-dtheta/2)@(pos-prev_pos)/abs(dtheta)
                        if np.linalg.norm(local)<.25:
                            a['arc_local_per_rad']=.75*a.get('arc_local_per_rad',local)+.25*local
                a['arc_prev']=(pos.copy(),yaw)
                if path>=.105 and progress>=25. and err>self.goal_deg and not a.get('continuation_used'):
                    v=a.get('arc_local_per_rad',np.asarray(a['contact'].normal)*.080)
                    end,centers,tools=arc_forecast(pos,yaw,angle,v,a['contact'])
                    path_needed=path*(1.+err/max(progress,1.))*1.08
                    steps_needed=int(np.ceil(a['servo_steps']*err/max(progress,1.)*1.20))+8
                    residual=rotation_2d(-env.target_yaw)@(self._goal()-end)
                    good=arc_safe(centers,tools)
                    a.update(continuation_forecast_ok=good,continuation_y_error_m=float(residual[1]),continuation_path_needed_m=float(path_needed))
                    # Finish as much of the useful arc as possible without a
                    # needless reposition, even if another contact may be needed.
                    if (good and path_needed<=.285 and self.steps+steps_needed<self.budget_steps-20
                            and not(overshoot or wrong or stalled or drift or outside)):
                        a['continuation_used']=True
                cap = .265 if a.get('continuation_used',False) else .105
                a['servo_path_cap_m'] = cap
                outside = outside or bool(np.any(eef[:2]<[-.29,-.28]) or np.any(eef[:2]>[.25,.28]))
                too_far = path>cap or a['servo_steps']>=80
                if overshoot or wrong or stalled or drift or outside or too_far:
                    reason = ('yaw_crossed' if overshoot else 'wrong_turn' if wrong else 'no_yaw_progress' if stalled else 'away_drift' if drift else 'workspace' if outside else 'stroke_limit')
                    self._retreat(eef,yaw,reason)
                else:
                    # Update the normal AND material contact's tangential position
                    # with measured yaw. Keep the gripper closed and rotate wrist
                    # with the object within one continuous contact segment.
                    speed = float(np.clip(.018*err,.08,.36))
                    correction = np.clip(12.*(a['tangent_offset']-float(offset@tangent)),-.20,.20)
                    xy = speed*n + correction*tangent
                    rz = np.eye(3); rz[:2,:2] = rotation_2d(wrapped(yaw-a['start_yaw']))
                    wrist = rz@a['wrist']
                    return np.r_[np.clip(xy,-.55,.55),np.clip((PUSH_Z-eef[2])*18.,-.25,.25),self._move(eef,eef,wrist)[3:6],1.]
        if self.phase == 'retreat':
            a = self.active
            if np.linalg.norm(eef-a['retreat_target'])<.005 or self.phase_steps>=12:
                self._phase('lift')
            else:
                return self._move(eef,a['retreat_target'],cap=.45)
        if self.phase == 'lift':
            if eef[2]<.896:
                # No sideways repositioning or opening after a lift timeout.
                if self.phase_steps>=24:
                    self._abort='lift_timeout'
                    self._finalize_attempt(pos,yaw,eef)
                    self._finish(self._abort,angle,False)
                return self._move(eef,np.r_[eef[:2],HIGH_Z])
            self._phase('settle')
        if self.phase == 'settle':
            if self.phase_steps>=4:
                self._finalize_attempt(pos,yaw,eef)
                if err<=self.goal_deg or self._abort:
                    self._finish(self._abort or 'aligned',angle,valid)
                else:
                    self._phase('plan')
            return np.r_[np.zeros(6),1.]
        return np.r_[np.zeros(6),1. if self.closed else -1.]
