"""MuJoCo / robosuite Push-T environments and configuration."""
from __future__ import annotations

import numpy as np
from geometry import OUTLINE, rotation_2d
from shapely.geometry import Polygon
from robosuite.environments.manipulation.lift import Lift
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import CompositeObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import new_body, new_geom


class PandaPushT(Lift):
    def __init__(self, *args, overlap_threshold=0.85, yaw_range=0.30, initial_yaw_error=0.15,
                 independent_poses=False, **kwargs):
        self.overlap_threshold = overlap_threshold
        self.yaw_range = yaw_range
        self.initial_yaw_error = initial_yaw_error
        self.independent_poses = independent_poses
        self.target_xy = np.zeros(2)
        self.target_yaw = 0.0
        super().__init__(*args, **kwargs)

    def _load_model(self):
        ManipulationEnv._load_model(self)
        offset = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(offset)
        arena = TableArena(table_full_size=self.table_full_size,
                           table_friction=self.table_friction, table_offset=self.table_offset)
        arena.set_origin([0, 0, 0])
        self.cube = CompositeObject(
            name="tee", total_size=[0.06, 0.06, 0.015],
            geom_types=["box", "box"],
            geom_sizes=[[0.0175, 0.06, 0.015], [0.0425, 0.0175, 0.015]],
            geom_locations=[[-0.0425, 0, 0], [0.0175, 0, 0]],
            geom_names=["crossbar", "stem"], rgba=[0.9, 0.12, 0.12, 1],
            geom_frictions=[[0.35, 0.005, 0.0001]] * 2,
            density=400, locations_relative_to_center=True,
        )
        target = new_body(name="push_t_target", pos=[0, 0, 0.8005])
        for name, position, size in [
            ("crossbar", [-0.0425, 0, 0], [0.0175, 0.06, 0.0004]),
            ("stem", [0.0175, 0, 0], [0.0425, 0.0175, 0.0004]),
        ]:
            target.append(new_geom(name=f"target_{name}", type="box", pos=position, size=size,
                                   rgba=[0.1, 0.8, 0.2, 0.65], contype="0", conaffinity="0", group="1"))
        arena.worldbody.append(target)
        self.model = ManipulationTask(mujoco_arena=arena,
                                      mujoco_robots=[robot.robot_model for robot in self.robots],
                                      mujoco_objects=self.cube)

    def _setup_references(self):
        super()._setup_references()
        self.target_body_id = self.sim.model.body_name2id("push_t_target")

    def _sample_task_pose(self):
        if self.independent_poses:
            # Sample both poses independently inside the central working area.
            # Reject trials that are already too close to the goal.
            for _ in range(500):
                initial_xy, target_xy = self.rng.uniform([-0.16, -0.13], [0.10, 0.13], (2, 2))
                initial_yaw, target_yaw = self.rng.uniform(-np.pi, np.pi, 2)
                obj = Polygon(OUTLINE @ rotation_2d(initial_yaw).T + initial_xy)
                goal = Polygon(OUTLINE @ rotation_2d(target_yaw).T + target_xy)
                if obj.intersection(goal).area < 0.1 * obj.union(goal).area:
                    return initial_xy, initial_yaw, target_xy, target_yaw
            raise RuntimeError("Could not sample a separated Push-T pose pair")

        initial_xy = self.rng.uniform([-0.13, -0.055], [-0.08, 0.055])
        target_yaw = self.rng.uniform(-self.yaw_range, self.yaw_range)
        distance = self.rng.uniform(0.11, 0.16)
        lateral = self.rng.uniform(-0.008, 0.008)
        target_xy = initial_xy + rotation_2d(target_yaw) @ np.array([distance, lateral])
        initial_yaw = target_yaw + self.rng.uniform(-self.initial_yaw_error, self.initial_yaw_error)
        return initial_xy, initial_yaw, target_xy, target_yaw

    def _reset_internal(self):
        ManipulationEnv._reset_internal(self)
        if not self.deterministic_reset:
            initial_xy, initial_yaw, self.target_xy, self.target_yaw = self._sample_task_pose()
            quaternion = [np.cos(self.target_yaw / 2), 0, 0, np.sin(self.target_yaw / 2)]
            initial_quaternion = [np.cos(initial_yaw / 2), 0, 0, np.sin(initial_yaw / 2)]
            self.sim.data.set_joint_qpos(self.cube.joints[0], np.r_[initial_xy, 0.815, initial_quaternion])
            self.sim.model.body_pos[self.target_body_id] = np.r_[self.target_xy, 0.8005]
            self.sim.model.body_quat[self.target_body_id] = quaternion

    def task_metrics(self):
        position = self.sim.data.body_xpos[self.cube_body_id]
        rotation = self.sim.data.body_xmat[self.cube_body_id].reshape(3, 3)
        target_position = self.sim.data.body_xpos[self.target_body_id]
        target_rotation = self.sim.data.body_xmat[self.target_body_id].reshape(3, 3)
        object_polygon = Polygon(OUTLINE @ rotation[:2, :2].T + position[:2])
        target_polygon = Polygon(OUTLINE @ target_rotation[:2, :2].T + target_position[:2])
        intersection = object_polygon.intersection(target_polygon).area
        union = object_polygon.union(target_polygon).area
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        target_yaw = np.arctan2(target_rotation[1, 0], target_rotation[0, 0])
        yaw_error = abs(np.arctan2(np.sin(yaw - target_yaw), np.cos(yaw - target_yaw)))
        return dict(iou=float(intersection / union), coverage=float(intersection / target_polygon.area),
                    position_error=float(np.linalg.norm(position[:2] - target_position[:2])),
                    yaw_error_degrees=float(np.rad2deg(yaw_error)),
                    upright=bool(rotation[2, 2] > 0.95), on_table=bool(abs(position[2] - 0.815) < 0.012))

    def _check_success(self):
        metrics = self.task_metrics()
        return metrics["iou"] >= self.overlap_threshold and metrics["upright"] and metrics["on_table"]

    def reward(self, action=None):
        return float(self._check_success())


def base_metadata():
    return dict(env_name="PandaPushT", env_version="custom-v1", type=1, env_kwargs=dict(
        robots="Panda", initialization_noise=None, table_friction=[0.35, 0.005, 0.0001],
        controller_configs=dict(type="BASIC", body_parts=dict(right=dict(
            type="OSC_POSE", input_max=1, input_min=-1,
            output_max=[0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
            output_min=[-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
            kp=150, damping_ratio=1, impedance_mode="fixed", uncouple_pos_ori=True,
            input_type="delta", input_ref_frame="world", interpolation=None,
            gripper=dict(type="GRIP")))),
        control_freq=20, lite_physics=False, use_object_obs=True,
        overlap_threshold=0.85, yaw_range=0.30, initial_yaw_error=0.15,
    ))


class PandaPushTRandom(PandaPushT):
    def __init__(self, *args, clutter_count=8, **kwargs):
        self.clutter_count = int(clutter_count)
        self.clutter = []
        super().__init__(*args, **kwargs)

    def _load_model(self):
        ManipulationEnv._load_model(self)
        offset = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(offset)
        arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        arena.set_origin([0, 0, 0])
        self.cube = CompositeObject(
            name="tee", total_size=[0.06, 0.06, 0.015],
            geom_types=["box", "box"],
            geom_sizes=[[0.0175, 0.06, 0.015], [0.0425, 0.0175, 0.015]],
            geom_locations=[[-0.0425, 0, 0], [0.0175, 0, 0]],
            geom_names=["crossbar", "stem"], rgba=[0.9, 0.12, 0.12, 1],
            geom_frictions=[[0.35, 0.005, 0.0001]] * 2,
            density=400, locations_relative_to_center=True,
        )
        colors = [
            [0.15, 0.35, 0.95, 1], [0.95, 0.65, 0.10, 1],
            [0.65, 0.20, 0.85, 1], [0.10, 0.75, 0.75, 1],
        ]
        self.clutter = []
        for index in range(self.clutter_count):
            half_x = 0.012 + 0.004 * (index % 3)
            half_y = 0.010 + 0.003 * ((index + 1) % 3)
            item = CompositeObject(
                name=f"clutter_{index}", total_size=[half_x, half_y, 0.012],
                geom_types=["box"], geom_sizes=[[half_x, half_y, 0.012]],
                geom_locations=[[0, 0, 0]], geom_names=["body"],
                rgba=colors[index % len(colors)],
                geom_frictions=[[0.45, 0.005, 0.0001]], density=300,
                locations_relative_to_center=True,
            )
            self.clutter.append(item)
        target = new_body(name="push_t_target", pos=[0, 0, 0.8005])
        for name, position, size in [
            ("crossbar", [-0.0425, 0, 0], [0.0175, 0.06, 0.0004]),
            ("stem", [0.0175, 0, 0], [0.0425, 0.0175, 0.0004]),
        ]:
            target.append(new_geom(
                name=f"target_{name}", type="box", pos=position, size=size,
                rgba=[0.1, 0.8, 0.2, 0.65], contype="0", conaffinity="0", group="1",
            ))
        arena.worldbody.append(target)
        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.cube, *self.clutter],
        )

    def _sample_task_pose(self):
        bounds_low = np.array([-0.17, -0.16])
        bounds_high = np.array([0.13, 0.16])
        for _ in range(500):
            target_yaw = self.rng.uniform(-np.pi, np.pi)
            direction = rotation_2d(target_yaw) @ np.array([1.0, 0.0])
            distance = self.rng.uniform(0.105, 0.15)
            lateral = self.rng.uniform(-0.015, 0.015)
            initial_xy = self.rng.uniform(bounds_low, bounds_high)
            target_xy = initial_xy + distance * direction + lateral * np.array([-direction[1], direction[0]])
            # Keep enough clearance for the T footprint and the pushing contact.
            contact_xy = initial_xy - 0.080 * direction
            if (np.all(target_xy > bounds_low) and np.all(target_xy < bounds_high)
                    and np.all(contact_xy > bounds_low - 0.04)
                    and np.all(contact_xy < bounds_high + 0.04)):
                return initial_xy, target_xy, target_yaw
        raise RuntimeError("Could not sample a valid randomized Push-T pose")

    def _reset_internal(self):
        ManipulationEnv._reset_internal(self)
        if self.deterministic_reset:
            return
        initial_xy, self.target_xy, self.target_yaw = self._sample_task_pose()
        initial_yaw = self.target_yaw + self.rng.uniform(-self.initial_yaw_error, self.initial_yaw_error)
        initial_quaternion = [np.cos(initial_yaw / 2), 0, 0, np.sin(initial_yaw / 2)]
        target_quaternion = [np.cos(self.target_yaw / 2), 0, 0, np.sin(self.target_yaw / 2)]
        self.sim.data.set_joint_qpos(self.cube.joints[0], np.r_[initial_xy, 0.815, initial_quaternion])
        self.sim.model.body_pos[self.target_body_id] = np.r_[self.target_xy, 0.8005]
        self.sim.model.body_quat[self.target_body_id] = target_quaternion

        direction = self.target_xy - initial_xy
        direction /= np.linalg.norm(direction)
        placed = []
        for item in self.clutter:
            for _ in range(500):
                xy = self.rng.uniform([-0.19, -0.19], [0.15, 0.19])
                along = np.clip(np.dot(xy - initial_xy, direction), 0.0,
                                np.linalg.norm(self.target_xy - initial_xy))
                path_distance = np.linalg.norm(xy - (initial_xy + along * direction))
                separated = all(np.linalg.norm(xy - other) > 0.045 for other in placed)
                if (path_distance > 0.075 and separated
                        and np.linalg.norm(xy - initial_xy) > 0.09
                        and np.linalg.norm(xy - self.target_xy) > 0.09):
                    break
            else:
                raise RuntimeError("Could not place randomized clutter")
            yaw = self.rng.uniform(-np.pi, np.pi)
            quat = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            self.sim.data.set_joint_qpos(item.joints[0], np.r_[xy, 0.812, quat])
            placed.append(xy)


def default_metadata():

    metadata = base_metadata()
    metadata["env_name"] = "PandaPushTRandom"
    metadata["env_version"] = "robodojo-inspired-random-v1"
    metadata["env_kwargs"].update(
        yaw_range=float(np.pi), initial_yaw_error=0.25, clutter_count=8,
    )
    return metadata
