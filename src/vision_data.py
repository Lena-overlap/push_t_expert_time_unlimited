"""Synchronous RGB + proprioception recording, independent of expert decisions."""
from pathlib import Path
import json
import numpy as np
from dataset import DemoRecorder

PROPRIO_KEYS = ('robot0_joint_pos_sin', 'robot0_joint_pos_cos', 'robot0_gripper_qpos')
DEFAULT_CAMERAS = ('birdview', 'agentview')
SCHEMA = 'push_t_act_v1'


def proprio(obs):
    # robosuite 1.5 marks the direct joint_pos observable inactive by default.
    # Use the same wrapped joint representation in collection and deployment.
    if all(k in obs for k in PROPRIO_KEYS[:2]):
        joint = np.arctan2(np.asarray(obs[PROPRIO_KEYS[0]]), np.asarray(obs[PROPRIO_KEYS[1]]))
    elif 'robot0_joint_pos' in obs:
        raw = np.asarray(obs['robot0_joint_pos']); joint = np.arctan2(np.sin(raw), np.cos(raw))
    else:
        raise KeyError(f'Missing joint observations; available: {list(obs)}')
    grip = np.asarray(obs['robot0_gripper_qpos'], dtype=np.float32).reshape(-1)
    joint = np.asarray(joint, dtype=np.float32).reshape(-1)
    q = np.concatenate((joint, grip))
    if len(joint) != 7 or len(grip) < 1 or not np.isfinite(q).all():
        raise ValueError('Invalid Panda joint/gripper observation')
    return q


def rgb(env, camera, size):
    # Same raw MuJoCo orientation correction as the original Video recorder.
    a = np.asarray(env.sim.render(camera_name=camera, width=size, height=size, depth=False))
    if a.shape != (size, size, 3) or a.dtype != np.uint8:
        raise ValueError(f'Unexpected RGB from {camera}: {a.shape}, {a.dtype}')
    return np.ascontiguousarray(a[::-1])


class VisionRecorder(DemoRecorder):
    """Stream images to a temporary HDF5, publish only successful episodes."""
    def __init__(self, path, cameras=DEFAULT_CAMERAS, size=224):
        super().__init__()
        self.path = Path(path)
        self.tmp = self.path.with_suffix('.partial.hdf5')
        self.cameras = tuple(cameras)
        self.size = int(size)
        self.file = None
        self.q = []
        if not self.cameras or len(set(self.cameras)) != len(self.cameras):
            raise ValueError('Specify unique camera names')

    def observe(self, obs, env, metrics, ok):
        import h5py
        # Capture BEFORE the next action, at precisely the state being recorded.
        q = proprio(obs)
        frames = [rgb(env, c, self.size) for c in self.cameras]
        if self.file is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists() or self.tmp.exists():
                raise FileExistsError(self.path)
            self.file = h5py.File(self.tmp, 'x')
            self.file.attrs['schema'] = SCHEMA
            self.file.attrs['cameras_json'] = json.dumps(self.cameras)
            self.file.attrs['proprio_keys_json'] = json.dumps(PROPRIO_KEYS)
            g = self.file.require_group('observations/images')
            for c in self.cameras:
                g.create_dataset(c, shape=(0, self.size, self.size, 3),
                                 maxshape=(None, self.size, self.size, 3),
                                 chunks=(1, self.size, self.size, 3), dtype='u1', compression='lzf')
        i = len(self.q)
        for c, frame in zip(self.cameras, frames):
            d = self.file['observations/images/'+c]
            d.resize(i+1, axis=0); d[i] = frame
        self.q.append(q)
        super().observe(obs, env, metrics, ok)

    def save(self, path, metadata):
        # Retain the native state NPZ. Images never enter it.
        t = len(self.actions)
        if self.file is None or t < 1 or len(self.q) != t+1 or len(self.obs) != t+1:
            raise ValueError('Expected T+1 synchronized observations for T actions')
        stable = int(metadata['stable_steps'])
        if stable < 1 or stable > t or not all(self.success[-stable:]):
            raise ValueError('Not a stable successful demonstration')
        for c in self.cameras:
            if len(self.file['observations/images/'+c]) != t+1:
                raise ValueError('Camera length mismatch')
        f = self.file
        f.create_dataset('observations/qpos', data=np.stack(self.q))
        f.create_dataset('action', data=np.asarray(self.actions, np.float32))
        f.create_dataset('task/target_xy_yaw', data=np.asarray(self.target))
        f.create_dataset('task/iou', data=np.asarray(self.ious, np.float32))
        f.create_dataset('task/success', data=np.asarray(self.success, bool))
        f.attrs['metadata_json'] = json.dumps(metadata, ensure_ascii=False)
        f.attrs['sim'] = True
        f.attrs['control_hz'] = 20
        f.flush(); f.close(); self.file = None
        saved = super().save(path, metadata)
        self.tmp.replace(self.path)
        return saved

    def close(self):
        if self.file is not None:
            self.file.close(); self.file = None
        self.tmp.unlink(missing_ok=True)
