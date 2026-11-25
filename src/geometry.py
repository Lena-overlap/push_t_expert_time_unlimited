"""T-shape geometry, contact candidates, pose and overlap utilities."""
from __future__ import annotations

import numpy as np
import json
from shapely.geometry import Polygon
from dataclasses import dataclass
from pathlib import Path


def rotation_2d(angle):
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


OUTLINE=np.array([[-.06,-.06],[-.025,-.06],[-.025,-.0175],[.06,-.0175],[.06,.0175],[-.025,.0175],[-.025,.06],[-.06,.06]])
TARGET=Polygon(OUTLINE)
rot = rotation_2d
# [local dx, local dy, yaw rad] per metre of positive forward object travel.
# Median of closed-gripper measurements, positive forward travel >6 mm.
# -X is an uncalibrated symmetric prior; other five measured n=217,22,11,14,6.
PRIORS={'xp_center':(1.,-.0092,-.0974),'xn_center':(-1.,0.,0.),
        'yp_bar_near':(.1429,1.,-3.2524),'yp_stem_mid':(-.075,1.,5.7663),
        'yn_bar_near':(.130,-1.,2.6522),'yn_stem_mid':(-.0747,-1.,-6.1179)}
POINTS={'xp_center':(-.06,0),'xn_center':(.06,0),'yp_bar_near':(-.034,-.06),
        'yp_stem_mid':(.012,-.0175),'yn_bar_near':(-.034,.06),'yn_stem_mid':(.012,.0175)}
NORMALS={k:np.array([1.,0.]) if k.startswith('xp') else np.array([-1.,0.]) if k.startswith('xn') else np.array([0.,1. if k.startswith('yp') else -1.]) for k in PRIORS}
CAPS={k:.115 if k[0]=='x' else .09 for k in PRIORS}
@dataclass
class Plan:
    names:tuple
    distances:np.ndarray
    predicted_iou:float
    estimated_steps:int
    residual:np.ndarray
    score:float

def predict_iou(pose):
    obj=Polygon(OUTLINE@rot(pose[2]).T+pose[:2]); inter=obj.intersection(TARGET).area
    return float(inter/(obj.area+TARGET.area-inter))


@dataclass(frozen=True)
class FaceCandidate:
    name: str
    direction: str
    xy: tuple[float, float]
    normal: tuple[float, float]
    surface: str


# Coordinates are on external T surfaces; no point is inside the solid T.
# Multiple points per face help identify actual gripper clearance / reach.
FACES = (
    FaceCandidate('xp_center', '+x', (-.060, .000), (1, 0), 'crossbar_left'),
    FaceCandidate('xp_upper', '+x', (-.060, .014), (1, 0), 'crossbar_left'),
    FaceCandidate('xp_lower', '+x', (-.060, -.014), (1, 0), 'crossbar_left'),
    FaceCandidate('xn_center', '-x', (.060, .000), (-1, 0), 'stem_right'),
    FaceCandidate('xn_upper', '-x', (.060, .007), (-1, 0), 'stem_right'),
    FaceCandidate('xn_lower', '-x', (.060, -.007), (-1, 0), 'stem_right'),
    FaceCandidate('yp_bar_near', '+y', (-.034, -.060), (0, 1), 'crossbar_bottom'),
    FaceCandidate('yp_bar_mid', '+y', (-.043, -.060), (0, 1), 'crossbar_bottom'),
    FaceCandidate('yp_bar_far', '+y', (-.052, -.060), (0, 1), 'crossbar_bottom'),
    FaceCandidate('yp_stem_mid', '+y', (.012, -.0175), (0, 1), 'stem_bottom'),
    FaceCandidate('yp_stem_tip', '+y', (.037, -.0175), (0, 1), 'stem_bottom'),
    FaceCandidate('yn_bar_near', '-y', (-.034, .060), (0, -1), 'crossbar_top'),
    FaceCandidate('yn_bar_mid', '-y', (-.043, .060), (0, -1), 'crossbar_top'),
    FaceCandidate('yn_bar_far', '-y', (-.052, .060), (0, -1), 'crossbar_top'),
    FaceCandidate('yn_stem_mid', '-y', (.012, .0175), (0, -1), 'stem_top'),
    FaceCandidate('yn_stem_tip', '-y', (.037, .0175), (0, -1), 'stem_top'),
)
FACE_BY_NAME = {f.name: f for f in FACES}
DIRECTIONS = ('+x', '-x', '+y', '-y')


def yaw_bin(yaw: float, count: int) -> int:
    return min(count - 1, int(((float(yaw) + np.pi) % (2 * np.pi)) / (2 * np.pi) * count))


def wrapped(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def angular_pose(environment):
    data = environment.sim.data
    xy = data.body_xpos[environment.cube_body_id, :2].copy()
    mat = data.body_xmat[environment.cube_body_id].reshape(3, 3)
    yaw = float(np.arctan2(mat[1, 0], mat[0, 0]))
    return xy, yaw


def _is_child_of(model, geom, root):
    """MuJoCo body hierarchy includes descendants of the Tee freejoint body."""
    try:
        body = int(model.geom_bodyid[int(geom)])
        for _ in range(100):
            if body == root:
                return True
            parent = int(model.body_parentid[body])
            if parent == body or body == 0:
                break
            body = parent
    except (AttributeError, IndexError, TypeError, ValueError):
        pass
    return False


def _geom_body_name(model, geom):
    try:
        bid = int(model.geom_bodyid[int(geom)])
        if hasattr(model, 'body_id2name'):
            return str(model.body_id2name(bid) or '').lower()
        import mujoco
        return str(mujoco.mj_id2name(model._model, mujoco.mjtObj.mjOBJ_BODY, bid) or '').lower()
    except (AttributeError, TypeError, ValueError, IndexError, ImportError):
        return ''


def tee_robot_contact(environment):
    """(API_available, Tee touching Panda). Unknown means NOT VERIFIED.

    Uses geom/body naming for identifying robot collisions; for a custom robot
    naming scheme, adapt this function or inspect the actual contact pairs.
    """
    sim = environment.sim
    if not hasattr(sim.data, 'ncon') or not hasattr(sim.data, 'contact'):
        return False, False
    try:
        for i in range(int(sim.data.ncon)):
            c = sim.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            t1 = _is_child_of(sim.model, g1, int(environment.cube_body_id))
            t2 = _is_child_of(sim.model, g2, int(environment.cube_body_id))
            if t1 == t2:
                continue
            other_geom = g2 if t1 else g1
            other_name = _geom_body_name(sim.model, other_geom)
            if any(s in other_name for s in ('panda', 'robot', 'gripper', 'finger', 'hand')):
                return True, True
        return True, False
    except (AttributeError, IndexError, TypeError, ValueError):
        return False, False


def read_face_model(model_path, environment=None):
    if not model_path:
        return None
    model = json.loads(Path(model_path).read_text(encoding='utf-8'))
    if model.get('version') != 1 or model.get('kind') != 'four_way_faces':
        raise ValueError('Unsupported four-face model. Re-run probe mode.')
    if int(model.get('yaw_bins', 0)) < 1:
        raise ValueError('Invalid yaw_bins in four-face model')
    if environment is not None:
        env_name = ('random' if type(environment).__name__ == 'PandaPushTRandom'
                    else 'independent' if getattr(environment, 'independent_poses', False)
                    else 'baseline')
        if model.get('environment') != env_name:
            raise ValueError(f'Four-face model environment mismatch: {model.get("environment")} vs {env_name}')
    return model
