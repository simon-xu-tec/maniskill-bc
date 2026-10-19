import numpy as np
import torch

from mani_skill.utils.geometry.rotation_conversions import (
    quaternion_to_matrix,
    matrix_to_quaternion,
    matrix_to_euler_angles,
)


# ============================================================
# Expert V3 parameters
# ============================================================

# ---------- grasp approach ----------
SAFE_STANDOFF = 0.07

# 第一次开始闭合时要求比较严格
GRASP_POSITION_TOLERANCE = 0.006       # 6 mm
GRASP_LATERAL_TOLERANCE = 0.004       # 4 mm
GRASP_ROTATION_TOLERANCE = np.deg2rad(3.0)

# 已经开始闭合之后，允许稍宽松一些
FINGER_STARTED_CLOSING_THRESHOLD = 0.038

CONTINUE_CLOSE_POSITION_TOLERANCE = 0.012
CONTINUE_CLOSE_LATERAL_TOLERANCE = 0.008
CONTINUE_CLOSE_ROTATION_TOLERANCE = np.deg2rad(8.0)

# approach 平滑控制
ANGLE_GOOD = np.deg2rad(4.0)
ANGLE_BAD = np.deg2rad(20.0)

LATERAL_GOOD = 0.004
LATERAL_BAD = 0.025

POSITION_SCALE = 0.1
MAX_APPROACH_STEP = 0.010
POSITION_K = 0.55
MIN_TRANSLATION_FACTOR = 0.20

# ---------- rotation ----------
# 保留你当前 ManiSkill nightly 已经验证可以工作的符号
ROTATION_SCALE = -0.1
MAX_ROTATION_STEP = 0.06
ROTATION_K = 0.8

# ---------- transport ----------
CUBE_REST_Z = 0.02

LIFT_RAMP_HEIGHT = 0.020
INITIAL_LIFT_FACTOR = 0.25

HORIZONTAL_START_Z = 0.030
HORIZONTAL_FULL_Z = 0.075

ARC_HEIGHT = 0.10

DESCENT_XY_START = 0.060
DESCENT_XY_FULL = 0.010

MAX_TRANSPORT_STEP = 0.010
TRANSPORT_K = 0.65

GOAL_HOLD_TOLERANCE = 0.005


# ============================================================
# Observation helpers
# ============================================================

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()

    return np.asarray(x)


def scalar_bool(x):
    x = to_numpy(x)
    return bool(x.reshape(-1)[0])


def get_qpos(obs):
    return to_numpy(obs["agent"]["qpos"])[0]


def get_tcp_pose(obs):
    pose = to_numpy(obs["extra"]["tcp_pose"])[0]

    position = pose[:3]
    quaternion = pose[3:]

    return position, quaternion


def get_cube_pose(obs):
    pose = to_numpy(obs["extra"]["obj_pose"])[0]

    position = pose[:3]
    quaternion = pose[3:]

    return position, quaternion


def get_goal_pos(obs):
    return to_numpy(obs["extra"]["goal_pos"])[0]


def get_is_grasped(obs):
    return scalar_bool(obs["extra"]["is_grasped"])


def get_success(info):
    if "success" not in info:
        return False

    return scalar_bool(info["success"])


def get_finger_mean(obs):
    """
    Panda qpos:
        qpos[0:7]  -> 7 个机械臂关节
        qpos[-2:]  -> 两个 finger joint
    """
    qpos = get_qpos(obs)

    return float(np.mean(qpos[-2:]))


# ============================================================
# Rotation helpers
# ============================================================

def quat_to_matrix(q):
    q_tensor = torch.tensor(
        q,
        dtype=torch.float32,
    ).reshape(1, 4)

    R = quaternion_to_matrix(q_tensor)[0]

    return R.cpu().numpy()


def matrix_to_quat(R):
    R_tensor = torch.tensor(
        R,
        dtype=torch.float32,
    ).reshape(1, 3, 3)

    q = matrix_to_quaternion(R_tensor)[0]

    return q.cpu().numpy()


def rotation_angle(R):
    cos_theta = (np.trace(R) - 1.0) / 2.0

    cos_theta = np.clip(
        cos_theta,
        -1.0,
        1.0,
    )

    return float(np.arccos(cos_theta))


# ============================================================
# Smooth interpolation helpers
# ============================================================

def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)

    return x * x * (3.0 - 2.0 * x)


def low_value_is_good(value, good, bad):
    """
    value 小时质量高。

    value <= good -> 1
    value >= bad  -> 0
    中间平滑过渡
    """
    if value <= good:
        return 1.0

    if value >= bad:
        return 0.0

    x = (bad - value) / (bad - good)

    return float(smoothstep(x))


def high_value_is_good(value, bad, good):
    """
    value 大时质量高。

    value <= bad  -> 0
    value >= good -> 1
    中间平滑过渡
    """
    if value <= bad:
        return 0.0

    if value >= good:
        return 1.0

    x = (value - bad) / (good - bad)

    return float(smoothstep(x))


# ============================================================
# Canonical grasp orientation
#
# V3 的核心修改：
#
# 不再从
#   +cube_x
#   -cube_x
#   +cube_y
#   -cube_y
#
# 四个方向中选择最近的一个。
#
# 唯一规定：
#
#   TCP local z = 世界坐标向下
#   TCP local y = cube local +x 在水平面的方向
#
# 因而：
#
#   cube orientation -> 唯一 target orientation
# ============================================================

def build_canonical_grasp_orientation(obs):
    _, tcp_q = get_tcp_pose(obs)
    _, cube_q = get_cube_pose(obs)

    R_tcp = quat_to_matrix(tcp_q)
    R_cube = quat_to_matrix(cube_q)

    # cube local +x 在世界坐标中的方向
    cube_x = R_cube[:, 0]

    # TCP local z 永远竖直向下
    target_z = np.array(
        [0.0, 0.0, -1.0],
        dtype=np.float32,
    )

    # 将 cube local +x 投影到水平面
    target_y = (
        cube_x
        - np.dot(cube_x, target_z) * target_z
    )

    target_y_norm = np.linalg.norm(target_y)

    if target_y_norm < 1e-8:
        raise RuntimeError(
            "Canonical grasp orientation became degenerate."
        )

    target_y = target_y / target_y_norm

    # 构造右手正交坐标系
    target_x = np.cross(
        target_y,
        target_z,
    )

    target_x = (
        target_x
        / np.linalg.norm(target_x)
    )

    # 再正交化一次
    target_y = np.cross(
        target_z,
        target_x,
    )

    target_y = (
        target_y
        / np.linalg.norm(target_y)
    )

    R_target = np.column_stack(
        (
            target_x,
            target_y,
            target_z,
        )
    )

    # 当前姿态相对于目标姿态的误差
    R_error = R_target @ R_tcp.T

    angle = rotation_angle(R_error)

    target_q = matrix_to_quat(R_target)

    return (
        target_q,
        target_z,
        angle,
    )


# ============================================================
# Rotation controller
# ============================================================

def calculate_rotation_action(current_q, target_q):
    R_current = quat_to_matrix(current_q)
    R_target = quat_to_matrix(target_q)

    R_error = R_target @ R_current.T

    total_angle = rotation_angle(R_error)

    if total_angle < 1e-7:
        return (
            np.zeros(3, dtype=np.float32),
            total_angle,
        )

    q_error = matrix_to_quat(R_error)

    # q 和 -q 表示同一个旋转。
    # 强制选择短旋转路径。
    if q_error[0] < 0:
        q_error = -q_error

    w = np.clip(
        q_error[0],
        -1.0,
        1.0,
    )

    sin_half = np.sqrt(
        max(
            1.0 - w * w,
            0.0,
        )
    )

    if sin_half < 1e-7:
        axis = q_error[1:]

        axis_norm = np.linalg.norm(axis)

        if axis_norm < 1e-8:
            return (
                np.zeros(3, dtype=np.float32),
                total_angle,
            )

        axis = axis / axis_norm

    else:
        axis = q_error[1:] / sin_half

    desired_angle = ROTATION_K * total_angle

    step_angle = min(
        desired_angle,
        MAX_ROTATION_STEP,
    )

    half_angle = step_angle / 2.0

    q_step = np.concatenate(
        (
            np.array(
                [np.cos(half_angle)],
                dtype=np.float32,
            ),
            axis.astype(np.float32)
            * np.sin(half_angle),
        )
    )

    R_step = quat_to_matrix(q_step)

    R_step_tensor = torch.tensor(
        R_step,
        dtype=torch.float32,
    ).reshape(1, 3, 3)

    xyz_step = matrix_to_euler_angles(
        R_step_tensor,
        "XYZ",
    )[0].cpu().numpy()

    rotation_action = (
        xyz_step
        / ROTATION_SCALE
    )

    return (
        rotation_action.astype(np.float32),
        total_angle,
    )


# ============================================================
# Translation controllers
# ============================================================

def proportional_translation(
    error,
    gain,
    max_step,
):
    distance = np.linalg.norm(error)

    if distance < 1e-8:
        return np.zeros(
            3,
            dtype=np.float32,
        )

    step_size = min(
        gain * distance,
        max_step,
        distance,
    )

    direction = error / distance

    return (
        direction * step_size
    ).astype(np.float32)


def calculate_approach_translation(
    position_error,
    rotation_error,
):
    base_step = proportional_translation(
        position_error,
        POSITION_K,
        MAX_APPROACH_STEP,
    )

    # 姿态越没有对准，位置移动越慢。
    orientation_quality = low_value_is_good(
        rotation_error,
        ANGLE_GOOD,
        ANGLE_BAD,
    )

    translation_factor = (
        MIN_TRANSLATION_FACTOR
        + (
            1.0
            - MIN_TRANSLATION_FACTOR
        )
        * orientation_quality
    )

    return (
        base_step
        * translation_factor
    ).astype(np.float32)


# ============================================================
# Build ManiSkill action
# ============================================================

def make_action(
    translation_delta,
    rotation_action,
    gripper,
):
    action = np.zeros(
        7,
        dtype=np.float32,
    )

    action[0:3] = (
        translation_delta
        / POSITION_SCALE
    )

    action[3:6] = rotation_action

    action[6] = gripper

    return np.clip(
        action,
        -1.0,
        1.0,
    )


# ============================================================
# Continuous arc transport
# ============================================================

def calculate_transport_delta(
    cube_pos,
    goal_pos,
):
    xy_error = (
        goal_pos[:2]
        - cube_pos[:2]
    )

    xy_distance = np.linalg.norm(
        xy_error
    )

    # cube 越高，越允许进行水平运动
    horizontal_permission = high_value_is_good(
        cube_pos[2],
        HORIZONTAL_START_Z,
        HORIZONTAL_FULL_Z,
    )

    # 水平越靠近 goal，越允许下降
    descent_permission = low_value_is_good(
        xy_distance,
        DESCENT_XY_FULL,
        DESCENT_XY_START,
    )

    # 决定当前路径上的参考高度
    if goal_pos[2] >= ARC_HEIGHT:
        path_target_z = goal_pos[2]

    else:
        path_target_z = (
            ARC_HEIGHT
            + descent_permission
            * (
                goal_pos[2]
                - ARC_HEIGHT
            )
        )

    transport_error = np.array(
        [
            horizontal_permission
            * xy_error[0],

            horizontal_permission
            * xy_error[1],

            path_target_z
            - cube_pos[2],
        ],
        dtype=np.float32,
    )

    translation_delta = proportional_translation(
        transport_error,
        TRANSPORT_K,
        MAX_TRANSPORT_STEP,
    )

    # 刚刚离开桌面时上升速度较慢。
    # 随高度增加逐渐恢复正常速度。
    if translation_delta[2] > 0:
        height_above_rest = max(
            cube_pos[2] - CUBE_REST_Z,
            0.0,
        )

        lift_progress = (
            height_above_rest
            / LIFT_RAMP_HEIGHT
        )

        lift_ramp = smoothstep(
            lift_progress
        )

        lift_factor = (
            INITIAL_LIFT_FACTOR
            + (
                1.0
                - INITIAL_LIFT_FACTOR
            )
            * lift_ramp
        )

        translation_delta[2] *= (
            lift_factor
        )

    return (
        translation_delta,
        float(horizontal_permission),
        float(descent_permission),
        float(path_target_z),
        float(xy_distance),
    )


# ============================================================
# Expert V3
#
# 这是外部脚本真正 import 的函数：
#
#     from expert_v3_policy import expert_action
#
# 它只依赖当前 observation：
#
#     action_t = F(obs_t)
#
# 没有：
#     phase
#     grasp_count
#     previous_obs
#     previous_action
# ============================================================

def expert_action(obs):
    tcp_pos, tcp_q = get_tcp_pose(obs)
    cube_pos, _ = get_cube_pose(obs)
    goal_pos = get_goal_pos(obs)

    is_grasped = get_is_grasped(obs)

    finger_mean = get_finger_mean(obs)

    # ========================================================
    # A. 已经抓住 cube
    # ========================================================

    if is_grasped:
        goal_error = (
            goal_pos
            - cube_pos
        )

        goal_distance = float(
            np.linalg.norm(goal_error)
        )

        # 已经接近目标：
        # 停止主动移动，继续保持夹紧，
        # 等机器人进入 static 状态。
        if goal_distance < GOAL_HOLD_TOLERANCE:
            action = make_action(
                translation_delta=np.zeros(
                    3,
                    dtype=np.float32,
                ),
                rotation_action=np.zeros(
                    3,
                    dtype=np.float32,
                ),
                gripper=-1.0,
            )

            debug = {
                "mode": "HOLD_GOAL",
                "distance": float(
                    np.linalg.norm(
                        tcp_pos - cube_pos
                    )
                ),
                "lateral": 0.0,
                "angle_deg": 0.0,
                "finger_mean": finger_mean,
                "goal_distance": goal_distance,
                "xy_goal_distance": float(
                    np.linalg.norm(
                        goal_error[:2]
                    )
                ),
                "cube_z": float(
                    cube_pos[2]
                ),
                "horizontal_permission": 1.0,
                "descent_permission": 1.0,
                "path_target_z": float(
                    goal_pos[2]
                ),
            }

            return action, debug

        # 连续弧形搬运
        (
            translation_delta,
            horizontal_permission,
            descent_permission,
            path_target_z,
            xy_distance,
        ) = calculate_transport_delta(
            cube_pos,
            goal_pos,
        )

        action = make_action(
            translation_delta=
                translation_delta,

            # 抓住以后不再主动旋转 cube
            rotation_action=np.zeros(
                3,
                dtype=np.float32,
            ),

            gripper=-1.0,
        )

        debug = {
            "mode": "TRANSPORT",
            "distance": float(
                np.linalg.norm(
                    tcp_pos - cube_pos
                )
            ),
            "lateral": 0.0,
            "angle_deg": 0.0,
            "finger_mean": finger_mean,
            "goal_distance": goal_distance,
            "xy_goal_distance": xy_distance,
            "cube_z": float(
                cube_pos[2]
            ),
            "horizontal_permission":
                horizontal_permission,
            "descent_permission":
                descent_permission,
            "path_target_z":
                path_target_z,
        }

        return action, debug

    # ========================================================
    # B. 尚未抓住 cube
    # ========================================================

    (
        target_grasp_q,
        approach_direction,
        rotation_error,
    ) = build_canonical_grasp_orientation(
        obs
    )

    # approach_direction = 世界坐标向下
    #
    # 所以 away_direction = 世界坐标向上
    away_direction = (
        -approach_direction
    )

    cube_to_tcp = (
        tcp_pos
        - cube_pos
    )

    # 沿着竖直方向的分量
    axial_component = (
        np.dot(
            cube_to_tcp,
            away_direction,
        )
        * away_direction
    )

    # 水平偏移
    lateral_vector = (
        cube_to_tcp
        - axial_component
    )

    lateral_error = float(
        np.linalg.norm(
            lateral_vector
        )
    )

    distance_to_cube = float(
        np.linalg.norm(
            tcp_pos
            - cube_pos
        )
    )

    # ========================================================
    # C. 第一次允许开始 close
    # ========================================================

    ready_to_start_close = (
        distance_to_cube
        < GRASP_POSITION_TOLERANCE

        and

        lateral_error
        < GRASP_LATERAL_TOLERANCE

        and

        rotation_error
        < GRASP_ROTATION_TOLERANCE
    )

    # ========================================================
    # D. 当前 finger 是否已经开始闭合
    #
    # 直接由 observation 中 qpos 判断，
    # 不是隐藏状态。
    # ========================================================

    fingers_started_closing = (
        finger_mean
        < FINGER_STARTED_CLOSING_THRESHOLD
    )

    # ========================================================
    # E. 已经开始 close 后：
    #
    # 只要还在合理抓取区域，
    # 就继续 close。
    # ========================================================

    ready_to_continue_close = (
        fingers_started_closing

        and

        distance_to_cube
        < CONTINUE_CLOSE_POSITION_TOLERANCE

        and

        lateral_error
        < CONTINUE_CLOSE_LATERAL_TOLERANCE

        and

        rotation_error
        < CONTINUE_CLOSE_ROTATION_TOLERANCE
    )

    should_close = (
        ready_to_start_close
        or
        ready_to_continue_close
    )

    # ========================================================
    # F. Close
    # ========================================================

    if should_close:
        (
            rotation_action,
            _,
        ) = calculate_rotation_action(
            tcp_q,
            target_grasp_q,
        )

        action = make_action(
            # 开始夹以后不再继续向 cube 冲
            translation_delta=np.zeros(
                3,
                dtype=np.float32,
            ),

            rotation_action=
                rotation_action,

            gripper=-1.0,
        )

        if ready_to_start_close:
            mode = "START_CLOSE"

        else:
            mode = "CONT_CLOSE"

        debug = {
            "mode": mode,
            "distance": distance_to_cube,
            "lateral": lateral_error,
            "angle_deg": float(
                np.rad2deg(
                    rotation_error
                )
            ),
            "finger_mean": finger_mean,
            "goal_distance": float(
                np.linalg.norm(
                    goal_pos - cube_pos
                )
            ),
            "xy_goal_distance": float(
                np.linalg.norm(
                    goal_pos[:2]
                    - cube_pos[:2]
                )
            ),
            "cube_z": float(
                cube_pos[2]
            ),
            "horizontal_permission": 0.0,
            "descent_permission": 0.0,
            "path_target_z": float(
                cube_pos[2]
            ),
        }

        return action, debug

    # ========================================================
    # G. Approach
    # ========================================================

    # cube 正上方 7 cm 的安全点
    safe_pos = (
        cube_pos
        + away_direction
        * SAFE_STANDOFF
    )

    # 姿态越好，越允许向下接近
    orientation_quality = low_value_is_good(
        rotation_error,
        ANGLE_GOOD,
        ANGLE_BAD,
    )

    # 水平方向越对准，越允许向下接近
    lateral_quality = low_value_is_good(
        lateral_error,
        LATERAL_GOOD,
        LATERAL_BAD,
    )

    approach_fraction = (
        orientation_quality
        * lateral_quality
    )

    # 从 safe_pos 连续移动到 cube center
    target_pos = (
        safe_pos
        + approach_fraction
        * (
            cube_pos
            - safe_pos
        )
    )

    position_error = (
        target_pos
        - tcp_pos
    )

    translation_delta = (
        calculate_approach_translation(
            position_error,
            rotation_error,
        )
    )

    (
        rotation_action,
        _,
    ) = calculate_rotation_action(
        tcp_q,
        target_grasp_q,
    )

    action = make_action(
        translation_delta=
            translation_delta,

        rotation_action=
            rotation_action,

        # 没达到抓取条件以前，
        # gripper 永远保持完全打开
        gripper=1.0,
    )

    debug = {
        "mode": "APPROACH",
        "distance": distance_to_cube,
        "lateral": lateral_error,
        "angle_deg": float(
            np.rad2deg(
                rotation_error
            )
        ),
        "finger_mean": finger_mean,
        "goal_distance": float(
            np.linalg.norm(
                goal_pos
                - cube_pos
            )
        ),
        "xy_goal_distance": float(
            np.linalg.norm(
                goal_pos[:2]
                - cube_pos[:2]
            )
        ),
        "cube_z": float(
            cube_pos[2]
        ),
        "horizontal_permission": 0.0,
        "descent_permission": 0.0,
        "path_target_z": float(
            cube_pos[2]
        ),
    }

    return action, debug