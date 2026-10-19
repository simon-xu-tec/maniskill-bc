import numpy as np
import torch
import gymnasium as gym
import mani_skill.envs

from mani_skill.utils.wrappers.record import RecordEpisode

from mani_skill.utils.geometry.rotation_conversions import (
    quaternion_to_matrix,
    matrix_to_quaternion,
    matrix_to_euler_angles,
)


# ============================================================
# 0. 数据集设置
# ============================================================

# 最终希望得到多少条成功的 expert demonstration
NUM_SUCCESSFUL_DEMOS = 100

# 最多尝试多少个随机 seed
MAX_SEEDS_TO_TRY = 500

# 输出目录
OUTPUT_DIR = "data/expert_demos"

# 输出文件名：
#
# pickcube_scripted.h5
# pickcube_scripted.json
TRAJECTORY_NAME = "pickcube_scripted"


# ============================================================
# 1. Expert policy 参数
# ============================================================

# 距离 cube 7 cm 时，
# 从远距离 approach 进入最终精细接近阶段
PREGRASP_DISTANCE = 0.07

# 抓住 cube 后先抬高 8 cm
LIFT_DISTANCE = 0.08


# ============================================================
# 2. 每个小阶段自己的最大 step
#
# 注意：
#
# 这些不是 Gymnasium 的 truncated。
#
# 它们只是我们自己规定：
# 一个 expert phase 做太久还没完成，
# 就认为这次 expert 尝试失败。
# ============================================================

MAX_APPROACH_STEPS = 80
MAX_FINAL_APPROACH_STEPS = 40
MAX_GRASP_STEPS = 30
MAX_LIFT_STEPS = 50
MAX_GOAL_STEPS = 100
MAX_STABILIZE_STEPS = 40


# ============================================================
# 3. 小阶段完成条件
# ============================================================

# TCP 距目标位置小于 4 mm
POSITION_TOLERANCE = 0.004

# TCP 姿态误差小于 2°
ROTATION_TOLERANCE = np.deg2rad(2.0)

# Cube 距 goal 小于 5 mm 后，
# 停止主动搬运，进入 stabilize。
#
# ManiSkill 官方真正 success 的位置阈值
# 是 2.5 cm，我们的 expert 控制得更精确。
CUBE_GOAL_TOLERANCE = 0.005


# ============================================================
# 4. pd_ee_delta_pose 参数
# ============================================================

# translation normalized action:
#
# ±1 约对应 ±0.1 m
POSITION_SCALE = 0.1

# 每一个 control step
# 最多要求 TCP 平移 1 cm
MAX_POSITION_STEP = 0.01


# 当前 ManiSkill PDEEPoseController 源码
# 对 rotation normalized action
# 仍然乘 rot_lower。
#
# Panda 的 rot_lower = -0.1。
#
# 所以保持和你已经测试成功的单 episode 程序一致。
ROTATION_SCALE = -0.1

# 每一个 control step
# 最多要求旋转 0.04 rad
#
# 约 2.29°
MAX_ROTATION_STEP = 0.04


# ============================================================
# 5. 创建环境
# ============================================================

base_env = gym.make(
    "PickCube-v1",

    # 使用结构化 observation
    obs_mode="state_dict",

    # action:
    #
    # [dx, dy, dz,
    #  dRx, dRy, dRz,
    #  gripper]
    control_mode="pd_ee_delta_pose",

    # 批量采数据，不需要 GUI
    render_mode=None,

    # PickCube 官方默认 50 steps。
    #
    # 我们 scripted expert 比较细，
    # 所以放宽到 300。
    max_episode_steps=300,
)


# ============================================================
# 6. RecordEpisode
# ============================================================

env = RecordEpisode(
    base_env,

    output_dir=OUTPUT_DIR,

    trajectory_name=TRAJECTORY_NAME,

    save_trajectory=True,

    # 不录视频
    save_video=False,

    # 非常重要：
    #
    # reset 时不要自动保存上一条 trajectory。
    #
    # 我们自己决定：
    #
    # success -> save=True
    # failed  -> save=False
    save_on_reset=False,

    # 第一版 Behavior Cloning 不使用 reward
    record_reward=False,

    # 第一版 BC 不需要完整 simulator state
    record_env_state=False,

    source_type="scripted",

    source_desc=(
        "Scripted expert for PickCube-v1. "
        "Approach and align, final approach, grasp, lift, "
        "then directly move cube toward goal using cube-goal error."
    ),
)


# ============================================================
# 7. 基础工具函数
# ============================================================

def to_numpy(x):
    """
    torch.Tensor -> numpy.ndarray
    """

    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()

    return np.asarray(x)


def scalar_bool(x):
    """
    例如：

        tensor([True])

    转成：

        True
    """

    x = to_numpy(x)

    return bool(
        x.reshape(-1)[0]
    )


# ============================================================
# 8. Observation 读取
# ============================================================

def get_tcp_pose(obs):
    """
    TCP pose:

        [x, y, z, qw, qx, qy, qz]

    返回：

        position     shape (3,)
        quaternion   shape (4,)
    """

    pose = to_numpy(
        obs["extra"]["tcp_pose"]
    )[0]

    return (
        pose[:3],
        pose[3:],
    )


def get_cube_pose(obs):
    """
    Cube 世界坐标位姿：

        [x, y, z, qw, qx, qy, qz]
    """

    pose = to_numpy(
        obs["extra"]["obj_pose"]
    )[0]

    return (
        pose[:3],
        pose[3:],
    )


def get_goal_pos(obs):
    """
    绿色 goal 中心的位置。
    """

    return to_numpy(
        obs["extra"]["goal_pos"]
    )[0]


def get_is_grasped(obs):
    """
    ManiSkill 是否认为 cube 已经被抓住。
    """

    return scalar_bool(
        obs["extra"]["is_grasped"]
    )


def get_success(info):
    """
    读取 PickCube 官方 success。
    """

    if "success" not in info:
        return False

    return scalar_bool(
        info["success"]
    )


# ============================================================
# 9. Quaternion <-> Rotation Matrix
# ============================================================

def quat_to_matrix(q):
    """
    ManiSkill quaternion 顺序：

        [qw, qx, qy, qz]

    转成：

        3 x 3 rotation matrix
    """

    q_tensor = torch.tensor(
        q,
        dtype=torch.float32,
    ).reshape(1, 4)

    R = quaternion_to_matrix(
        q_tensor
    )[0]

    return R.cpu().numpy()


def matrix_to_quat(R):
    """
    3 x 3 rotation matrix

        ->

    quaternion [qw, qx, qy, qz]
    """

    R_tensor = torch.tensor(
        R,
        dtype=torch.float32,
    ).reshape(1, 3, 3)

    q = matrix_to_quaternion(
        R_tensor
    )[0]

    return q.cpu().numpy()


# ============================================================
# 10. 一个 rotation matrix 对应的总旋转角
# ============================================================

def rotation_angle(R):
    """
    theta =
        arccos((Tr(R) - 1) / 2)

    返回单位：

        rad
    """

    cos_theta = (
        np.trace(R) - 1.0
    ) / 2.0

    cos_theta = np.clip(
        cos_theta,
        -1.0,
        1.0,
    )

    return np.arccos(
        cos_theta
    )


# ============================================================
# 11. 根据 Cube 姿态构造目标抓取姿态
# ============================================================

def build_grasp_orientation(obs):
    """
    我们要求：

        z_TCP = -z_cube

    即：
        TCP 朝向 cube 顶面。


    Panda TCP local y
    是夹爪 closing direction。


    对正方形：

        +x_cube
        -x_cube
        +y_cube
        -y_cube

    四种 closing direction 都可以。


    我们分别构造四个完整 TCP 姿态，
    然后选择：

        当前 TCP -> 目标 TCP

    所需总旋转角最小的那个。
    """


    # --------------------------------------------------------
    # 当前 TCP 姿态
    # --------------------------------------------------------

    _, tcp_q = get_tcp_pose(
        obs
    )

    R_tcp = quat_to_matrix(
        tcp_q
    )


    # --------------------------------------------------------
    # Cube 姿态
    # --------------------------------------------------------

    _, cube_q = get_cube_pose(
        obs
    )

    R_cube = quat_to_matrix(
        cube_q
    )


    # --------------------------------------------------------
    # Cube local xyz
    # 在世界坐标系中的方向
    # --------------------------------------------------------

    cube_x = R_cube[:, 0]
    cube_y = R_cube[:, 1]
    cube_z = R_cube[:, 2]


    # ========================================================
    # TCP target z
    #
    # z_TCP = -z_cube
    # ========================================================

    target_z = -cube_z

    target_z = (
        target_z
        / np.linalg.norm(target_z)
    )


    # ========================================================
    # 正方形的四个等价 closing direction
    # ========================================================

    closing_candidates = [
        cube_x,
        -cube_x,
        cube_y,
        -cube_y,
    ]


    best_R = None
    best_angle = None


    for candidate_y in closing_candidates:

        target_y = (
            candidate_y
            / np.linalg.norm(candidate_y)
        )


        # ----------------------------------------------------
        # 已知 TCP target y 和 target z。
        #
        # 右手系：
        #
        # x = y × z
        # ----------------------------------------------------

        target_x = np.cross(
            target_y,
            target_z,
        )

        target_x = (
            target_x
            / np.linalg.norm(target_x)
        )


        # ----------------------------------------------------
        # 再重新算一次 y，
        # 去掉数值误差，保证严格正交。
        # ----------------------------------------------------

        target_y = np.cross(
            target_z,
            target_x,
        )

        target_y = (
            target_y
            / np.linalg.norm(target_y)
        )


        # ----------------------------------------------------
        # Rotation matrix 三列：
        #
        # local x 在 world 中的方向
        # local y 在 world 中的方向
        # local z 在 world 中的方向
        # ----------------------------------------------------

        R_target = np.column_stack(
            (
                target_x,
                target_y,
                target_z,
            )
        )


        # ----------------------------------------------------
        # 当前 TCP -> 目标 TCP
        # ----------------------------------------------------

        R_error = (
            R_target
            @ R_tcp.T
        )


        angle = rotation_angle(
            R_error
        )


        # 选择整体旋转最少的抓法
        if (
            best_angle is None
            or angle < best_angle
        ):

            best_angle = angle
            best_R = R_target


    target_q = matrix_to_quat(
        best_R
    )


    return (
        target_q,
        target_z,
    )


# ============================================================
# 12. 限制向量长度
# ============================================================

def limit_norm(
    vector,
    max_norm,
):

    norm = np.linalg.norm(
        vector
    )


    if norm <= max_norm:
        return vector


    return (
        vector
        / norm
        * max_norm
    )


# ============================================================
# 13. 当前姿态 -> 本 control step 的旋转增量
# ============================================================

def calculate_rotation_step(
    current_q,
    target_q,
):


    R_current = quat_to_matrix(
        current_q
    )

    R_target = quat_to_matrix(
        target_q
    )


    # ========================================================
    # 完整相对旋转
    #
    # R_target =
    # R_error @ R_current
    #
    # 所以：
    #
    # R_error =
    # R_target @ R_current.T
    # ========================================================

    R_error = (
        R_target
        @ R_current.T
    )


    total_angle = rotation_angle(
        R_error
    )


    # 已经基本完全对准
    if total_angle < 1e-6:

        return (
            np.zeros(
                3,
                dtype=np.float32,
            ),

            total_angle,
        )


    # ========================================================
    # R_error -> quaternion
    # ========================================================

    q_error = matrix_to_quat(
        R_error
    )


    # q 和 -q 表示同一个 rotation。
    #
    # 令 qw >= 0，
    # 保证选择较短旋转。
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


    # ========================================================
    # 求 rotation axis
    # ========================================================

    if sin_half < 1e-6:

        axis = q_error[1:]

        axis_norm = np.linalg.norm(
            axis
        )


        if axis_norm < 1e-8:

            return (
                np.zeros(
                    3,
                    dtype=np.float32,
                ),

                total_angle,
            )


        axis = (
            axis
            / axis_norm
        )


    else:

        axis = (
            q_error[1:]
            / sin_half
        )


    # ========================================================
    # 本 step 最多只执行 MAX_ROTATION_STEP
    # ========================================================

    step_angle = min(
        total_angle,
        MAX_ROTATION_STEP,
    )


    half_angle = (
        step_angle / 2.0
    )


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


    # ========================================================
    # 小旋转 quaternion
    # ->
    # XYZ Euler delta
    #
    # 因为 ManiSkill PDEEPoseController
    # 内部使用 XYZ Euler delta。
    # ========================================================

    R_step = quat_to_matrix(
        q_step
    )


    R_step_tensor = torch.tensor(
        R_step,
        dtype=torch.float32,
    ).reshape(1, 3, 3)


    xyz_step = matrix_to_euler_angles(
        R_step_tensor,
        "XYZ",
    )[0]


    return (
        xyz_step.cpu().numpy(),
        total_angle,
    )


# ============================================================
# 14. 普通 TCP pose -> action
# ============================================================

def make_pose_action(
    obs,
    target_pos,
    target_q,
    gripper,
):


    current_pos, current_q = (
        get_tcp_pose(obs)
    )


    # ========================================================
    # A. Position
    # ========================================================

    position_error = (
        target_pos
        - current_pos
    )


    position_step = limit_norm(
        position_error,
        MAX_POSITION_STEP,
    )


    position_action = (
        position_step
        / POSITION_SCALE
    )


    # ========================================================
    # B. Rotation
    # ========================================================

    (
        rotation_step,
        _,
    ) = calculate_rotation_step(
        current_q,
        target_q,
    )


    rotation_action = (
        rotation_step
        / ROTATION_SCALE
    )


    # ========================================================
    # C. 7D action
    # ========================================================

    action = np.zeros(
        env.action_space.shape,
        dtype=np.float32,
    )


    action[0:3] = (
        position_action
    )

    action[3:6] = (
        rotation_action
    )

    action[6] = gripper


    # normalized action
    # 最终必须在 [-1, 1]
    action = np.clip(
        action,
        -1.0,
        1.0,
    )


    return action


# ============================================================
# 15. 判断当前 TCP 是否到达某个目标 pose
# ============================================================

def reached_pose(
    obs,
    target_pos,
    target_q,
):


    current_pos, current_q = (
        get_tcp_pose(obs)
    )


    # --------------------------------------------------------
    # 位置误差
    # --------------------------------------------------------

    position_error = np.linalg.norm(
        target_pos
        - current_pos
    )


    # --------------------------------------------------------
    # 姿态误差
    # --------------------------------------------------------

    R_current = quat_to_matrix(
        current_q
    )

    R_target = quat_to_matrix(
        target_q
    )


    R_error = (
        R_target
        @ R_current.T
    )


    rotation_error = rotation_angle(
        R_error
    )


    return (
        position_error < POSITION_TOLERANCE

        and

        rotation_error < ROTATION_TOLERANCE
    )


# ============================================================
# 16. 普通 TCP movement phase
# ============================================================

def move_to_pose(
    obs,
    target_pos,
    target_q,
    gripper,
    max_steps,
    require_grasp=False,
):
    """
    返回：

        obs
        info
        phase_ok
        task_success
        used_steps


    phase_ok:
        当前小阶段是否完成。


    task_success:
        整个 PickCube 是否已经成功。
    """


    info = {}

    used_steps = 0


    for _ in range(max_steps):


        action = make_pose_action(
            obs,
            target_pos,
            target_q,
            gripper,
        )


        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            action
        )


        used_steps += 1


        # ====================================================
        # 整个任务已经成功
        # ====================================================

        if get_success(info):

            return (
                obs,
                info,
                True,
                True,
                used_steps,
            )


        # ====================================================
        # Environment episode 已结束
        # ====================================================

        if (
            scalar_bool(terminated)
            or
            scalar_bool(truncated)
        ):

            return (
                obs,
                info,
                False,
                False,
                used_steps,
            )


        # ====================================================
        # 必须保持抓取的阶段，
        # 如果 cube 掉了，
        # expert 立即失败。
        # ====================================================

        if (
            require_grasp
            and
            not get_is_grasped(obs)
        ):

            return (
                obs,
                info,
                False,
                False,
                used_steps,
            )


        # ====================================================
        # 当前小阶段目标已经完成
        # ====================================================

        if reached_pose(
            obs,
            target_pos,
            target_q,
        ):

            return (
                obs,
                info,
                True,
                False,
                used_steps,
            )


    # ========================================================
    # 小阶段自己的 step 已经用完，
    # 但还没有到达目标。
    #
    # 这是 expert_failed。
    #
    # 此时：
    #
    # terminated=False
    # truncated=False
    #
    # 完全可能成立。
    # ========================================================

    return (
        obs,
        info,
        False,
        False,
        used_steps,
    )


# ============================================================
# 17. Grasp phase
# ============================================================

def grasp_cube(
    obs,
    target_q,
):


    info = {}

    used_steps = 0


    # 闭合过程中保持 TCP 当前位置
    hold_pos, _ = get_tcp_pose(
        obs
    )


    # 连续 3 个 control steps
    # 都检测到 is_grasped
    # 才认为抓取稳定。
    grasp_count = 0


    for _ in range(
        MAX_GRASP_STEPS
    ):


        action = make_pose_action(
            obs,
            hold_pos,
            target_q,

            # -1 = close
            gripper=-1.0,
        )


        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            action
        )


        used_steps += 1


        if get_success(info):

            return (
                obs,
                info,
                True,
                True,
                used_steps,
            )


        if (
            scalar_bool(terminated)
            or
            scalar_bool(truncated)
        ):

            return (
                obs,
                info,
                False,
                False,
                used_steps,
            )


        if get_is_grasped(obs):

            grasp_count += 1

        else:

            grasp_count = 0


        if grasp_count >= 3:

            return (
                obs,
                info,
                True,
                False,
                used_steps,
            )


    # MAX_GRASP_STEPS 用完仍然没抓稳
    return (
        obs,
        info,
        False,
        False,
        used_steps,
    )


# ============================================================
# 18. 抓住以后：
#     cube-goal error -> action
# ============================================================

def make_cube_to_goal_action(
    obs,
    target_q,
):
    """
    这里不再计算固定 TCP target。

    每一个 control step 都重新计算：

        cube_error
        =
        goal_pos - cube_pos


    然后让 TCP 沿同一个方向移动。


    所以：

        当前 cube
            ↓
        当前 goal
            ↓
        当前误差
            ↓
        当前 action


    下一步重新观察、重新计算。
    """


    cube_pos, _ = get_cube_pose(
        obs
    )

    goal_pos = get_goal_pos(
        obs
    )

    _, current_tcp_q = get_tcp_pose(
        obs
    )


    # ========================================================
    # Cube 当前距离 goal 还差多少
    # ========================================================

    cube_error = (
        goal_pos
        - cube_pos
    )


    # ========================================================
    # 每一步最多移动 1 cm
    # ========================================================

    position_step = limit_norm(
        cube_error,
        MAX_POSITION_STEP,
    )


    position_action = (
        position_step
        / POSITION_SCALE
    )


    # ========================================================
    # 搬运时继续保持抓取姿态
    # ========================================================

    (
        rotation_step,
        _,
    ) = calculate_rotation_step(
        current_tcp_q,
        target_q,
    )


    rotation_action = (
        rotation_step
        / ROTATION_SCALE
    )


    # ========================================================
    # 完整 action
    # ========================================================

    action = np.zeros(
        env.action_space.shape,
        dtype=np.float32,
    )


    action[0:3] = (
        position_action
    )

    action[3:6] = (
        rotation_action
    )

    # 搬运过程中继续夹紧
    action[6] = -1.0


    return np.clip(
        action,
        -1.0,
        1.0,
    )


# ============================================================
# 19. 直接把 Cube 搬到 Goal
# ============================================================

def move_cube_to_goal(
    obs,
    target_q,
):


    info = {}

    used_steps = 0


    for _ in range(
        MAX_GOAL_STEPS
    ):


        # ----------------------------------------------------
        # 如果开始这一 step 前就已经掉了 cube
        # ----------------------------------------------------

        if not get_is_grasped(obs):

            return (
                obs,
                info,
                False,
                False,
                used_steps,
            )


        # ----------------------------------------------------
        # 当前 cube-goal error
        # 直接生成 action
        # ----------------------------------------------------

        action = make_cube_to_goal_action(
            obs,
            target_q,
        )


        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            action
        )


        used_steps += 1


        # ----------------------------------------------------
        # 官方 success 已经满足
        # ----------------------------------------------------

        if get_success(info):

            return (
                obs,
                info,
                True,
                True,
                used_steps,
            )


        # ----------------------------------------------------
        # 整个 episode 已结束
        # ----------------------------------------------------

        if (
            scalar_bool(terminated)
            or
            scalar_bool(truncated)
        ):

            return (
                obs,
                info,
                False,
                False,
                used_steps,
            )


        # ----------------------------------------------------
        # 搬运途中掉了 cube
        # ----------------------------------------------------

        if not get_is_grasped(obs):

            return (
                obs,
                info,
                False,
                False,
                used_steps,
            )


        # ----------------------------------------------------
        # 检查 cube 当前距离 goal 多远
        # ----------------------------------------------------

        cube_pos, _ = get_cube_pose(
            obs
        )

        goal_pos = get_goal_pos(
            obs
        )


        cube_goal_distance = np.linalg.norm(
            goal_pos
            - cube_pos
        )


        # ----------------------------------------------------
        # 已经达到我们 expert 自己的 5 mm 精度，
        # 进入 stabilize。
        # ----------------------------------------------------

        if (
            cube_goal_distance
            < CUBE_GOAL_TOLERANCE
        ):

            return (
                obs,
                info,
                True,
                False,
                used_steps,
            )


    # MAX_GOAL_STEPS 用完仍未到 goal
    return (
        obs,
        info,
        False,
        False,
        used_steps,
    )


# ============================================================
# 20. Stabilize
# ============================================================

def stabilize(
    obs,
    target_q,
):
    """
    Cube 已经到 goal 附近。

    现在不再继续往 goal 主动移动。

    保持当前 TCP position，
    同时保持抓取姿态和夹紧。

    等机器人 qvel 降下来以后，
    ManiSkill success=True。
    """


    info = {}

    used_steps = 0


    hold_pos, _ = get_tcp_pose(
        obs
    )


    for _ in range(
        MAX_STABILIZE_STEPS
    ):


        action = make_pose_action(
            obs,
            hold_pos,
            target_q,

            gripper=-1.0,
        )


        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            action
        )


        used_steps += 1


        if get_success(info):

            return (
                obs,
                info,
                True,
                used_steps,
            )


        if (
            scalar_bool(terminated)
            or
            scalar_bool(truncated)
        ):

            return (
                obs,
                info,
                False,
                used_steps,
            )


        if not get_is_grasped(obs):

            return (
                obs,
                info,
                False,
                used_steps,
            )


    return (
        obs,
        info,
        False,
        used_steps,
    )


# ============================================================
# 21. 一个完整 Expert Episode
# ============================================================

def run_expert_episode(seed):
    """
    一个 seed = 一个新的 episode。


    返回：

        status

            "success"
            "failed"
            "skip"

        total_steps

        reason
    """


    # ========================================================
    # RESET
    # ========================================================

    obs, info = env.reset(
        seed=seed
    )


    # ========================================================
    # 特殊情况：
    #
    # reset 后任务本身已经 success。
    #
    # 这种场景是合法场景，
    # 不是 failure。
    #
    # 但是它没有任何 expert action 可以学习，
    # 所以不写入 BC demo。
    # ========================================================

    initial_eval = env.unwrapped.evaluate()


    if get_success(initial_eval):

        return (
            "skip",
            0,
            "initial_success",
        )


    total_steps = 0


    # ========================================================
    # 构造这一次 episode 的目标抓取姿态
    # ========================================================

    (
        target_grasp_q,
        approach_direction,
    ) = build_grasp_orientation(
        obs
    )


    # ========================================================
    # PHASE 1
    #
    # Approach + Align
    # ========================================================

    cube_pos, _ = get_cube_pose(
        obs
    )


    pregrasp_pos = (
        cube_pos
        - approach_direction
        * PREGRASP_DISTANCE
    )


    (
        obs,
        info,
        phase_ok,
        task_success,
        used_steps,
    ) = move_to_pose(
        obs=obs,

        target_pos=pregrasp_pos,

        target_q=target_grasp_q,

        # 夹爪打开
        gripper=1.0,

        max_steps=MAX_APPROACH_STEPS,

        require_grasp=False,
    )


    total_steps += used_steps


    if task_success:

        return (
            "success",
            total_steps,
            None,
        )


    if not phase_ok:

        return (
            "failed",
            total_steps,
            "approach",
        )


    # ========================================================
    # PHASE 2
    #
    # Final Approach
    # ========================================================

    cube_pos, _ = get_cube_pose(
        obs
    )


    grasp_pos = (
        cube_pos.copy()
    )


    (
        obs,
        info,
        phase_ok,
        task_success,
        used_steps,
    ) = move_to_pose(
        obs=obs,

        target_pos=grasp_pos,

        target_q=target_grasp_q,

        gripper=1.0,

        max_steps=MAX_FINAL_APPROACH_STEPS,

        require_grasp=False,
    )


    total_steps += used_steps


    if task_success:

        return (
            "success",
            total_steps,
            None,
        )


    if not phase_ok:

        return (
            "failed",
            total_steps,
            "final_approach",
        )


    # ========================================================
    # PHASE 3
    #
    # Grasp
    # ========================================================

    (
        obs,
        info,
        phase_ok,
        task_success,
        used_steps,
    ) = grasp_cube(
        obs,
        target_grasp_q,
    )


    total_steps += used_steps


    if task_success:

        return (
            "success",
            total_steps,
            None,
        )


    if not phase_ok:

        return (
            "failed",
            total_steps,
            "grasp",
        )


    # ========================================================
    # PHASE 4
    #
    # Lift
    # ========================================================

    tcp_pos, _ = get_tcp_pose(
        obs
    )


    # approach_direction 指向 cube。
    #
    # 反方向就是远离 cube：
    # 对当前 PickCube 就是向上。
    lift_pos = (
        tcp_pos
        - approach_direction
        * LIFT_DISTANCE
    )


    (
        obs,
        info,
        phase_ok,
        task_success,
        used_steps,
    ) = move_to_pose(
        obs=obs,

        target_pos=lift_pos,

        target_q=target_grasp_q,

        gripper=-1.0,

        max_steps=MAX_LIFT_STEPS,

        require_grasp=True,
    )


    total_steps += used_steps


    if task_success:

        return (
            "success",
            total_steps,
            None,
        )


    if not phase_ok:

        return (
            "failed",
            total_steps,
            "lift",
        )


    # ========================================================
    # PHASE 5
    #
    # Direct Cube -> Goal
    #
    # 每一步使用：
    #
    # goal_pos - cube_pos
    # ========================================================

    (
        obs,
        info,
        phase_ok,
        task_success,
        used_steps,
    ) = move_cube_to_goal(
        obs,
        target_grasp_q,
    )


    total_steps += used_steps


    if task_success:

        return (
            "success",
            total_steps,
            None,
        )


    if not phase_ok:

        return (
            "failed",
            total_steps,
            "move_to_goal",
        )


    # ========================================================
    # PHASE 6
    #
    # Stabilize
    # ========================================================

    (
        obs,
        info,
        success,
        used_steps,
    ) = stabilize(
        obs,
        target_grasp_q,
    )


    total_steps += used_steps


    if success:

        return (
            "success",
            total_steps,
            None,
        )


    return (
        "failed",
        total_steps,
        "stabilize",
    )


# ============================================================
# 22. 批量生成 Expert Demos
# ============================================================

successful_demos = 0
attempted_episodes = 0
skipped_episodes = 0

successful_lengths = []

failure_counts = {}


print()
print("========================================")
print("Collecting PickCube Expert Demos")
print("========================================")

print(
    f"Target successful demos: "
    f"{NUM_SUCCESSFUL_DEMOS}"
)

print()


try:

    for seed in range(
        MAX_SEEDS_TO_TRY
    ):


        (
            status,
            steps,
            reason,
        ) = run_expert_episode(
            seed
        )


        attempted_episodes += 1


        # ====================================================
        # 当前 RecordEpisode buffer
        # 就是刚才这个 episode。
        #
        # 只有真正 success 才写进 HDF5。
        # ====================================================

        save_this_demo = (
            status == "success"
        )


        env.flush_trajectory(
            save=save_this_demo,
            verbose=False,
        )


        # ====================================================
        # SUCCESS
        # ====================================================

        if status == "success":

            successful_demos += 1

            successful_lengths.append(
                steps
            )


            print(
                f"[SUCCESS] "
                f"seed={seed:4d} | "
                f"steps={steps:3d} | "
                f"saved="
                f"{successful_demos}/"
                f"{NUM_SUCCESSFUL_DEMOS}"
            )


        # ====================================================
        # RESET 后已经 success
        #
        # 不是失败，只是没有 action 可供 BC 学。
        # ====================================================

        elif status == "skip":

            skipped_episodes += 1


            print(
                f"[SKIP   ] "
                f"seed={seed:4d} | "
                f"steps={steps:3d} | "
                f"reason={reason}"
            )


        # ====================================================
        # Expert failure
        # ====================================================

        else:

            failure_counts[
                reason
            ] = (
                failure_counts.get(
                    reason,
                    0,
                )
                + 1
            )


            print(
                f"[FAILED ] "
                f"seed={seed:4d} | "
                f"steps={steps:3d} | "
                f"stage={reason}"
            )


        # ====================================================
        # 已经获得足够成功 demos
        # ====================================================

        if (
            successful_demos
            >= NUM_SUCCESSFUL_DEMOS
        ):

            break


finally:

    # 正常关闭 HDF5 / JSON
    env.close()


# ============================================================
# 23. 最终统计
# ============================================================

print()
print("========================================")
print("Collection Finished")
print("========================================")


print(
    f"Attempted episodes : "
    f"{attempted_episodes}"
)

print(
    f"Successful demos   : "
    f"{successful_demos}"
)

print(
    f"Skipped episodes   : "
    f"{skipped_episodes}"
)


failed_episodes = (
    attempted_episodes
    - successful_demos
    - skipped_episodes
)


print(
    f"Failed episodes    : "
    f"{failed_episodes}"
)


# ------------------------------------------------------------
# 专家成功率：
#
# 这里不把 initial_success 的 skip
# 当成 expert failure。
# ------------------------------------------------------------

valid_attempts = (
    attempted_episodes
    - skipped_episodes
)


if valid_attempts > 0:

    success_rate = (
        successful_demos
        / valid_attempts
    )


    print(
        f"Expert success rate: "
        f"{success_rate * 100:.1f}%"
    )


# ------------------------------------------------------------
# 成功轨迹长度统计
# ------------------------------------------------------------

if successful_lengths:

    print(
        f"Mean demo length   : "
        f"{np.mean(successful_lengths):.1f} steps"
    )

    print(
        f"Shortest demo      : "
        f"{np.min(successful_lengths)} steps"
    )

    print(
        f"Longest demo       : "
        f"{np.max(successful_lengths)} steps"
    )


# ------------------------------------------------------------
# 各阶段失败次数
# ------------------------------------------------------------

if failure_counts:

    print()
    print("Failure stages:")


    for stage, count in (
        failure_counts.items()
    ):

        print(
            f"  {stage}: {count}"
        )


# ------------------------------------------------------------
# 输出文件
# ------------------------------------------------------------

print()
print("Dataset files:")

print(
    f"  {OUTPUT_DIR}/"
    f"{TRAJECTORY_NAME}.h5"
)

print(
    f"  {OUTPUT_DIR}/"
    f"{TRAJECTORY_NAME}.json"
)