import time
import numpy as np
import torch
import gymnasium as gym
import mani_skill.envs

from mani_skill.utils.geometry.rotation_conversions import (
    quaternion_to_matrix,
    matrix_to_quaternion,
    matrix_to_euler_angles,
)


# ============================================================
# 0. 实验参数
# ============================================================

# 改这个数字就可以测试不同初始场景
SEED = 10


# ------------------------------------------------------------
# 显示速度
# ------------------------------------------------------------

# 每执行一个真正的 control step 后，
# 在现实世界停 0.20 秒方便我们观察。
#
# 注意：
# visual_pause 不会推进模拟时间。
STEP_VIEW_DELAY = 0.20

# 每个大阶段结束以后暂停一下
PHASE_VIEW_DELAY = 1.5


# ------------------------------------------------------------
# 轨迹参数
# ------------------------------------------------------------

# 最后精细抓取开始前，
# TCP 离方块中心的距离。
#
# 7 cm 只是合理的工程取值，
# 不是什么特殊常数。
PREGRASP_DISTANCE = 0.07

# 抓住以后向上抬 8 cm，
# 主要目的是离开桌面，避免横向搬运时碰桌子。
LIFT_DISTANCE = 0.08


# ------------------------------------------------------------
# pd_ee_delta_pose 控制参数
# ------------------------------------------------------------

# Panda 当前配置：
#
# normalized translation action ±1
# 对应约 ±0.1 m
POSITION_SCALE = 0.1

# 每个 control step 最多请求移动 1 cm
MAX_POSITION_STEP = 0.01


# 当前 ManiSkill main 中 rotation action
# 仍然乘 rot_lower=-0.1。
#
# 所以这里用 -0.1 对当前实现做匹配。
ROTATION_SCALE = -0.1

# 每个 control step 最多旋转：
#
# 0.04 rad ≈ 2.29°
MAX_ROTATION_STEP = 0.04


# ============================================================
# 1. 创建环境
# ============================================================

env = gym.make(
    "PickCube-v1",

    # 我们需要有名字的 observation，
    # 方便读取 tcp_pose、obj_pose、goal_pos 等。
    obs_mode="state_dict",

    # 7维 action：
    #
    # [dx, dy, dz,
    #  dRx, dRy, dRz,
    #  gripper]
    control_mode="pd_ee_delta_pose",

    # 打开 GUI
    render_mode="human",

    # PickCube 默认只有 50 steps。
    #
    # 我们的教学专家动作比较慢，
    # 所以先给 300。
    max_episode_steps=300,
)


# ============================================================
# 2. 基础辅助函数
# ============================================================

def to_numpy(x):
    """
    如果是 torch.Tensor，就转换成 numpy。

    不修改原 tensor。
    """

    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()

    return np.asarray(x)


def scalar_bool(x):
    """
    把：

        tensor([True])

    或 numpy array

    转成普通 Python bool。
    """

    x = to_numpy(x)

    return bool(
        x.reshape(-1)[0]
    )


def visual_pause(seconds):
    """
    只让现实时间暂停，方便看动画。

    这里没有 env.step()，
    因此模拟时间不会前进。
    """

    start = time.time()

    while time.time() - start < seconds:

        env.render()

        time.sleep(0.02)


# ============================================================
# 3. 从 observation 中读取状态
# ============================================================

def get_tcp_pose(obs):
    """
    tcp_pose:

        [x, y, z, qw, qx, qy, qz]

    返回：

        position
        quaternion
    """

    pose = to_numpy(
        obs["extra"]["tcp_pose"]
    )[0]

    position = pose[:3]

    quaternion = pose[3:]

    return position, quaternion


def get_cube_pose(obs):
    """
    方块的绝对世界位姿：

        [x, y, z, qw, qx, qy, qz]
    """

    pose = to_numpy(
        obs["extra"]["obj_pose"]
    )[0]

    position = pose[:3]

    quaternion = pose[3:]

    return position, quaternion


def get_goal_pos(obs):
    """
    绿色目标球中心的世界坐标。
    """

    return to_numpy(
        obs["extra"]["goal_pos"]
    )[0]


def get_finger_qpos(obs):
    """
    Panda 最后两个 qpos
    对应左右两个夹爪手指。
    """

    qpos = to_numpy(
        obs["agent"]["qpos"]
    )[0]

    return qpos[-2:]


def get_is_grasped(obs):
    """
    ManiSkill 是否判断方块已经被夹住。
    """

    return scalar_bool(
        obs["extra"]["is_grasped"]
    )


def get_success(info):
    """
    PickCube 当前是否成功。
    """

    if "success" not in info:
        return False

    return scalar_bool(
        info["success"]
    )


# ============================================================
# 4. 四元数 <-> 旋转矩阵
# ============================================================

def quat_to_matrix(q):
    """
    ManiSkill quaternion 顺序：

        [qw, qx, qy, qz]

    转成 3×3 rotation matrix。
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
    3×3 rotation matrix
        ↓
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
# 5. 旋转矩阵对应的最小旋转角度
# ============================================================

def rotation_angle(R):
    """
    对相对旋转矩阵：

        R

    计算总旋转角：

        theta =
        arccos((Tr(R)-1)/2)

    返回单位 rad。
    """

    value = (
        np.trace(R) - 1.0
    ) / 2.0

    value = np.clip(
        value,
        -1.0,
        1.0,
    )

    return np.arccos(
        value
    )


# ============================================================
# 6. 根据方块姿态构造 TCP 的目标抓取姿态
# ============================================================

def build_grasp_orientation(obs):
    """
    我们自己构造目标 TCP 姿态。


    规定：

        z_TCP,target = -z_cube


    也就是：

        TCP 的接近方向
        指向方块表面。


    Panda TCP 的 local y
    是夹爪闭合方向。


    正方形有四种等价抓法：

        +x_cube
        -x_cube
        +y_cube
        -y_cube


    我们分别构造四个完整目标姿态，
    然后选择：

        当前 TCP -> 目标 TCP

    所需总旋转角最小的一个。
    """


    # --------------------------------------------------------
    # 当前 TCP 绝对姿态
    # --------------------------------------------------------

    _, tcp_q = get_tcp_pose(
        obs
    )

    R_tcp = quat_to_matrix(
        tcp_q
    )


    # --------------------------------------------------------
    # 方块绝对姿态
    # --------------------------------------------------------

    _, cube_q = get_cube_pose(
        obs
    )

    R_cube = quat_to_matrix(
        cube_q
    )


    # --------------------------------------------------------
    # R_cube 的三列：
    #
    # 方块 local x/y/z
    # 在世界坐标系中的方向。
    # --------------------------------------------------------

    cube_x = R_cube[:, 0]

    cube_y = R_cube[:, 1]

    cube_z = R_cube[:, 2]


    # ========================================================
    # TCP 的目标 z 轴
    # ========================================================

    target_z = -cube_z

    target_z = (
        target_z
        / np.linalg.norm(target_z)
    )


    # ========================================================
    # TCP 的目标 y 轴候选
    # ========================================================

    closing_candidates = [
        cube_x,
        -cube_x,
        cube_y,
        -cube_y,
    ]


    best_R = None
    best_angle = None


    # ========================================================
    # 检查四个等价抓取姿态
    # ========================================================

    for candidate_y in closing_candidates:

        target_y = (
            candidate_y
            / np.linalg.norm(candidate_y)
        )


        # ----------------------------------------------------
        # 已知 TCP 的目标 y 和 z，
        # 由右手系确定 x：
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


        # 再算一次 y，
        # 去掉可能的数值误差，
        # 保证严格正交。
        target_y = np.cross(
            target_z,
            target_x,
        )

        target_y = (
            target_y
            / np.linalg.norm(target_y)
        )


        # ----------------------------------------------------
        # 旋转矩阵的三列就是：
        #
        # TCP local x/y/z
        # 在世界坐标中的方向。
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
        #
        # root/world aligned：
        #
        # R_target =
        # R_error @ R_tcp
        #
        # 因此：
        #
        # R_error =
        # R_target @ R_tcp.T
        # ----------------------------------------------------

        R_error = (
            R_target
            @ R_tcp.T
        )


        angle = rotation_angle(
            R_error
        )


        if (
            best_angle is None
            or angle < best_angle
        ):

            best_angle = angle

            best_R = R_target


    # 完整目标姿态转成四元数
    target_q = matrix_to_quat(
        best_R
    )


    return (
        target_q,
        target_z,
        best_angle,
    )


# ============================================================
# 7. 限制一个向量的长度
# ============================================================

def limit_norm(vector, max_norm):
    """
    如果向量已经够小：
        原样返回。

    如果太大：
        保持方向，
        只把长度缩到 max_norm。
    """

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
# 8. 计算当前 control step 的三维旋转增量
# ============================================================

def calculate_rotation_step(
    current_q,
    target_q,
):
    """
    已知：

        当前 TCP 绝对姿态 current_q

        目标 TCP 绝对姿态 target_q


    先计算完整相对旋转：

        R_error =
        R_target @ R_current.T


    但是我们不会一口气全部旋转完。

    每一步最多旋转：

        MAX_ROTATION_STEP
    """


    R_current = quat_to_matrix(
        current_q
    )

    R_target = quat_to_matrix(
        target_q
    )


    # 完整姿态误差
    R_error = (
        R_target
        @ R_current.T
    )


    total_angle = rotation_angle(
        R_error
    )


    # 已经基本没有姿态误差
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


    # q 和 -q 表示同一个旋转。
    #
    # 令 qw >= 0，
    # 选择较短的旋转路径。
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
    # 从 quaternion 得到 rotation axis
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
    # 本 step 只执行一个小角度
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
    # ManiSkill pd_ee_delta_pose
    # rotation action 最终使用 XYZ Euler delta。
    #
    # 所以：
    #
    # q_step
    # ↓
    # R_step
    # ↓
    # XYZ Euler
    # ========================================================

    R_step = quat_to_matrix(
        q_step
    )


    R_tensor = torch.tensor(
        R_step,
        dtype=torch.float32,
    ).reshape(1, 3, 3)


    xyz_step = matrix_to_euler_angles(
        R_tensor,
        "XYZ",
    )[0]


    return (
        xyz_step.cpu().numpy(),
        total_angle,
    )


# ============================================================
# 9. 根据目标 TCP pose 生成 action
# ============================================================

def make_pose_action(
    obs,
    target_pos,
    target_q,
    gripper,
):
    """
    输入：

        当前 observation

        目标 TCP position

        目标 TCP quaternion

        gripper command


    输出：

        7维 normalized action
    """


    current_pos, current_q = (
        get_tcp_pose(obs)
    )


    # ========================================================
    # A. 位置
    # ========================================================

    position_error = (
        target_pos
        - current_pos
    )


    position_step = limit_norm(
        position_error,
        MAX_POSITION_STEP,
    )


    # 实际小位移
    # ->
    # normalized action
    position_action = (
        position_step
        / POSITION_SCALE
    )


    # ========================================================
    # B. 姿态
    # ========================================================

    (
        rotation_step,
        rotation_error,
    ) = calculate_rotation_step(
        current_q,
        target_q,
    )


    # 实际 XYZ rotation delta
    # ->
    # normalized action
    rotation_action = (
        rotation_step
        / ROTATION_SCALE
    )


    # ========================================================
    # C. 7维 action
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


    # 最后保险：
    #
    # action 必须在 [-1, 1]
    action = np.clip(
        action,
        -1.0,
        1.0,
    )


    return (
        action,
        position_error,
        rotation_error,
    )


# ============================================================
# 10. 通用闭环移动函数
# ============================================================

def move_to_pose(
    obs,
    target_pos,
    target_q,
    gripper,
    title,
    max_steps,
    pos_tolerance=0.004,
    rot_tolerance=np.deg2rad(2.0),
):
    """
    每一步都：

        当前 obs
            ↓
        重新计算误差
            ↓
        重新生成 action
            ↓
        env.step(action)
            ↓
        新 obs

    所以这是闭环控制。
    """


    print("\n")
    print("########################################")
    print(title)
    print("########################################")


    last_info = {}


    for i in range(max_steps):


        (
            action,
            pos_error,
            rot_error,
        ) = make_pose_action(
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
            last_info,
        ) = env.step(
            action
        )


        env.render()


        pos_distance = np.linalg.norm(
            pos_error
        )


        print(
            f"step {i:02d} | "
            f"pos error = "
            f"{pos_distance * 100:.2f} cm | "
            f"rot error = "
            f"{np.degrees(rot_error):.2f} deg"
        )


        visual_pause(
            STEP_VIEW_DELAY
        )


        # ----------------------------------------------------
        # 到达目标 pose
        # ----------------------------------------------------

        if (
            pos_distance < pos_tolerance
            and
            rot_error < rot_tolerance
        ):

            print(
                ">>> 到达目标位姿"
            )

            return (
                obs,
                last_info,
                False,
            )


        # ----------------------------------------------------
        # 整个 episode 超时
        # ----------------------------------------------------

        if scalar_bool(truncated):

            print(
                ">>> Episode timeout"
            )

            return (
                obs,
                last_info,
                True,
            )


    print(
        ">>> 当前阶段达到自己的最大 step"
    )


    return (
        obs,
        last_info,
        False,
    )


# ============================================================
# 11. 打印当前状态
# ============================================================

def print_state(
    title,
    obs,
):

    tcp_pos, tcp_q = (
        get_tcp_pose(obs)
    )

    cube_pos, cube_q = (
        get_cube_pose(obs)
    )


    print("\n========================================")
    print(title)


    print("\nTCP position:")
    print(tcp_pos)


    print("\nTCP quaternion:")
    print(tcp_q)


    print("\nCube position:")
    print(cube_pos)


    print("\nCube quaternion:")
    print(cube_q)


    print("\nGoal position:")
    print(
        get_goal_pos(obs)
    )


    print("\nFinger qpos:")
    print(
        get_finger_qpos(obs)
    )


    print(
        "\nis_grasped:",
        get_is_grasped(obs),
    )


# ============================================================
# 12. 开始唯一的一个 episode
# ============================================================

obs, info = env.reset(
    seed=SEED
)


env.render()


print_state(
    "INITIAL STATE",
    obs,
)


print(
    "\n观察初始场景 3 秒..."
)

visual_pause(3.0)


# ============================================================
# 13. 计算目标抓取姿态
# ============================================================

(
    target_grasp_q,
    approach_direction,
    rotation_needed,
) = build_grasp_orientation(
    obs
)


print("\n========================================")
print("TARGET GRASP")


print("\nTarget TCP quaternion:")
print(
    target_grasp_q
)


print("\nApproach direction:")
print(
    approach_direction
)


print(
    "\nCurrent -> target rotation:",
    f"{np.degrees(rotation_needed):.2f} deg",
)


# ============================================================
# PHASE 1
#
# 接近方块
# +
# 同时调整完整三维姿态
# ============================================================

cube_pos, _ = get_cube_pose(
    obs
)


# approach_direction 是：
#
# TCP -> cube
#
# 所以预抓取点位于反方向：
#
# cube - approach * d_pre
pregrasp_pos = (
    cube_pos
    - approach_direction
    * PREGRASP_DISTANCE
)


obs, info, aborted = move_to_pose(
    obs=obs,

    target_pos=pregrasp_pos,

    target_q=target_grasp_q,

    # 接近时保持夹爪打开
    gripper=1.0,

    title=(
        "PHASE 1: APPROACH + ALIGN"
    ),

    max_steps=80,
)


if aborted:

    print(
        "\nEpisode 因 timeout 结束。"
    )

    visual_pause(3.0)

    env.close()

    raise SystemExit


print_state(
    "AT PRE-GRASP",
    obs,
)


visual_pause(
    PHASE_VIEW_DELAY
)


# ============================================================
# PHASE 2
#
# 最后精细下降
# ============================================================

cube_pos, _ = get_cube_pose(
    obs
)


# TCP 最终来到方块中心附近
grasp_pos = cube_pos.copy()


obs, info, aborted = move_to_pose(
    obs=obs,

    target_pos=grasp_pos,

    target_q=target_grasp_q,

    gripper=1.0,

    title=(
        "PHASE 2: FINAL APPROACH"
    ),

    max_steps=40,
)


if aborted:

    env.close()

    raise SystemExit


print_state(
    "READY TO GRASP",
    obs,
)


print(
    "\n先观察夹爪和方块的位置 2 秒..."
)

visual_pause(2.0)


# ============================================================
# PHASE 3
#
# 闭合夹爪
# ============================================================

print("\n")
print("########################################")
print("PHASE 3: GRASP")
print("########################################")


# 闭合开始后，
# 希望 TCP 基本保持当前位置。
hold_pos, _ = get_tcp_pose(
    obs
)


grasp_confirm_count = 0


for i in range(30):


    (
        action,
        pos_error,
        rot_error,
    ) = make_pose_action(
        obs,
        hold_pos,
        target_grasp_q,

        # -1 = 闭合
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


    env.render()


    grasped = get_is_grasped(
        obs
    )


    print(
        f"step {i:02d} | "
        f"finger = "
        f"{get_finger_qpos(obs)} | "
        f"grasped = {grasped}"
    )


    visual_pause(
        0.35
    )


    if scalar_bool(truncated):

        print(
            ">>> Episode timeout"
        )

        env.close()

        raise SystemExit


    # 连续三步都检测到抓住，
    # 才认为抓取稳定。
    if grasped:

        grasp_confirm_count += 1

    else:

        grasp_confirm_count = 0


    if grasp_confirm_count >= 3:

        print(
            ">>> 方块已经稳定抓住"
        )

        break


# ============================================================
# 抓取失败直接结束
# ============================================================

if not get_is_grasped(obs):

    print(
        "\n>>> GRASP FAILED"
    )

    print(
        "本 episode 到这里结束。"
    )

    visual_pause(4.0)

    env.close()

    raise SystemExit


print_state(
    "AFTER GRASP",
    obs,
)


visual_pause(
    PHASE_VIEW_DELAY
)


# ============================================================
# PHASE 4
#
# 抬起方块
# ============================================================

tcp_pos, _ = get_tcp_pose(
    obs
)


# approach_direction 指向方块，
# 因此反方向就是远离方块/向上。
lift_pos = (
    tcp_pos
    - approach_direction
    * LIFT_DISTANCE
)


obs, info, aborted = move_to_pose(
    obs=obs,

    target_pos=lift_pos,

    target_q=target_grasp_q,

    gripper=-1.0,

    title=(
        "PHASE 4: LIFT"
    ),

    max_steps=50,
)


if aborted:

    env.close()

    raise SystemExit


print_state(
    "AFTER LIFT",
    obs,
)


visual_pause(
    PHASE_VIEW_DELAY
)


# ============================================================
# 14. 测量真实的 Cube-TCP 偏移
# ============================================================

tcp_pos, _ = get_tcp_pose(
    obs
)

cube_pos, _ = get_cube_pose(
    obs
)


# ------------------------------------------------------------
# 理论上如果完全居中：
#
# cube_pos ≈ tcp_pos
#
# 但真实抓取可能偏几毫米。
#
# 所以我们记录实际 offset。
# ------------------------------------------------------------

cube_tcp_offset = (
    cube_pos
    - tcp_pos
)


print("\n========================================")
print("ACTUAL GRASP OFFSET")

print("\nCube - TCP:")
print(
    cube_tcp_offset
)


# ============================================================
# PHASE 5
#
# 直接移动方块到绿色目标
#
# 注意：
#
# 没有：
#
#     先去 goal 上方
#         ↓
#     再下降
#
# 就是直接朝目标走。
# ============================================================

goal_pos = get_goal_pos(
    obs
)


# 已知：
#
# cube_pos
# =
# tcp_pos + cube_tcp_offset
#
#
# 希望：
#
# cube_pos = goal_pos
#
#
# 因此：
#
# tcp_target
# =
# goal_pos - cube_tcp_offset
target_tcp_at_goal = (
    goal_pos
    - cube_tcp_offset
)


obs, info, aborted = move_to_pose(
    obs=obs,

    target_pos=target_tcp_at_goal,

    target_q=target_grasp_q,

    # 搬运过程中一直夹住
    gripper=-1.0,

    title=(
        "PHASE 5: DIRECT MOVE TO GOAL"
    ),

    max_steps=100,

    pos_tolerance=0.003,
)


if aborted:

    env.close()

    raise SystemExit


print_state(
    "AT GOAL",
    obs,
)


visual_pause(
    PHASE_VIEW_DELAY
)


# ============================================================
# PHASE 6
#
# 保持当前位置，让机器人静下来
#
# PickCube success 不要求松手。
#
# 它要求：
#
# cube 足够接近 goal
# +
# robot static
# ============================================================

print("\n")
print("########################################")
print("PHASE 6: STABILIZE")
print("########################################")


for i in range(40):


    # --------------------------------------------------------
    # 继续控制 TCP 保持在 goal 对应位置，
    # 同时继续夹紧。
    # --------------------------------------------------------

    (
        action,
        pos_error,
        rot_error,
    ) = make_pose_action(
        obs,
        target_tcp_at_goal,
        target_grasp_q,
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


    env.render()


    cube_pos, _ = get_cube_pose(
        obs
    )

    goal_pos = get_goal_pos(
        obs
    )


    cube_goal_distance = np.linalg.norm(
        cube_pos
        - goal_pos
    )


    success = get_success(
        info
    )


    print(
        f"step {i:02d} | "
        f"cube-goal = "
        f"{cube_goal_distance * 100:.2f} cm | "
        f"success = {success}"
    )


    visual_pause(
        STEP_VIEW_DELAY
    )


    if success:

        print("\n")
        print("========================================")
        print(">>> PICKCUBE SUCCESS")
        print("========================================")

        break


    if scalar_bool(truncated):

        print(
            "\n>>> Episode timeout"
        )

        break


# ============================================================
# 15. 最终结果
# ============================================================

print_state(
    "FINAL STATE",
    obs,
)


print("\n========================================")

if get_success(info):

    print(
        "FINAL RESULT: SUCCESS"
    )

else:

    print(
        "FINAL RESULT: NOT SUCCESS"
    )


print(
    "\n最后观察 6 秒..."
)

visual_pause(6.0)


# ============================================================
# 16. 关闭环境
# ============================================================

env.close()