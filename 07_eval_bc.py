import time

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import mani_skill.envs


# ============================================================
# 0. 设置
# ============================================================

CHECKPOINT_PATH = "checkpoints/bc_state_mlp.pt"

# 使用没有出现在 expert dataset 中的新 seed
EVAL_SEED = 1000

# 最多跑 300 个 control steps
MAX_EPISODE_STEPS = 300

# 为了让你肉眼看得清楚，
# 每个 control step 后暂停一点现实时间。
#
# 这不会改变模拟物理时间。
DISPLAY_DELAY = 0.04


# ============================================================
# 1. 和训练时完全相同的 MLP
# ============================================================

class BCPolicy(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                42,
                256,
            ),

            nn.ReLU(),

            nn.Linear(
                256,
                256,
            ),

            nn.ReLU(),

            nn.Linear(
                256,
                256,
            ),

            nn.ReLU(),

            nn.Linear(
                256,
                7,
            ),

            nn.Tanh(),
        )


    def forward(
        self,
        x,
    ):

        return self.network(
            x
        )


# ============================================================
# 2. 选择计算设备
# ============================================================

if torch.backends.mps.is_available():

    device = torch.device(
        "mps"
    )

else:

    device = torch.device(
        "cpu"
    )


print(
    f"Device: {device}"
)


# ============================================================
# 3. 读取训练好的 checkpoint
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
)


model = BCPolicy()


model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)


model = model.to(
    device
)


# Evaluation 模式
model.eval()


# ============================================================
# 4. 读取训练时保存的 normalization
# ============================================================

obs_mean = checkpoint[
    "obs_mean"
].cpu().numpy()


obs_std = checkpoint[
    "obs_std"
].cpu().numpy()


print(
    f"Observation dimension: "
    f"{len(obs_mean)}"
)


# ============================================================
# 5. torch / numpy 工具
# ============================================================

def to_numpy(x):

    if isinstance(
        x,
        torch.Tensor,
    ):

        return (
            x.detach()
            .cpu()
            .numpy()
        )

    return np.asarray(x)


def scalar_bool(x):

    x = to_numpy(x)

    return bool(
        x.reshape(-1)[0]
    )


# ============================================================
# 6. 把 ManiSkill state_dict
#    变成训练时完全相同的 42 维向量
# ============================================================

def observation_to_vector(obs):
    """
    顺序必须和 06_train_bc.py 完全一致：

        qpos               9
        qvel               9
        is_grasped         1
        tcp_pose           7
        goal_pos           3
        obj_pose           7
        tcp_to_obj_pos     3
        obj_to_goal_pos    3

    总共：

        42
    """


    qpos = to_numpy(
        obs["agent"]["qpos"]
    )[0]


    qvel = to_numpy(
        obs["agent"]["qvel"]
    )[0]


    is_grasped = np.array(
        [
            float(
                scalar_bool(
                    obs["extra"][
                        "is_grasped"
                    ]
                )
            )
        ],
        dtype=np.float32,
    )


    tcp_pose = to_numpy(
        obs["extra"]["tcp_pose"]
    )[0]


    goal_pos = to_numpy(
        obs["extra"]["goal_pos"]
    )[0]


    obj_pose = to_numpy(
        obs["extra"]["obj_pose"]
    )[0]


    tcp_to_obj_pos = to_numpy(
        obs["extra"][
            "tcp_to_obj_pos"
        ]
    )[0]


    obj_to_goal_pos = to_numpy(
        obs["extra"][
            "obj_to_goal_pos"
        ]
    )[0]


    vector = np.concatenate(
        [
            qpos,
            qvel,
            is_grasped,
            tcp_pose,
            goal_pos,
            obj_pose,
            tcp_to_obj_pos,
            obj_to_goal_pos,
        ]
    ).astype(
        np.float32
    )


    assert vector.shape == (
        42,
    )


    return vector


# ============================================================
# 7. Observation -> Neural Network -> Action
# ============================================================

def predict_action(obs):

    # --------------------------------------------------------
    # ManiSkill observation
    # ->
    # 42-dimensional vector
    # --------------------------------------------------------

    obs_vector = (
        observation_to_vector(
            obs
        )
    )


    # --------------------------------------------------------
    # 使用训练时完全相同的 mean/std
    # 做 normalization
    # --------------------------------------------------------

    normalized_obs = (
        obs_vector
        - obs_mean
    ) / obs_std


    # --------------------------------------------------------
    # NumPy
    # ->
    # PyTorch Tensor
    #
    # 原来：
    #
    # (42,)
    #
    # 加 batch dimension：
    #
    # (1, 42)
    # --------------------------------------------------------

    x = torch.tensor(
        normalized_obs,
        dtype=torch.float32,
    ).unsqueeze(
        0
    )


    x = x.to(
        device
    )


    # --------------------------------------------------------
    # Evaluation 不需要 gradient
    # --------------------------------------------------------

    with torch.no_grad():

        action = model(
            x
        )


    # --------------------------------------------------------
    # (1, 7)
    # ->
    # (7,)
    # --------------------------------------------------------

    action = (
        action[0]
        .cpu()
        .numpy()
    )


    # 理论上 Tanh 已经保证 [-1,1]，
    # 再 clip 一次作为安全检查。
    action = np.clip(
        action,
        -1.0,
        1.0,
    )


    return action.astype(
        np.float32
    )


# ============================================================
# 8. 创建真正用于 evaluation 的环境
# ============================================================

env = gym.make(
    "PickCube-v1",

    obs_mode="state_dict",

    control_mode="pd_ee_delta_pose",

    # 这次打开窗口
    render_mode="human",

    max_episode_steps=MAX_EPISODE_STEPS,
)


# ============================================================
# 9. Reset
# ============================================================

obs, info = env.reset(
    seed=EVAL_SEED
)


print()
print(
    "========================================"
)

print(
    "BC Policy Rollout"
)

print(
    "========================================"
)

print(
    f"Evaluation seed: "
    f"{EVAL_SEED}"
)

print()


# ============================================================
# 10. 神经网络正式接管机器人
# ============================================================

success = False


try:

    for step in range(
        MAX_EPISODE_STEPS
    ):


        # ====================================================
        # 最关键的一行：
        #
        # observation
        # ->
        # neural network
        # ->
        # action
        #
        # 从这里开始已经没有 scripted expert。
        # ====================================================

        action = predict_action(
            obs
        )


        # ====================================================
        # 神经网络输出的 action
        # 直接交给 ManiSkill
        # ====================================================

        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            action
        )


        # 更新窗口
        env.render()


        # 只是为了让你看清楚
        time.sleep(
            DISPLAY_DELAY
        )


        # ----------------------------------------------------
        # 每 10 step 打印一次简单状态
        # ----------------------------------------------------

        if (
            step % 10 == 0
        ):

            is_grasped = scalar_bool(
                obs["extra"][
                    "is_grasped"
                ]
            )


            cube_to_goal = to_numpy(
                obs["extra"][
                    "obj_to_goal_pos"
                ]
            )[0]


            distance = np.linalg.norm(
                cube_to_goal
            )


            print(
                f"step={step:3d} | "
                f"grasped={is_grasped} | "
                f"cube-goal distance="
                f"{distance:.4f} m"
            )


        # ====================================================
        # 官方 success
        # ====================================================

        if (
            "success" in info
            and
            scalar_bool(
                info["success"]
            )
        ):

            success = True


        # ====================================================
        # Episode 正式结束
        # ====================================================

        if (
            scalar_bool(
                terminated
            )
            or
            scalar_bool(
                truncated
            )
        ):

            break


finally:

    env.close()


# ============================================================
# 11. 最终结果
# ============================================================

print()
print(
    "========================================"
)

print(
    "Evaluation Finished"
)

print(
    "========================================"
)

print(
    f"Seed          : "
    f"{EVAL_SEED}"
)

print(
    f"Steps executed: "
    f"{step + 1}"
)

print(
    f"Success       : "
    f"{success}"
)


if success:

    print()
    print(
        "BC policy successfully completed PickCube."
    )

else:

    print()
    print(
        "BC policy did not complete PickCube on this seed."
    )