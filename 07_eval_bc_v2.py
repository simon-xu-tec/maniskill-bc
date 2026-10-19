import os
import time

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import mani_skill.envs


# ============================================================
# 0. Evaluation settings
# ============================================================

CHECKPOINT_PATH = (
    "checkpoints/"
    "bc_state_mlp_v2.pt"
)


# ------------------------------------------------------------
# 使用训练 demonstrations 之外的新 seed。
#
# 之前数据主要来自 seed 0, 1, 2, ...
#
# 所以这里选 1000，
# 测试模型面对一个新的初始状态。
# ------------------------------------------------------------

EVAL_SEED = 1910


MAX_EPISODE_STEPS = 300


# ------------------------------------------------------------
# 这里只影响现实中的播放速度。
#
# 不改变 ManiSkill 的 simulation physics，
# 也不改变 policy action。
#
# 0.10 比较适合肉眼观察。
# ------------------------------------------------------------

DISPLAY_DELAY = 0.10


# ============================================================
# 1. Observation field 顺序
#
# 必须和训练时完全相同。
# ============================================================

OBS_FIELDS = [
    "agent/qpos",
    "agent/qvel",
    "extra/is_grasped",
    "extra/tcp_pose",
    "extra/goal_pos",
    "extra/obj_pose",
    "extra/tcp_to_obj_pos",
    "extra/obj_to_goal_pos",
]


# ============================================================
# 2. 检查 checkpoint
# ============================================================

if not os.path.exists(
    CHECKPOINT_PATH
):

    raise FileNotFoundError(
        f"Checkpoint not found:\n"
        f"{CHECKPOINT_PATH}"
    )


# ============================================================
# 3. Device
#
# MacBook Apple Silicon：
# 优先使用 MPS。
# ============================================================

if torch.backends.mps.is_available():

    DEVICE = torch.device(
        "mps"
    )

elif torch.cuda.is_available():

    DEVICE = torch.device(
        "cuda"
    )

else:

    DEVICE = torch.device(
        "cpu"
    )


print()
print(
    "========================================"
)

print(
    "Evaluate BC V2 - State MLP"
)

print(
    "========================================"
)

print(
    f"Device     : {DEVICE}"
)

print(
    f"Checkpoint : {CHECKPOINT_PATH}"
)

print(
    f"Eval seed  : {EVAL_SEED}"
)

print()


# ============================================================
# 4. BC Policy
#
# 必须和训练时网络结构完全一致。
# ============================================================

class BCPolicy(nn.Module):

    def __init__(
        self,
        input_dim,
        action_dim,
    ):

        super().__init__()


        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
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
                action_dim,
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
# 5. Load checkpoint
#
# map_location="cpu"
#
# 先把 checkpoint 安全加载到 CPU，
# 然后再把 model 移到 MPS。
#
# weights_only=False 是因为我们的 checkpoint
# 不只有 weights，
# 里面还有 numpy mean/std、trajectory names 等信息。
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,

    map_location="cpu",

    weights_only=False,
)


INPUT_DIM = int(
    checkpoint[
        "input_dim"
    ]
)


ACTION_DIM = int(
    checkpoint[
        "action_dim"
    ]
)


obs_mean = np.asarray(
    checkpoint[
        "obs_mean"
    ],
    dtype=np.float32,
)


obs_std = np.asarray(
    checkpoint[
        "obs_std"
    ],
    dtype=np.float32,
)


checkpoint_obs_fields = (
    checkpoint.get(
        "obs_fields",
        None,
    )
)


# ============================================================
# 6. 检查 checkpoint 和当前程序是否一致
# ============================================================

if INPUT_DIM != 42:

    raise ValueError(
        f"Expected input_dim=42, "
        f"but checkpoint has "
        f"{INPUT_DIM}"
    )


if ACTION_DIM != 7:

    raise ValueError(
        f"Expected action_dim=7, "
        f"but checkpoint has "
        f"{ACTION_DIM}"
    )


if (
    checkpoint_obs_fields
    is not None
    and
    checkpoint_obs_fields
    != OBS_FIELDS
):

    raise ValueError(
        "Observation field order "
        "does not match training."
    )


if obs_mean.shape != (
    INPUT_DIM,
):

    raise ValueError(
        f"obs_mean shape is "
        f"{obs_mean.shape}, "
        f"expected ({INPUT_DIM},)"
    )


if obs_std.shape != (
    INPUT_DIM,
):

    raise ValueError(
        f"obs_std shape is "
        f"{obs_std.shape}, "
        f"expected ({INPUT_DIM},)"
    )


# ============================================================
# 7. 创建模型并载入训练好的参数
# ============================================================

model = BCPolicy(
    INPUT_DIM,
    ACTION_DIM,
)


model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)


model = model.to(
    DEVICE
)


# ------------------------------------------------------------
# evaluation mode
#
# 当前网络只有 Linear/ReLU/Tanh，
# eval() 和 train() 的数值行为其实一样。
#
# 但这是正规的推理写法。
# ------------------------------------------------------------

model.eval()


print(
    "Checkpoint information:"
)

print(
    f"  best epoch   : "
    f"{checkpoint.get('best_epoch')}"
)

print(
    f"  best val MSE : "
    f"{checkpoint.get('best_val_loss')}"
)

print(
    f"  test MSE     : "
    f"{checkpoint.get('test_loss')}"
)

print()


# ============================================================
# 8. 创建 PickCube environment
# ============================================================

env = gym.make(
    "PickCube-v1",

    obs_mode="state_dict",

    control_mode="pd_ee_delta_pose",

    # 打开窗口
    render_mode="human",

    max_episode_steps=MAX_EPISODE_STEPS,
)


# ============================================================
# 9. 基础工具
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


def get_is_grasped(obs):

    return scalar_bool(
        obs[
            "extra"
        ][
            "is_grasped"
        ]
    )


def get_tcp_pos(obs):

    tcp_pose = to_numpy(
        obs[
            "extra"
        ][
            "tcp_pose"
        ]
    )[0]

    return tcp_pose[
        :3
    ]


def get_cube_pos(obs):

    obj_pose = to_numpy(
        obs[
            "extra"
        ][
            "obj_pose"
        ]
    )[0]

    return obj_pose[
        :3
    ]


def get_goal_pos(obs):

    return to_numpy(
        obs[
            "extra"
        ][
            "goal_pos"
        ]
    )[0]


def get_success(info):

    if "success" not in info:

        return False

    return scalar_bool(
        info[
            "success"
        ]
    )


# ============================================================
# 10. 当前 state_dict -> 42D vector
#
# 这里必须和 06_train_bc_v2.py
# 的拼接顺序完全一致。
# ============================================================

def flatten_observation(obs):


    # --------------------------------------------------------
    # qpos
    # 9 dimensions
    # --------------------------------------------------------

    qpos = to_numpy(
        obs[
            "agent"
        ][
            "qpos"
        ]
    )[0].astype(
        np.float32
    )


    # --------------------------------------------------------
    # qvel
    # 9 dimensions
    # --------------------------------------------------------

    qvel = to_numpy(
        obs[
            "agent"
        ][
            "qvel"
        ]
    )[0].astype(
        np.float32
    )


    # --------------------------------------------------------
    # is_grasped
    # 1 dimension
    #
    # bool -> float32
    # False -> 0
    # True  -> 1
    # --------------------------------------------------------

    is_grasped = np.array(
        [
            float(
                get_is_grasped(
                    obs
                )
            )
        ],
        dtype=np.float32,
    )


    # --------------------------------------------------------
    # tcp_pose
    # 7 dimensions
    # --------------------------------------------------------

    tcp_pose = to_numpy(
        obs[
            "extra"
        ][
            "tcp_pose"
        ]
    )[0].astype(
        np.float32
    )


    # --------------------------------------------------------
    # goal_pos
    # 3 dimensions
    # --------------------------------------------------------

    goal_pos = to_numpy(
        obs[
            "extra"
        ][
            "goal_pos"
        ]
    )[0].astype(
        np.float32
    )


    # --------------------------------------------------------
    # obj_pose
    # 7 dimensions
    # --------------------------------------------------------

    obj_pose = to_numpy(
        obs[
            "extra"
        ][
            "obj_pose"
        ]
    )[0].astype(
        np.float32
    )


    # --------------------------------------------------------
    # tcp_to_obj_pos
    # 3 dimensions
    # --------------------------------------------------------

    tcp_to_obj_pos = to_numpy(
        obs[
            "extra"
        ][
            "tcp_to_obj_pos"
        ]
    )[0].astype(
        np.float32
    )


    # --------------------------------------------------------
    # obj_to_goal_pos
    # 3 dimensions
    # --------------------------------------------------------

    obj_to_goal_pos = to_numpy(
        obs[
            "extra"
        ][
            "obj_to_goal_pos"
        ]
    )[0].astype(
        np.float32
    )


    # ========================================================
    # 拼接：
    #
    # 9 + 9 + 1 + 7 + 3 + 7 + 3 + 3
    #
    # = 42
    # ========================================================

    observation_vector = np.concatenate(
        [
            qpos,
            qvel,
            is_grasped,
            tcp_pose,
            goal_pos,
            obj_pose,
            tcp_to_obj_pos,
            obj_to_goal_pos,
        ],
        axis=0,
    ).astype(
        np.float32
    )


    if observation_vector.shape != (
        INPUT_DIM,
    ):

        raise ValueError(
            f"Observation has shape "
            f"{observation_vector.shape}, "
            f"expected ({INPUT_DIM},)"
        )


    return observation_vector


# ============================================================
# 11. 当前 observation -> BC action
# ============================================================

def predict_action(obs):


    # ========================================================
    # A. state_dict -> 42D vector
    # ========================================================

    observation_vector = (
        flatten_observation(
            obs
        )
    )


    # ========================================================
    # B. 使用 TRAIN set 保存下来的 mean/std
    #
    # 注意：
    #
    # evaluation 时绝对不能重新计算 mean/std。
    #
    # 必须和训练时完全一样。
    # ========================================================

    normalized_obs = (
        observation_vector
        - obs_mean
    ) / obs_std


    # ========================================================
    # C. numpy -> torch
    #
    # 原来：
    #
    #     shape = (42,)
    #
    # unsqueeze(0) 后：
    #
    #     shape = (1, 42)
    #
    # 其中 1 是 batch dimension。
    # ========================================================

    obs_tensor = torch.tensor(
        normalized_obs,
        dtype=torch.float32,
    ).unsqueeze(
        0
    ).to(
        DEVICE
    )


    # ========================================================
    # D. 网络推理
    #
    # no_grad():
    #
    # evaluation 不需要计算 gradient。
    # ========================================================

    with torch.no_grad():


        action_tensor = model(
            obs_tensor
        )


    # ========================================================
    # E. torch -> numpy
    #
    # (1,7)
    # ->
    # (7,)
    # ========================================================

    action = (
        action_tensor[
            0
        ]
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )


    # Tanh 本身已经保证 [-1,1]，
    # 再 clip 一次只是保险。
    action = np.clip(
        action,
        -1.0,
        1.0,
    )


    return action


# ============================================================
# 12. Reset environment
# ============================================================

obs, info = env.reset(
    seed=EVAL_SEED
)


initial_cube_pos = (
    get_cube_pos(
        obs
    )
)


initial_goal_pos = (
    get_goal_pos(
        obs
    )
)


initial_goal_distance = (
    np.linalg.norm(
        initial_goal_pos
        - initial_cube_pos
    )
)


print(
    "Initial state:"
)

print(
    f"  cube position : "
    f"{initial_cube_pos}"
)

print(
    f"  goal position : "
    f"{initial_goal_pos}"
)

print(
    f"  goal distance : "
    f"{initial_goal_distance:.4f} m"
)

print(
    f"  grasped       : "
    f"{get_is_grasped(obs)}"
)

print()


# ============================================================
# 13. Closed-loop rollout
#
# 非常重要：
#
# 这里完全没有 scripted expert。
#
#
# 每一步都是：
#
# obs_t
#   ↓
# MLP
#   ↓
# action_t
#   ↓
# env.step(action_t)
#   ↓
# obs_{t+1}
#
# 然后重复。
# ============================================================

success = False

ever_grasped = False

lost_grasp = False

previously_grasped = False


try:


    for step in range(
        MAX_EPISODE_STEPS
    ):


        # ====================================================
        # 1. 当前 observation 输入 BC policy
        # ====================================================

        action = predict_action(
            obs
        )


        # ====================================================
        # 2. 执行动作
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


        # ----------------------------------------------------
        # 只是慢放，
        # 不改变 simulation。
        # ----------------------------------------------------

        time.sleep(
            DISPLAY_DELAY
        )


        # ====================================================
        # 3. 读取执行后的实际状态
        # ====================================================

        now_grasped = (
            get_is_grasped(
                obs
            )
        )


        cube_pos = (
            get_cube_pos(
                obs
            )
        )


        tcp_pos = (
            get_tcp_pos(
                obs
            )
        )


        goal_pos = (
            get_goal_pos(
                obs
            )
        )


        tcp_cube_distance = np.linalg.norm(
            tcp_pos
            - cube_pos
        )


        goal_distance = np.linalg.norm(
            goal_pos
            - cube_pos
        )


        xy_goal_distance = np.linalg.norm(
            goal_pos[:2]
            - cube_pos[:2]
        )


        # ====================================================
        # 4. 这些变量只是 evaluation statistics
        #
        # 不会输入 model。
        # ====================================================

        if now_grasped:

            ever_grasped = True


        if (
            previously_grasped
            and
            not now_grasped
        ):

            lost_grasp = True


        previously_grasped = (
            now_grasped
        )


        # ====================================================
        # 5. Official success
        # ====================================================

        if get_success(
            info
        ):

            success = True


        # ====================================================
        # 6. 每 5 steps 打印一次
        #
        # 尤其注意 action[6]：
        #
        # +1 -> open
        # -1 -> close
        #
        # 现在网络可能预测中间值，
        # 例如：
        #
        # +0.7
        # +0.2
        # -0.3
        # -0.9
        #
        # 这是我们这次特别需要观察的。
        # ====================================================

        if step % 5 == 0:


            print(
                f"step={step:3d} | "

                f"grasped="
                f"{now_grasped} | "

                f"cube_z="
                f"{cube_pos[2]:.4f} | "

                f"tcp_cube="
                f"{tcp_cube_distance:.4f} | "

                f"goal_dist="
                f"{goal_distance:.4f} | "

                f"xy_goal="
                f"{xy_goal_distance:.4f}"
            )


            print(
                "    action = "
                f"["
                f"{action[0]: .3f}, "
                f"{action[1]: .3f}, "
                f"{action[2]: .3f}, "
                f"{action[3]: .3f}, "
                f"{action[4]: .3f}, "
                f"{action[5]: .3f}, "
                f"{action[6]: .3f}"
                f"]"
            )


        # ====================================================
        # 7. Environment 已结束
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
# 14. 最终状态
# ============================================================

final_cube_pos = (
    get_cube_pos(
        obs
    )
)


final_goal_pos = (
    get_goal_pos(
        obs
    )
)


final_goal_distance = np.linalg.norm(
    final_goal_pos
    - final_cube_pos
)


final_xy_goal_distance = (
    np.linalg.norm(
        final_goal_pos[:2]
        - final_cube_pos[:2]
    )
)


final_grasped = (
    get_is_grasped(
        obs
    )
)


print()
print(
    "========================================"
)

print(
    "BC V2 Rollout Finished"
)

print(
    "========================================"
)


print(
    f"Steps executed      : "
    f"{step + 1}"
)


print(
    f"Ever grasped        : "
    f"{ever_grasped}"
)


print(
    f"Lost grasp          : "
    f"{lost_grasp}"
)


print(
    f"Final grasped       : "
    f"{final_grasped}"
)


print(
    f"Final cube position : "
    f"{final_cube_pos}"
)


print(
    f"Goal position       : "
    f"{final_goal_pos}"
)


print(
    f"Final goal distance : "
    f"{final_goal_distance:.4f} m"
)


print(
    f"Final XY distance   : "
    f"{final_xy_goal_distance:.4f} m"
)


print(
    f"Official success    : "
    f"{success}"
)