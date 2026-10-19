import os
import copy
import random

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 0. 基本设置
# ============================================================

DATA_PATH = (
    "data/expert_demos_v2/"
    "pickcube_arc_scripted.h5"
)

CHECKPOINT_PATH = (
    "checkpoints/"
    "bc_state_mlp_v2.pt"
)


# ------------------------------------------------------------
# 为了和 V1 公平比较，
# 下面训练参数保持不变。
# ------------------------------------------------------------

SPLIT_SEED = 42

BATCH_SIZE = 256

LEARNING_RATE = 1e-3

NUM_EPOCHS = 100


# ============================================================
# 1. Observation field 顺序
#
# 一定要固定。
#
# 训练和之后 rollout 时必须完全使用同样的顺序。
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


# 对应维度：
#
# qpos              9
# qvel              9
# is_grasped        1
# tcp_pose          7
# goal_pos          3
# obj_pose          7
# tcp_to_obj_pos    3
# obj_to_goal_pos   3
#
# 总共：
#
# 9+9+1+7+3+7+3+3 = 42


# ============================================================
# 2. 固定随机种子
# ============================================================

random.seed(
    SPLIT_SEED
)

np.random.seed(
    SPLIT_SEED
)

torch.manual_seed(
    SPLIT_SEED
)


# ============================================================
# 3. 选择设备
#
# 你的 MacBook Air 是 Apple Silicon，
# 所以优先使用 MPS。
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
    "Train BC V2 - State MLP"
)

print(
    "========================================"
)

print(
    f"Device: {DEVICE}"
)

print(
    f"Data  : {DATA_PATH}"
)

print(
    f"Model : {CHECKPOINT_PATH}"
)

print()


# ============================================================
# 4. 检查数据文件
# ============================================================

if not os.path.exists(
    DATA_PATH
):

    raise FileNotFoundError(
        f"Dataset not found:\n"
        f"{DATA_PATH}"
    )


# 创建 checkpoint 文件夹
#
# exist_ok=True：
# 如果文件夹已经存在，不会报错。
#
# 这会修改文件系统：
# 如果 checkpoints/ 不存在，会创建它。
os.makedirs(
    os.path.dirname(
        CHECKPOINT_PATH
    ),
    exist_ok=True,
)


# ============================================================
# 5. trajectory 排序函数
#
# traj_2 应该排在 traj_10 前面。
# ============================================================

def trajectory_index(name):

    return int(
        name.split("_")[-1]
    )


# ============================================================
# 6. 从一条 trajectory 中构造 42D observation
# ============================================================

def load_observations(
    traj
):

    # --------------------------------------------------------
    # agent/qpos
    #
    # shape:
    #
    # (T+1, 9)
    # --------------------------------------------------------

    qpos = np.asarray(
        traj[
            "obs"
        ][
            "agent"
        ][
            "qpos"
        ],
        dtype=np.float32,
    )


    # --------------------------------------------------------
    # agent/qvel
    #
    # (T+1, 9)
    # --------------------------------------------------------

    qvel = np.asarray(
        traj[
            "obs"
        ][
            "agent"
        ][
            "qvel"
        ],
        dtype=np.float32,
    )


    # --------------------------------------------------------
    # is_grasped
    #
    # 原始 shape：
    #
    # (T+1,)
    #
    # 我们要变成：
    #
    # (T+1, 1)
    #
    # 才能和其他特征 concatenate。
    # --------------------------------------------------------

    is_grasped = np.asarray(
        traj[
            "obs"
        ][
            "extra"
        ][
            "is_grasped"
        ],
        dtype=np.float32,
    ).reshape(
        -1,
        1,
    )


    # --------------------------------------------------------
    # tcp_pose
    #
    # (T+1, 7)
    # --------------------------------------------------------

    tcp_pose = np.asarray(
        traj[
            "obs"
        ][
            "extra"
        ][
            "tcp_pose"
        ],
        dtype=np.float32,
    )


    # --------------------------------------------------------
    # goal_pos
    #
    # (T+1, 3)
    # --------------------------------------------------------

    goal_pos = np.asarray(
        traj[
            "obs"
        ][
            "extra"
        ][
            "goal_pos"
        ],
        dtype=np.float32,
    )


    # --------------------------------------------------------
    # obj_pose
    #
    # (T+1, 7)
    # --------------------------------------------------------

    obj_pose = np.asarray(
        traj[
            "obs"
        ][
            "extra"
        ][
            "obj_pose"
        ],
        dtype=np.float32,
    )


    # --------------------------------------------------------
    # tcp_to_obj_pos
    #
    # (T+1, 3)
    # --------------------------------------------------------

    tcp_to_obj_pos = np.asarray(
        traj[
            "obs"
        ][
            "extra"
        ][
            "tcp_to_obj_pos"
        ],
        dtype=np.float32,
    )


    # --------------------------------------------------------
    # obj_to_goal_pos
    #
    # (T+1, 3)
    # --------------------------------------------------------

    obj_to_goal_pos = np.asarray(
        traj[
            "obs"
        ][
            "extra"
        ][
            "obj_to_goal_pos"
        ],
        dtype=np.float32,
    )


    # ========================================================
    # 拼成：
    #
    # (T+1, 42)
    # ========================================================

    obs = np.concatenate(
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
        axis=1,
    )


    return obs.astype(
        np.float32
    )


# ============================================================
# 7. 读取指定 trajectories
# ============================================================

def load_trajectory_set(
    h5_file,
    trajectory_names,
):

    all_obs = []

    all_actions = []


    for traj_name in trajectory_names:


        traj = h5_file[
            traj_name
        ]


        # observation:
        #
        # T+1 个
        obs = load_observations(
            traj
        )


        # action:
        #
        # T 个
        actions = np.asarray(
            traj[
                "actions"
            ],
            dtype=np.float32,
        )


        # ====================================================
        # BC pair:
        #
        # obs_t -> action_t
        #
        # 所以最后一个 obs 没有对应 action，
        # 不用于训练。
        # ====================================================

        obs_for_actions = (
            obs[:-1]
        )


        if (
            obs_for_actions.shape[0]
            !=
            actions.shape[0]
        ):

            raise ValueError(
                f"{traj_name}: "
                f"obs/action length mismatch: "
                f"{obs_for_actions.shape[0]} "
                f"vs "
                f"{actions.shape[0]}"
            )


        all_obs.append(
            obs_for_actions
        )


        all_actions.append(
            actions
        )


    all_obs = np.concatenate(
        all_obs,
        axis=0,
    )


    all_actions = np.concatenate(
        all_actions,
        axis=0,
    )


    return (
        all_obs,
        all_actions,
    )


# ============================================================
# 8. 打开 HDF5 并按 trajectory 划分
# ============================================================

with h5py.File(
    DATA_PATH,
    "r",
) as f:


    trajectory_names = sorted(
        list(
            f.keys()
        ),
        key=trajectory_index,
    )


    num_trajectories = len(
        trajectory_names
    )


    print(
        f"Total trajectories: "
        f"{num_trajectories}"
    )


    if num_trajectories < 10:

        raise ValueError(
            "Too few trajectories."
        )


    # ========================================================
    # 按 trajectory shuffle。
    #
    # 注意：
    #
    # 不是把所有 timestep 打乱以后再 split。
    #
    # 否则同一条轨迹中非常相似的相邻状态
    # 可能同时出现在 train 和 test 中，
    # 会造成数据泄漏。
    # ========================================================

    shuffled_names = (
        trajectory_names.copy()
    )


    rng = random.Random(
        SPLIT_SEED
    )


    rng.shuffle(
        shuffled_names
    )


    # ========================================================
    # 80 / 10 / 10 trajectory split
    # ========================================================

    n_train = int(
        0.8
        * num_trajectories
    )


    n_val = int(
        0.1
        * num_trajectories
    )


    train_names = (
        shuffled_names[
            :n_train
        ]
    )


    val_names = (
        shuffled_names[
            n_train:
            n_train + n_val
        ]
    )


    test_names = (
        shuffled_names[
            n_train + n_val:
        ]
    )


    print()
    print(
        "Trajectory split:"
    )

    print(
        f"  train: "
        f"{len(train_names)}"
    )

    print(
        f"  val  : "
        f"{len(val_names)}"
    )

    print(
        f"  test : "
        f"{len(test_names)}"
    )


    # ========================================================
    # 读取数据
    # ========================================================

    train_obs, train_actions = (
        load_trajectory_set(
            f,
            train_names,
        )
    )


    val_obs, val_actions = (
        load_trajectory_set(
            f,
            val_names,
        )
    )


    test_obs, test_actions = (
        load_trajectory_set(
            f,
            test_names,
        )
    )


# HDF5 在这里已经关闭。


# ============================================================
# 9. 检查 observation/action 维度
# ============================================================

INPUT_DIM = (
    train_obs.shape[1]
)

ACTION_DIM = (
    train_actions.shape[1]
)


print()
print(
    "Sample statistics:"
)

print(
    f"  train samples: "
    f"{len(train_obs)}"
)

print(
    f"  val samples  : "
    f"{len(val_obs)}"
)

print(
    f"  test samples : "
    f"{len(test_obs)}"
)

print()

print(
    f"Observation dim: "
    f"{INPUT_DIM}"
)

print(
    f"Action dim     : "
    f"{ACTION_DIM}"
)


if INPUT_DIM != 42:

    raise ValueError(
        f"Expected 42D observation, "
        f"got {INPUT_DIM}"
    )


if ACTION_DIM != 7:

    raise ValueError(
        f"Expected 7D action, "
        f"got {ACTION_DIM}"
    )


# ============================================================
# 10. Observation normalization
#
# 只使用 TRAIN 数据计算 mean/std。
#
# ============================================================

obs_mean = train_obs.mean(
    axis=0
).astype(
    np.float32
)


obs_std = train_obs.std(
    axis=0
).astype(
    np.float32
)


# ------------------------------------------------------------
# 有些 feature 可能几乎不变化。
#
# 如果 std 非常接近 0，
# 除法会变得数值不稳定。
#
# 所以直接把这种 std 设置为 1。
# ------------------------------------------------------------

obs_std[
    obs_std < 1e-6
] = 1.0


# ============================================================
# 11. 使用 TRAIN 的 mean/std
#     统一标准化 train / val / test
# ============================================================

train_obs_norm = (
    train_obs
    - obs_mean
) / obs_std


val_obs_norm = (
    val_obs
    - obs_mean
) / obs_std


test_obs_norm = (
    test_obs
    - obs_mean
) / obs_std


# ============================================================
# 12. PyTorch Dataset
# ============================================================

class BCDataset(Dataset):

    def __init__(
        self,
        observations,
        actions,
    ):

        self.observations = (
            torch.tensor(
                observations,
                dtype=torch.float32,
            )
        )

        self.actions = (
            torch.tensor(
                actions,
                dtype=torch.float32,
            )
        )


    def __len__(self):

        return (
            self.observations.shape[0]
        )


    def __getitem__(
        self,
        index,
    ):

        return (
            self.observations[index],
            self.actions[index],
        )


# ============================================================
# 13. 创建三个 dataset
# ============================================================

train_dataset = BCDataset(
    train_obs_norm,
    train_actions,
)


val_dataset = BCDataset(
    val_obs_norm,
    val_actions,
)


test_dataset = BCDataset(
    test_obs_norm,
    test_actions,
)


# ============================================================
# 14. DataLoader
#
# train:
#     shuffle=True
#
# val/test:
#     不需要 shuffle
# ============================================================

train_loader = DataLoader(
    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,
)


val_loader = DataLoader(
    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,
)


test_loader = DataLoader(
    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,
)


# ============================================================
# 15. BC MLP
#
# 和 V1 完全一样：
#
# 42
# ↓
# 256
# ↓
# 256
# ↓
# 256
# ↓
# 7
#
# 最后 Tanh 保证：
#
# action ∈ [-1, 1]
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
# 16. 创建模型
# ============================================================

model = BCPolicy(
    INPUT_DIM,
    ACTION_DIM,
).to(
    DEVICE
)


print()
print(
    model
)

print()


# ============================================================
# 17. Loss + optimizer
# ============================================================

criterion = nn.MSELoss()


optimizer = torch.optim.Adam(
    model.parameters(),

    lr=LEARNING_RATE,
)


# ============================================================
# 18. Evaluation function
# ============================================================

def evaluate_loss(
    model,
    loader,
):


    model.eval()


    total_loss = 0.0

    total_samples = 0


    with torch.no_grad():


        for (
            obs_batch,
            action_batch,
        ) in loader:


            obs_batch = (
                obs_batch.to(
                    DEVICE
                )
            )


            action_batch = (
                action_batch.to(
                    DEVICE
                )
            )


            predicted_action = model(
                obs_batch
            )


            loss = criterion(
                predicted_action,
                action_batch,
            )


            batch_size = (
                obs_batch.shape[0]
            )


            total_loss += (
                loss.item()
                * batch_size
            )


            total_samples += (
                batch_size
            )


    return (
        total_loss
        / total_samples
    )


# ============================================================
# 19. Training
# ============================================================

best_val_loss = float(
    "inf"
)

best_epoch = -1

best_model_state = None


print(
    "========================================"
)

print(
    "Start Training"
)

print(
    "========================================"
)

print()


for epoch in range(
    1,
    NUM_EPOCHS + 1,
):


    # ========================================================
    # TRAIN
    # ========================================================

    model.train()


    total_train_loss = 0.0

    total_train_samples = 0


    for (
        obs_batch,
        action_batch,
    ) in train_loader:


        obs_batch = (
            obs_batch.to(
                DEVICE
            )
        )


        action_batch = (
            action_batch.to(
                DEVICE
            )
        )


        # ----------------------------------------------------
        # 清空上一 batch 的 gradient
        # ----------------------------------------------------

        optimizer.zero_grad()


        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        predicted_action = model(
            obs_batch
        )


        # ----------------------------------------------------
        # MSE
        # ----------------------------------------------------

        loss = criterion(
            predicted_action,
            action_batch,
        )


        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()


        # ----------------------------------------------------
        # 更新网络参数
        # ----------------------------------------------------

        optimizer.step()


        batch_size = (
            obs_batch.shape[0]
        )


        total_train_loss += (
            loss.item()
            * batch_size
        )


        total_train_samples += (
            batch_size
        )


    train_loss = (
        total_train_loss
        / total_train_samples
    )


    # ========================================================
    # VALIDATION
    #
    # 每个 epoch 都计算，
    # 即使不是每个 epoch 都打印。
    # ========================================================

    val_loss = evaluate_loss(
        model,
        val_loader,
    )


    # ========================================================
    # 保存当前 best validation model
    # ========================================================

    if val_loss < best_val_loss:


        best_val_loss = (
            val_loss
        )


        best_epoch = (
            epoch
        )


        # 必须 deepcopy。
        #
        # 否则后续训练会继续修改这些参数。
        best_model_state = copy.deepcopy(
            model.state_dict()
        )


    # ========================================================
    # 打印
    #
    # 第 1 epoch 打一次，
    # 后面每 5 epoch 打一次。
    # ========================================================

    if (
        epoch == 1
        or
        epoch % 5 == 0
    ):


        print(
            f"Epoch "
            f"{epoch:3d}/"
            f"{NUM_EPOCHS} | "
            f"train MSE = "
            f"{train_loss:.6f} | "
            f"val MSE = "
            f"{val_loss:.6f}"
        )


# ============================================================
# 20. 恢复 validation 最好的模型
# ============================================================

if best_model_state is None:

    raise RuntimeError(
        "No best model was saved."
    )


model.load_state_dict(
    best_model_state
)


print()
print(
    "========================================"
)

print(
    "Training Finished"
)

print(
    "========================================"
)


print(
    f"Best epoch   : "
    f"{best_epoch}"
)


print(
    f"Best val MSE : "
    f"{best_val_loss:.6f}"
)


# ============================================================
# 21. Test set 总 MSE
# ============================================================

test_loss = evaluate_loss(
    model,
    test_loader,
)


print(
    f"Test MSE     : "
    f"{test_loss:.6f}"
)


# ============================================================
# 22. Test set 每个 action dimension 的 MSE
# ============================================================

model.eval()


squared_error_sum = np.zeros(
    ACTION_DIM,
    dtype=np.float64,
)


num_test_samples = 0


with torch.no_grad():


    for (
        obs_batch,
        action_batch,
    ) in test_loader:


        obs_batch = (
            obs_batch.to(
                DEVICE
            )
        )


        action_batch = (
            action_batch.to(
                DEVICE
            )
        )


        predicted_action = model(
            obs_batch
        )


        squared_error = (
            predicted_action
            - action_batch
        ) ** 2


        squared_error_sum += (
            squared_error
            .sum(
                dim=0
            )
            .cpu()
            .numpy()
        )


        num_test_samples += (
            obs_batch.shape[0]
        )


per_dim_mse = (
    squared_error_sum
    / num_test_samples
)


action_names = [
    "dx",
    "dy",
    "dz",
    "dRx",
    "dRy",
    "dRz",
    "gripper",
]


print()
print(
    "Per-action test MSE:"
)


for (
    name,
    mse,
) in zip(
    action_names,
    per_dim_mse,
):


    print(
        f"  {name:<8}: "
        f"{mse:.6f}"
    )


# ============================================================
# 23. 保存 checkpoint
# ============================================================

checkpoint = {

    # 最好的模型参数
    "model_state_dict":
        model.state_dict(),

    # --------------------------------------------------------
    # rollout 时必须使用完全相同的 normalization
    # --------------------------------------------------------

    "obs_mean":
        obs_mean,

    "obs_std":
        obs_std,

    # 网络尺寸
    "input_dim":
        INPUT_DIM,

    "action_dim":
        ACTION_DIM,

    # observation 拼接顺序
    "obs_fields":
        OBS_FIELDS,

    # --------------------------------------------------------
    # 保存 trajectory split，
    # 方便以后知道哪些轨迹进入了 train/val/test
    # --------------------------------------------------------

    "train_trajectories":
        train_names,

    "val_trajectories":
        val_names,

    "test_trajectories":
        test_names,

    # best validation information
    "best_epoch":
        best_epoch,

    "best_val_loss":
        best_val_loss,

    "test_loss":
        test_loss,

    "per_dim_test_mse":
        per_dim_mse,

    # training config
    "split_seed":
        SPLIT_SEED,

    "batch_size":
        BATCH_SIZE,

    "learning_rate":
        LEARNING_RATE,

    "num_epochs":
        NUM_EPOCHS,

    # 数据来源
    "dataset_path":
        DATA_PATH,
}


torch.save(
    checkpoint,
    CHECKPOINT_PATH,
)


print()
print(
    "========================================"
)

print(
    "Checkpoint Saved"
)

print(
    "========================================"
)


print(
    CHECKPOINT_PATH
)

print()