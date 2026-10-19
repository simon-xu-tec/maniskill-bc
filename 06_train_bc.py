from pathlib import Path
import copy

import h5py
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import (
    TensorDataset,
    DataLoader,
)


# ============================================================
# 0. 配置
# ============================================================

DATA_PATH = Path(
    "data/expert_demos/pickcube_scripted.h5"
)

CHECKPOINT_DIR = Path(
    "checkpoints"
)

CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / "bc_state_mlp.pt"
)


# 随机种子：
# 保证 trajectory split 可以重复
RANDOM_SEED = 42


# 100 条轨迹：
#
# 80 train
# 10 validation
# 10 test
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1


# 每次拿多少个 (obs, action)
# 样本更新一次网络
BATCH_SIZE = 256


# 完整训练数据重复学习多少轮
NUM_EPOCHS = 100


# Adam learning rate
LEARNING_RATE = 1e-3


# ============================================================
# 1. 固定随机性
# ============================================================

np.random.seed(
    RANDOM_SEED
)

torch.manual_seed(
    RANDOM_SEED
)


# ============================================================
# 2. trajectory 名字排序
# ============================================================

def trajectory_number(name):

    return int(
        name.split("_")[-1]
    )


# ============================================================
# 3. 一条 trajectory -> BC 输入和标签
# ============================================================

def load_trajectory(traj):
    """
    对一条 T-step trajectory：

        observation 有 T+1 个
        action 有 T 个


    BC 使用：

        o_0 -> a_0
        o_1 -> a_1
        ...
        o_(T-1) -> a_(T-1)

    所以所有 observation 都使用：

        [:-1]

    最后的 o_T 不用于训练。
    """


    obs = traj["obs"]

    agent = obs["agent"]

    extra = obs["extra"]


    actions = np.asarray(
        traj["actions"],
        dtype=np.float32,
    )


    T = actions.shape[0]


    # ========================================================
    # Robot joint position
    #
    # shape:
    #
    # (T, 9)
    # ========================================================

    qpos = np.asarray(
        agent["qpos"][:-1],
        dtype=np.float32,
    )


    # ========================================================
    # Robot joint velocity
    #
    # (T, 9)
    # ========================================================

    qvel = np.asarray(
        agent["qvel"][:-1],
        dtype=np.float32,
    )


    # ========================================================
    # is_grasped
    #
    # 原来：
    #
    # (T,)
    #
    # reshape 后：
    #
    # (T, 1)
    # ========================================================

    is_grasped = np.asarray(
        extra["is_grasped"][:-1],
        dtype=np.float32,
    ).reshape(
        T,
        1,
    )


    # ========================================================
    # TCP pose
    #
    # (T, 7)
    # ========================================================

    tcp_pose = np.asarray(
        extra["tcp_pose"][:-1],
        dtype=np.float32,
    )


    # ========================================================
    # Goal position
    #
    # (T, 3)
    # ========================================================

    goal_pos = np.asarray(
        extra["goal_pos"][:-1],
        dtype=np.float32,
    )


    # ========================================================
    # Cube pose
    #
    # (T, 7)
    # ========================================================

    obj_pose = np.asarray(
        extra["obj_pose"][:-1],
        dtype=np.float32,
    )


    # ========================================================
    # TCP -> cube
    #
    # (T, 3)
    # ========================================================

    tcp_to_obj_pos = np.asarray(
        extra["tcp_to_obj_pos"][:-1],
        dtype=np.float32,
    )


    # ========================================================
    # Cube -> goal
    #
    # (T, 3)
    # ========================================================

    obj_to_goal_pos = np.asarray(
        extra["obj_to_goal_pos"][:-1],
        dtype=np.float32,
    )


    # ========================================================
    # 拼接为一个 42 维 observation vector
    #
    # 9 + 9 + 1 + 7 + 3 + 7 + 3 + 3
    #
    # = 42
    # ========================================================

    observations = np.concatenate(
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


    # 安全检查
    assert observations.shape == (
        T,
        42,
    )

    assert actions.shape == (
        T,
        7,
    )


    return (
        observations,
        actions,
    )


# ============================================================
# 4. 读取整个数据集
# ============================================================

print()
print("========================================")
print("Loading BC dataset")
print("========================================")


with h5py.File(
    DATA_PATH,
    "r",
) as f:


    trajectory_names = [
        name
        for name in f.keys()
        if name.startswith("traj_")
    ]


    trajectory_names.sort(
        key=trajectory_number
    )


    print(
        f"Total trajectories: "
        f"{len(trajectory_names)}"
    )


    # --------------------------------------------------------
    # 先在 trajectory 层面随机打乱
    # --------------------------------------------------------

    rng = np.random.default_rng(
        RANDOM_SEED
    )


    shuffled_names = (
        trajectory_names.copy()
    )


    rng.shuffle(
        shuffled_names
    )


    num_trajectories = len(
        shuffled_names
    )


    num_train = int(
        num_trajectories
        * TRAIN_RATIO
    )


    num_val = int(
        num_trajectories
        * VAL_RATIO
    )


    train_names = (
        shuffled_names[:num_train]
    )


    val_names = (
        shuffled_names[
            num_train:
            num_train + num_val
        ]
    )


    test_names = (
        shuffled_names[
            num_train + num_val:
        ]
    )


    print(
        f"Train trajectories: "
        f"{len(train_names)}"
    )

    print(
        f"Val trajectories  : "
        f"{len(val_names)}"
    )

    print(
        f"Test trajectories : "
        f"{len(test_names)}"
    )


    # ========================================================
    # 辅助函数：
    #
    # 多条 trajectory
    # -> 一个大的 sample array
    # ========================================================

    def load_split(names):

        obs_list = []
        action_list = []


        for name in names:

            observations, actions = (
                load_trajectory(
                    f[name]
                )
            )


            obs_list.append(
                observations
            )

            action_list.append(
                actions
            )


        observations = np.concatenate(
            obs_list,
            axis=0,
        )


        actions = np.concatenate(
            action_list,
            axis=0,
        )


        return (
            observations,
            actions,
        )


    train_obs, train_actions = (
        load_split(
            train_names
        )
    )


    val_obs, val_actions = (
        load_split(
            val_names
        )
    )


    test_obs, test_actions = (
        load_split(
            test_names
        )
    )


print()
print("Sample counts:")

print(
    f"Train: {len(train_obs)}"
)

print(
    f"Val  : {len(val_obs)}"
)

print(
    f"Test : {len(test_obs)}"
)


# ============================================================
# 5. Observation normalization
# ============================================================

# 非常重要：
#
# mean/std 只允许用 TRAIN 数据计算。
#
# validation/test 不能参与，
# 否则就是 information leakage。

obs_mean = train_obs.mean(
    axis=0
)

obs_std = train_obs.std(
    axis=0
)


# 某一维如果几乎完全不变化，
# std 可能接近 0。
#
# 防止除以 0。
obs_std[
    obs_std < 1e-6
] = 1.0


train_obs_normalized = (
    train_obs - obs_mean
) / obs_std


val_obs_normalized = (
    val_obs - obs_mean
) / obs_std


test_obs_normalized = (
    test_obs - obs_mean
) / obs_std


# ============================================================
# 6. NumPy -> PyTorch Tensor
# ============================================================

train_x = torch.tensor(
    train_obs_normalized,
    dtype=torch.float32,
)

train_y = torch.tensor(
    train_actions,
    dtype=torch.float32,
)


val_x = torch.tensor(
    val_obs_normalized,
    dtype=torch.float32,
)

val_y = torch.tensor(
    val_actions,
    dtype=torch.float32,
)


test_x = torch.tensor(
    test_obs_normalized,
    dtype=torch.float32,
)

test_y = torch.tensor(
    test_actions,
    dtype=torch.float32,
)


# ============================================================
# 7. Dataset / DataLoader
# ============================================================

train_dataset = TensorDataset(
    train_x,
    train_y,
)


train_loader = DataLoader(
    train_dataset,

    batch_size=BATCH_SIZE,

    # 每个 epoch 都重新打乱训练样本
    shuffle=True,
)


# ============================================================
# 8. 选择计算设备
# ============================================================

if torch.backends.mps.is_available():

    device = torch.device(
        "mps"
    )

elif torch.cuda.is_available():

    device = torch.device(
        "cuda"
    )

else:

    device = torch.device(
        "cpu"
    )


print()
print(
    f"Device: {device}"
)


# ============================================================
# 9. 定义 MLP Policy
# ============================================================

class BCPolicy(nn.Module):

    def __init__(self):

        super().__init__()


        self.network = nn.Sequential(

            # 42 -> 256
            nn.Linear(
                42,
                256,
            ),

            nn.ReLU(),


            # 256 -> 256
            nn.Linear(
                256,
                256,
            ),

            nn.ReLU(),


            # 256 -> 256
            nn.Linear(
                256,
                256,
            ),

            nn.ReLU(),


            # 256 -> 7
            nn.Linear(
                256,
                7,
            ),


            # ManiSkill action:
            #
            # [-1, 1]
            nn.Tanh(),
        )


    def forward(
        self,
        x,
    ):

        return self.network(
            x
        )


model = BCPolicy().to(
    device
)


# ============================================================
# 10. Loss
# ============================================================

criterion = nn.MSELoss()


# ============================================================
# 11. Optimizer
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),

    lr=LEARNING_RATE,
)


# ============================================================
# 12. Validation 函数
# ============================================================

def calculate_loss(
    model,
    x,
    y,
):

    model.eval()


    with torch.no_grad():

        prediction = model(
            x.to(device)
        )


        loss = criterion(
            prediction,
            y.to(device),
        )


    return loss.item()


# ============================================================
# 13. Training
# ============================================================

print()
print("========================================")
print("Training BC policy")
print("========================================")


best_val_loss = float(
    "inf"
)

best_model_state = None
best_epoch = None


for epoch in range(
    1,
    NUM_EPOCHS + 1,
):


    model.train()


    total_train_loss = 0.0

    total_train_samples = 0


    for batch_x, batch_y in (
        train_loader
    ):


        batch_x = batch_x.to(
            device
        )

        batch_y = batch_y.to(
            device
        )


        # ----------------------------------------------------
        # 清空上一 batch 的 gradient
        # ----------------------------------------------------

        optimizer.zero_grad()


        # ----------------------------------------------------
        # Forward
        #
        # observation -> predicted action
        # ----------------------------------------------------

        predicted_action = model(
            batch_x
        )


        # ----------------------------------------------------
        # Expert action 与预测 action 的 MSE
        # ----------------------------------------------------

        loss = criterion(
            predicted_action,
            batch_y,
        )


        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()


        # ----------------------------------------------------
        # 更新 neural network 参数
        # ----------------------------------------------------

        optimizer.step()


        batch_size = (
            batch_x.shape[0]
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


    val_loss = calculate_loss(
        model,
        val_x,
        val_y,
    )


    # ========================================================
    # 保存 validation 最好的那个模型
    # ========================================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_epoch = epoch

        best_model_state = copy.deepcopy(
            model.state_dict()
        )


    # 每 5 个 epoch 打印一次
    if (
        epoch == 1
        or epoch % 5 == 0
    ):

        print(
            f"Epoch {epoch:3d} | "
            f"train MSE={train_loss:.6f} | "
            f"val MSE={val_loss:.6f}"
        )


# ============================================================
# 14. 恢复 validation 最好的模型
# ============================================================

model.load_state_dict(
    best_model_state
)


# ============================================================
# 15. Offline test MSE
# ============================================================

test_loss = calculate_loss(
    model,
    test_x,
    test_y,
)


print()
print("========================================")
print("Training Finished")
print("========================================")

print(
    f"Best epoch   : {best_epoch}"
)

print(
    f"Best val MSE : "
    f"{best_val_loss:.6f}"
)

print(
    f"Test MSE     : "
    f"{test_loss:.6f}"
)


# ============================================================
# 16. 看每一个 action dimension 的误差
# ============================================================

model.eval()


with torch.no_grad():

    test_prediction = model(
        test_x.to(device)
    ).cpu()


squared_error = (
    test_prediction
    - test_y
) ** 2


per_action_mse = (
    squared_error.mean(
        dim=0
    )
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
print("Per-action test MSE:")


for name, mse in zip(
    action_names,
    per_action_mse,
):

    print(
        f"  {name:<8}: "
        f"{mse.item():.6f}"
    )


# ============================================================
# 17. 保存 checkpoint
# ============================================================

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


checkpoint = {

    # 网络参数
    "model_state_dict":
        model.state_dict(),

    # normalization
    "obs_mean":
        torch.tensor(
            obs_mean,
            dtype=torch.float32,
        ),

    "obs_std":
        torch.tensor(
            obs_std,
            dtype=torch.float32,
        ),

    # 模型输入输出维度
    "input_dim": 42,
    "action_dim": 7,

    # observation 拼接顺序
    "observation_fields": [
        "agent/qpos",
        "agent/qvel",
        "extra/is_grasped",
        "extra/tcp_pose",
        "extra/goal_pos",
        "extra/obj_pose",
        "extra/tcp_to_obj_pos",
        "extra/obj_to_goal_pos",
    ],

    # 数据集划分
    "train_trajectories":
        train_names,

    "val_trajectories":
        val_names,

    "test_trajectories":
        test_names,

    # 训练结果
    "best_epoch":
        best_epoch,

    "best_val_loss":
        best_val_loss,
}


torch.save(
    checkpoint,
    CHECKPOINT_PATH,
)


print()
print(
    f"Checkpoint saved to:"
)

print(
    f"  {CHECKPOINT_PATH}"
)