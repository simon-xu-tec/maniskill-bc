import json
import os

import h5py
import numpy as np


# ============================================================
# 0. 文件路径
# ============================================================

H5_PATH = (
    "data/expert_demos_v2/"
    "pickcube_arc_scripted.h5"
)

JSON_PATH = (
    "data/expert_demos_v2/"
    "pickcube_arc_scripted.json"
)


# ============================================================
# 1. 检查文件是否存在
# ============================================================

if not os.path.exists(H5_PATH):

    raise FileNotFoundError(
        f"HDF5 file not found:\n"
        f"{H5_PATH}"
    )


if not os.path.exists(JSON_PATH):

    raise FileNotFoundError(
        f"JSON file not found:\n"
        f"{JSON_PATH}"
    )


print()
print(
    "========================================"
)

print(
    "Inspect Expert V2 Dataset"
)

print(
    "========================================"
)

print(
    f"HDF5: {H5_PATH}"
)

print(
    f"JSON : {JSON_PATH}"
)

print()


# ============================================================
# 2. 工具函数：
#    把 traj_10 这种名字按照数字排序
# ============================================================

def trajectory_index(name):

    # "traj_37"
    # ->
    # 37

    return int(
        name.split("_")[-1]
    )


# ============================================================
# 3. 工具函数：
#    递归打印 HDF5 group 结构
# ============================================================

def print_h5_structure(
    group,
    indent=0,
):

    prefix = (
        "  " * indent
    )


    for key in group.keys():


        item = group[key]


        if isinstance(
            item,
            h5py.Group,
        ):

            print(
                f"{prefix}"
                f"{key}/"
            )


            print_h5_structure(
                item,
                indent + 1,
            )


        elif isinstance(
            item,
            h5py.Dataset,
        ):

            print(
                f"{prefix}"
                f"{key}: "
                f"shape={item.shape}, "
                f"dtype={item.dtype}"
            )


# ============================================================
# 4. 工具函数：
#    读取 scalar bool
# ============================================================

def final_bool(dataset):

    x = np.asarray(
        dataset
    )

    return bool(
        x.reshape(-1)[-1]
    )


# ============================================================
# 5. 读取 HDF5
#
# "r" = read-only
#
# 这里只读取，不会修改数据文件。
# ============================================================

with h5py.File(
    H5_PATH,
    "r",
) as f:


    # ========================================================
    # 5.1 顶层 trajectory 名称
    # ========================================================

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
        "----------------------------------------"
    )

    print(
        "1. HDF5 overview"
    )

    print(
        "----------------------------------------"
    )


    print(
        f"Number of trajectories: "
        f"{num_trajectories}"
    )


    print(
        f"First trajectory       : "
        f"{trajectory_names[0]}"
    )


    print(
        f"Last trajectory        : "
        f"{trajectory_names[-1]}"
    )

    print()


    # ========================================================
    # 5.2 统计每条 trajectory 的 action 数量
    #
    # 一条长度 T 的轨迹：
    #
    # actions:
    #     T
    #
    # observations:
    #     T + 1
    # ========================================================

    lengths = []


    for traj_name in trajectory_names:


        traj = f[
            traj_name
        ]


        T = traj[
            "actions"
        ].shape[0]


        lengths.append(
            T
        )


    lengths = np.asarray(
        lengths,
        dtype=np.int32,
    )


    print(
        "----------------------------------------"
    )

    print(
        "2. Trajectory length statistics"
    )

    print(
        "----------------------------------------"
    )


    print(
        f"Number of trajectories : "
        f"{len(lengths)}"
    )


    print(
        f"Mean length            : "
        f"{lengths.mean():.2f}"
    )


    print(
        f"Shortest trajectory    : "
        f"{lengths.min()}"
    )


    print(
        f"Longest trajectory     : "
        f"{lengths.max()}"
    )


    print(
        f"Total action samples   : "
        f"{lengths.sum()}"
    )

    print()


    # ========================================================
    # 5.3 打印第一条 trajectory 的完整结构
    # ========================================================

    first_name = (
        trajectory_names[0]
    )


    first_traj = f[
        first_name
    ]


    print(
        "----------------------------------------"
    )

    print(
        f"3. Structure of {first_name}"
    )

    print(
        "----------------------------------------"
    )


    print_h5_structure(
        first_traj
    )

    print()


    # ========================================================
    # 5.4 检查 T observations / actions 对齐
    # ========================================================

    T_action = first_traj[
        "actions"
    ].shape[0]


    T_qpos = first_traj[
        "obs"
    ][
        "agent"
    ][
        "qpos"
    ].shape[0]


    print(
        "----------------------------------------"
    )

    print(
        "4. T / T+1 consistency"
    )

    print(
        "----------------------------------------"
    )


    print(
        f"actions length      : "
        f"{T_action}"
    )


    print(
        f"observations length : "
        f"{T_qpos}"
    )


    print(
        f"obs = action + 1    : "
        f"{T_qpos == T_action + 1}"
    )

    print()


    # ========================================================
    # 5.5 检查所有轨迹是不是都满足 T+1
    # ========================================================

    all_length_consistent = True


    for traj_name in trajectory_names:


        traj = f[
            traj_name
        ]


        T = traj[
            "actions"
        ].shape[0]


        obs_T = traj[
            "obs"
        ][
            "agent"
        ][
            "qpos"
        ].shape[0]


        if obs_T != T + 1:


            all_length_consistent = False


            print(
                f"[WARNING] "
                f"{traj_name}: "
                f"actions={T}, "
                f"obs={obs_T}"
            )


    print(
        f"All trajectories satisfy "
        f"obs = action + 1: "
        f"{all_length_consistent}"
    )

    print()


    # ========================================================
    # 5.6 第一帧 observation
    # ========================================================

    print(
        "----------------------------------------"
    )

    print(
        "5. First observation example"
    )

    print(
        "----------------------------------------"
    )


    first_qpos = np.asarray(
        first_traj[
            "obs"
        ][
            "agent"
        ][
            "qpos"
        ][0]
    )


    first_qvel = np.asarray(
        first_traj[
            "obs"
        ][
            "agent"
        ][
            "qvel"
        ][0]
    )


    first_tcp = np.asarray(
        first_traj[
            "obs"
        ][
            "extra"
        ][
            "tcp_pose"
        ][0]
    )


    first_cube = np.asarray(
        first_traj[
            "obs"
        ][
            "extra"
        ][
            "obj_pose"
        ][0]
    )


    first_goal = np.asarray(
        first_traj[
            "obs"
        ][
            "extra"
        ][
            "goal_pos"
        ][0]
    )


    first_grasped = np.asarray(
        first_traj[
            "obs"
        ][
            "extra"
        ][
            "is_grasped"
        ][0]
    )


    print(
        "qpos:"
    )

    print(
        first_qpos
    )


    print()
    print(
        "qvel:"
    )

    print(
        first_qvel
    )


    print()
    print(
        "tcp_pose:"
    )

    print(
        first_tcp
    )


    print()
    print(
        "obj_pose:"
    )

    print(
        first_cube
    )


    print()
    print(
        "goal_pos:"
    )

    print(
        first_goal
    )


    print()
    print(
        "is_grasped:"
    )

    print(
        first_grasped
    )

    print()


    # ========================================================
    # 5.7 第一条 action
    # ========================================================

    actions_first_traj = np.asarray(
        first_traj[
            "actions"
        ]
    )


    print(
        "----------------------------------------"
    )

    print(
        "6. First actions"
    )

    print(
        "----------------------------------------"
    )


    print(
        "action format:"
    )

    print(
        "[dx, dy, dz, dRx, dRy, dRz, gripper]"
    )

    print()


    num_show = min(
        10,
        len(
            actions_first_traj
        ),
    )


    for i in range(
        num_show
    ):


        print(
            f"action[{i:2d}] = "
            f"{actions_first_traj[i]}"
        )


    print()


    # ========================================================
    # 5.8 第一条轨迹最终状态
    # ========================================================

    print(
        "----------------------------------------"
    )

    print(
        "7. First trajectory outcome"
    )

    print(
        "----------------------------------------"
    )


    if (
        "terminated"
        in first_traj
    ):

        print(
            f"final terminated: "
            f"{final_bool(first_traj['terminated'])}"
        )


    if (
        "truncated"
        in first_traj
    ):

        print(
            f"final truncated : "
            f"{final_bool(first_traj['truncated'])}"
        )


    if (
        "success"
        in first_traj
    ):

        print(
            f"final success   : "
            f"{final_bool(first_traj['success'])}"
        )


    print()


    # ========================================================
    # 5.9 检查所有 trajectory outcome
    # ========================================================

    success_count = 0

    terminated_count = 0

    truncated_count = 0


    for traj_name in trajectory_names:


        traj = f[
            traj_name
        ]


        if (
            "success"
            in traj
            and
            final_bool(
                traj["success"]
            )
        ):

            success_count += 1


        if (
            "terminated"
            in traj
            and
            final_bool(
                traj["terminated"]
            )
        ):

            terminated_count += 1


        if (
            "truncated"
            in traj
            and
            final_bool(
                traj["truncated"]
            )
        ):

            truncated_count += 1


    print(
        "----------------------------------------"
    )

    print(
        "8. Outcome statistics"
    )

    print(
        "----------------------------------------"
    )


    print(
        f"Successful trajectories : "
        f"{success_count}/"
        f"{num_trajectories}"
    )


    print(
        f"Final terminated=True   : "
        f"{terminated_count}"
    )


    print(
        f"Final truncated=True    : "
        f"{truncated_count}"
    )

    print()


    # ========================================================
    # 5.10 把所有 actions 合起来
    #
    # 用于检查 V2 expert action distribution。
    # ========================================================

    all_actions = []


    for traj_name in trajectory_names:


        actions = np.asarray(
            f[
                traj_name
            ][
                "actions"
            ]
        )


        all_actions.append(
            actions
        )


    all_actions = np.concatenate(
        all_actions,
        axis=0,
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


    print(
        "----------------------------------------"
    )

    print(
        "9. Action statistics"
    )

    print(
        "----------------------------------------"
    )


    print(
        f"All actions shape: "
        f"{all_actions.shape}"
    )

    print()


    print(
        f"{'dim':<10}"
        f"{'mean':>12}"
        f"{'std':>12}"
        f"{'min':>12}"
        f"{'max':>12}"
    )


    print(
        "-" * 58
    )


    for i, name in enumerate(
        action_names
    ):


        values = all_actions[
            :,
            i
        ]


        print(
            f"{name:<10}"
            f"{values.mean():>12.6f}"
            f"{values.std():>12.6f}"
            f"{values.min():>12.6f}"
            f"{values.max():>12.6f}"
        )


    print()


    # ========================================================
    # 5.11 Gripper action 分布
    # ========================================================

    gripper = all_actions[
        :,
        6
    ]


    open_count = np.sum(
        gripper > 0
    )


    close_count = np.sum(
        gripper < 0
    )


    zero_count = np.sum(
        np.isclose(
            gripper,
            0.0,
        )
    )


    print(
        "----------------------------------------"
    )

    print(
        "10. Gripper action distribution"
    )

    print(
        "----------------------------------------"
    )


    print(
        f"Open  (>0): "
        f"{open_count}"
    )


    print(
        f"Close (<0): "
        f"{close_count}"
    )


    print(
        f"Zero      : "
        f"{zero_count}"
    )

    print()


# ============================================================
# 6. 读取 JSON metadata
# ============================================================

with open(
    JSON_PATH,
    "r",
    encoding="utf-8",
) as file:


    metadata = json.load(
        file
    )


print(
    "----------------------------------------"
)

print(
    "11. JSON metadata"
)

print(
    "----------------------------------------"
)


print(
    "Top-level keys:"
)

print(
    list(
        metadata.keys()
    )
)

print()


# ============================================================
# 7. env_info
# ============================================================

env_info = metadata.get(
    "env_info",
    {},
)


print(
    "env_info:"
)


for key, value in (
    env_info.items()
):


    print(
        f"  {key}: "
        f"{value}"
    )


print()


# ============================================================
# 8. Source information
# ============================================================

print(
    "source_type:"
)

print(
    metadata.get(
        "source_type",
        None,
    )
)


print()


print(
    "source_desc:"
)

print(
    metadata.get(
        "source_desc",
        None,
    )
)


print()


# ============================================================
# 9. Episodes metadata
# ============================================================

episodes = metadata.get(
    "episodes",
    [],
)


print(
    "----------------------------------------"
)

print(
    "12. Episode metadata"
)

print(
    "----------------------------------------"
)


print(
    f"Number of JSON episodes: "
    f"{len(episodes)}"
)

print()


if len(
    episodes
) > 0:


    print(
        "First episode metadata:"
    )


    for key, value in (
        episodes[0].items()
    ):


        print(
            f"  {key}: "
            f"{value}"
        )


    print()


# ============================================================
# 10. JSON episode length statistics
# ============================================================

json_lengths = []


for episode in episodes:


    if (
        "elapsed_steps"
        in episode
    ):


        json_lengths.append(
            episode[
                "elapsed_steps"
            ]
        )


if len(
    json_lengths
) > 0:


    json_lengths = np.asarray(
        json_lengths,
        dtype=np.int32,
    )


    print(
        "JSON elapsed_steps:"
    )


    print(
        f"  mean : "
        f"{json_lengths.mean():.2f}"
    )


    print(
        f"  min  : "
        f"{json_lengths.min()}"
    )


    print(
        f"  max  : "
        f"{json_lengths.max()}"
    )


    print()


# ============================================================
# 11. Final summary
# ============================================================

print(
    "========================================"
)

print(
    "Dataset Inspection Finished"
)

print(
    "========================================"
)


print(
    f"Trajectories : "
    f"{num_trajectories}"
)


print(
    f"Total samples: "
    f"{lengths.sum()}"
)


print(
    f"Mean length  : "
    f"{lengths.mean():.2f}"
)


print(
    f"Success      : "
    f"{success_count}/"
    f"{num_trajectories}"
)


print(
    f"Truncated    : "
    f"{truncated_count}"
)


print(
    f"T/T+1 valid : "
    f"{all_length_consistent}"
)