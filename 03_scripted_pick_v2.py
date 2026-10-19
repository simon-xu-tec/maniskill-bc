import time

import numpy as np
import gymnasium as gym
import mani_skill.envs

from expert_v3_policy import (
    expert_action,
    get_cube_pose,
    get_goal_pos,
    get_is_grasped,
    get_success,
    scalar_bool,
)


# ============================================================
# 0. Test settings
# ============================================================

SEED = 0

MAX_EPISODE_STEPS = 300


# 只是 GUI 慢放。
#
# 不改变 simulation，
# 不改变 policy。
DISPLAY_DELAY = 0.10


# ============================================================
# 1. Environment
# ============================================================

env = gym.make(
    "PickCube-v1",

    obs_mode="state_dict",

    control_mode="pd_ee_delta_pose",

    render_mode="human",

    max_episode_steps=
        MAX_EPISODE_STEPS,
)


# ============================================================
# 2. Reset
# ============================================================

obs, info = env.reset(
    seed=SEED
)


cube_pos, _ = (
    get_cube_pose(
        obs
    )
)


goal_pos = (
    get_goal_pos(
        obs
    )
)


print()
print(
    "========================================"
)

print(
    "Expert V3 Scripted Test"
)

print(
    "========================================"
)

print(
    f"Seed: {SEED}"
)

print(
    f"Initial cube: "
    f"{cube_pos}"
)

print(
    f"Goal        : "
    f"{goal_pos}"
)

print()


# ============================================================
# 3. Rollout
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
        # 唯一 expert policy
        # ====================================================

        (
            action,
            debug,
        ) = expert_action(
            obs
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


        time.sleep(
            DISPLAY_DELAY
        )


        now_grasped = (
            get_is_grasped(
                obs
            )
        )


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


        if get_success(
            info
        ):

            success = True


        # ====================================================
        # 每 5 step 打印一次
        # ====================================================

        if step % 5 == 0:


            cube_pos, _ = (
                get_cube_pose(
                    obs
                )
            )


            print(
                f"step={step:3d} | "

                f"mode="
                f"{debug['mode']:<11} | "

                f"grasped="
                f"{now_grasped} | "

                f"dist="
                f"{debug['distance']:.4f} | "

                f"lat="
                f"{debug['lateral']:.4f} | "

                f"angle="
                f"{debug['angle_deg']:.2f}° | "

                f"finger="
                f"{debug['finger_mean']:.4f} | "

                f"grip_act="
                f"{action[6]:+.1f}"
            )


            print(
                f"    cube_z="
                f"{cube_pos[2]:.4f} | "

                f"goal_dist="
                f"{debug['goal_distance']:.4f} | "

                f"xy_goal="
                f"{debug['xy_goal_distance']:.4f} | "

                f"horiz="
                f"{debug['horizontal_permission']:.2f} | "

                f"descend="
                f"{debug['descent_permission']:.2f} | "

                f"path_z="
                f"{debug['path_target_z']:.4f}"
            )


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
# 4. Result
# ============================================================

cube_pos, _ = (
    get_cube_pose(
        obs
    )
)


goal_pos = (
    get_goal_pos(
        obs
    )
)


goal_distance = np.linalg.norm(
    goal_pos
    - cube_pos
)


print()
print(
    "========================================"
)

print(
    "Expert V3 Test Finished"
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
    f"{get_is_grasped(obs)}"
)

print(
    f"Final cube position : "
    f"{cube_pos}"
)

print(
    f"Goal position       : "
    f"{goal_pos}"
)

print(
    f"Final goal distance : "
    f"{goal_distance:.4f} m"
)

print(
    f"Official success    : "
    f"{success}"
)