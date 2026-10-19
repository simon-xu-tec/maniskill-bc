import os

import numpy as np
import gymnasium as gym
import mani_skill.envs

from mani_skill.utils.wrappers.record import (
    RecordEpisode,
)

from expert_v3_policy import (
    expert_action,
    get_is_grasped,
    get_success,
    scalar_bool,
)


# ============================================================
# 0. Collection settings
# ============================================================

NUM_DEMOS = 100


MAX_ATTEMPTS = 500


MAX_EPISODE_STEPS = 300


OUTPUT_DIR = (
    "data/expert_demos_v3"
)


TRAJECTORY_NAME = (
    "pickcube_canonical_arc_scripted"
)


# ------------------------------------------------------------
# 如果已经成功抓住后又丢失：
#
# 当前整条 trajectory 不保存。
#
# 注意：
# 这只影响“是否保存”，
# 不影响 expert_action。
# ------------------------------------------------------------

REQUIRE_CLEAN_GRASP = True


# ============================================================
# 1. Output directory
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# 2. ManiSkill environment
# ============================================================

base_env = gym.make(
    "PickCube-v1",

    obs_mode="state_dict",

    control_mode="pd_ee_delta_pose",

    # 批量采集不显示 GUI
    render_mode=None,

    max_episode_steps=
        MAX_EPISODE_STEPS,
)


# ============================================================
# 3. Recorder
# ============================================================

env = RecordEpisode(
    base_env,

    output_dir=
        OUTPUT_DIR,

    trajectory_name=
        TRAJECTORY_NAME,

    save_video=False,

    # 我们自己决定成功才保存
    save_on_reset=False,

    # BC 不需要 reward label
    record_reward=False,

    # 第一版 BC 不需要完整 simulator state
    record_env_state=False,

    source_type="scripted",

    source_desc=(
        "Expert V3: state-driven canonical grasp "
        "orientation, stable state-based gripper "
        "closing rule, and continuous arc transport."
    ),
)


# ============================================================
# 4. Run one episode
# ============================================================

def run_episode(
    seed,
):


    obs, info = env.reset(
        seed=seed
    )


    # ========================================================
    # reset 后已经 success：
    #
    # 没有有效 obs -> action pair，
    # 所以 skip。
    # ========================================================

    initial_eval = (
        env.unwrapped.evaluate()
    )


    if get_success(
        initial_eval
    ):


        env.flush_trajectory(
            save=False
        )


        return (
            "skip",
            0,
            "initial_success",
        )


    # ========================================================
    # 以下状态只负责：
    #
    # 判断 trajectory 是否值得保存。
    #
    # 不传给 expert_action。
    # ========================================================

    ever_grasped = False

    lost_grasp = False


    episode_length = 0


    for step in range(
        MAX_EPISODE_STEPS
    ):


        # ====================================================
        # 当前 observation
        # ->
        # 当前 expert action
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


        episode_length += 1


        now_grasped = (
            get_is_grasped(
                obs
            )
        )


        if now_grasped:

            ever_grasped = True


        # ====================================================
        # 曾经抓住过，现在又没抓住：
        #
        # 说明发生掉落。
        # ====================================================

        if (
            ever_grasped
            and
            not now_grasped
        ):

            lost_grasp = True


        # ====================================================
        # 清洁 demonstration：
        #
        # 一旦掉落，这一整条不保存。
        # ====================================================

        if (
            REQUIRE_CLEAN_GRASP
            and
            lost_grasp
        ):


            env.flush_trajectory(
                save=False
            )


            return (
                "failure",
                episode_length,
                "lost_grasp",
            )


        # ====================================================
        # Official success
        # ====================================================

        if get_success(
            info
        ):


            env.flush_trajectory(
                save=True
            )


            return (
                "success",
                episode_length,
                "success",
            )


        # ====================================================
        # Environment 已终止
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


            env.flush_trajectory(
                save=False
            )


            if scalar_bool(
                truncated
            ):

                reason = (
                    "truncated"
                )

            else:

                reason = (
                    "terminated_without_success"
                )


            return (
                "failure",
                episode_length,
                reason,
            )


    # 理论保险
    env.flush_trajectory(
        save=False
    )


    return (
        "failure",
        episode_length,
        "max_steps",
    )


# ============================================================
# 5. Batch collection
# ============================================================

saved_demos = 0

attempted_seeds = 0

skipped_initial_success = 0

failed_attempts = 0


successful_lengths = []


failure_reasons = {}


print()
print(
    "========================================"
)

print(
    "Collect Expert V3 Demonstrations"
)

print(
    "========================================"
)

print(
    f"Target demos : "
    f"{NUM_DEMOS}"
)

print(
    f"Output dir   : "
    f"{OUTPUT_DIR}"
)

print(
    f"Trajectory   : "
    f"{TRAJECTORY_NAME}"
)

print()


try:


    for seed in range(
        MAX_ATTEMPTS
    ):


        if (
            saved_demos
            >= NUM_DEMOS
        ):

            break


        attempted_seeds += 1


        (
            status,
            length,
            reason,
        ) = run_episode(
            seed
        )


        # ====================================================
        # Success
        # ====================================================

        if status == "success":


            saved_demos += 1


            successful_lengths.append(
                length
            )


            print(
                f"[SAVE] "
                f"seed={seed:3d} | "
                f"demo="
                f"{saved_demos:3d}/"
                f"{NUM_DEMOS} | "
                f"steps={length:3d}"
            )


        # ====================================================
        # Reset already successful
        # ====================================================

        elif status == "skip":


            skipped_initial_success += 1


            print(
                f"[SKIP] "
                f"seed={seed:3d} | "
                f"{reason}"
            )


        # ====================================================
        # Failure
        # ====================================================

        else:


            failed_attempts += 1


            failure_reasons[
                reason
            ] = (
                failure_reasons.get(
                    reason,
                    0,
                )
                + 1
            )


            print(
                f"[FAIL] "
                f"seed={seed:3d} | "
                f"steps={length:3d} | "
                f"{reason}"
            )


finally:

    env.close()


# ============================================================
# 6. Statistics
# ============================================================

print()
print(
    "========================================"
)

print(
    "Collection Finished"
)

print(
    "========================================"
)


print(
    f"Attempted seeds        : "
    f"{attempted_seeds}"
)


print(
    f"Saved demonstrations   : "
    f"{saved_demos}"
)


print(
    f"Skipped initial success: "
    f"{skipped_initial_success}"
)


print(
    f"Failed attempts        : "
    f"{failed_attempts}"
)


if len(
    successful_lengths
) > 0:


    lengths = np.asarray(
        successful_lengths,
        dtype=np.int32,
    )


    print(
        f"Mean demo length       : "
        f"{lengths.mean():.2f}"
    )


    print(
        f"Shortest demo          : "
        f"{lengths.min()}"
    )


    print(
        f"Longest demo           : "
        f"{lengths.max()}"
    )


if len(
    failure_reasons
) > 0:


    print()
    print(
        "Failure reasons:"
    )


    for (
        reason,
        count,
    ) in failure_reasons.items():


        print(
            f"  "
            f"{reason:<28}: "
            f"{count}"
        )


print()
print(
    "Generated files:"
)


print(
    os.path.join(
        OUTPUT_DIR,
        TRAJECTORY_NAME
        + ".h5",
    )
)


print(
    os.path.join(
        OUTPUT_DIR,
        TRAJECTORY_NAME
        + ".json",
    )
)