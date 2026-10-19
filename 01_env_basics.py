import numpy as np
import gymnasium as gym
import mani_skill.envs


# ============================================================
# 1. 创建 PickCube 环境
# ============================================================

env = gym.make(
    "PickCube-v1",

    # 不再把 observation 展平成 42 个没有名字的数，
    # 而是保留字典结构，方便我们学习每一个变量。
    obs_mode="state_dict",

    # 机械臂末端的增量位姿控制：
    #
    # action 前 3 维：TCP 平移增量
    # action 中间 3 维：TCP 旋转增量
    # action 最后 1 维：夹爪
    control_mode="pd_ee_delta_pose",

    # 显示 SAPIEN 的 3D 窗口
    render_mode="human",
)


# ============================================================
# 2. 得到真正的 ManiSkill 环境
# ============================================================

base_env = env.unwrapped


# ============================================================
# 3. reset：开始一个新的 episode
# ============================================================

obs, info = env.reset(seed=0)


# ============================================================
# 4. 查看环境的时间结构
# ============================================================

print("\n========== Environment timing ==========")

print("Simulation frequency:")
print(base_env.sim_freq, "Hz")

print("Control frequency:")
print(base_env.control_freq, "Hz")

print("Simulation timestep:")
print(base_env.sim_timestep, "seconds")

print("Control timestep:")
print(base_env.control_timestep, "seconds")

print("Physics steps per control step:")
print(base_env._sim_steps_per_control)

print("Maximum control steps per episode:")
print(env.spec.max_episode_steps)


# ============================================================
# 5. 定义一个函数，用来把 observation 清楚地打印出来
# ============================================================

def print_observation(obs, title):

    print("\n")
    print("========================================")
    print(title)
    print("========================================")

    # --------------------------------------------------------
    # Robot proprioception
    # --------------------------------------------------------

    print("\n[Robot] qpos")
    print("机器人各关节的位置：")
    print(obs["agent"]["qpos"])

    print("\n[Robot] qvel")
    print("机器人各关节的速度：")
    print(obs["agent"]["qvel"])


    # --------------------------------------------------------
    # Task-specific information
    # --------------------------------------------------------

    print("\n[Task] is_grasped")
    print("当前是否已经抓住方块：")
    print(obs["extra"]["is_grasped"])


    print("\n[Task] tcp_pose")
    print("TCP 的完整位姿：")
    print("(x, y, z, quaternion)")
    print(obs["extra"]["tcp_pose"])


    print("\n[Task] goal_pos")
    print("绿色目标点的位置：")
    print("(x, y, z)")
    print(obs["extra"]["goal_pos"])


    print("\n[Task] obj_pose")
    print("方块的完整位姿：")
    print("(x, y, z, quaternion)")
    print(obs["extra"]["obj_pose"])


    print("\n[Task] tcp_to_obj_pos")
    print("从 TCP 指向方块的相对位置向量：")
    print(obs["extra"]["tcp_to_obj_pos"])


    print("\n[Task] obj_to_goal_pos")
    print("从方块指向目标的相对位置向量：")
    print(obs["extra"]["obj_to_goal_pos"])


# ============================================================
# 6. 打印 reset 后的初始 observation
# ============================================================

print_observation(
    obs,
    "Observation BEFORE action"
)


# ============================================================
# 7. 查看 action space
# ============================================================

print("\n========== Action space ==========")

print(env.action_space)

print("\nAction shape:")
print(env.action_space.shape)

print(
    """
Action 的 7 个分量可以先理解成：

action[0] : TCP x 方向平移增量
action[1] : TCP y 方向平移增量
action[2] : TCP z 方向平移增量

action[3] : x 方向旋转增量
action[4] : y 方向旋转增量
action[5] : z 方向旋转增量

action[6] : gripper
"""
)


# ============================================================
# 8. 创建一个测试 action
# ============================================================

action = np.zeros(
    env.action_space.shape,
    dtype=np.float32,
)

# 给 TCP 一个较小的 +x 方向控制量
#
# 注意：
# 这里的 0.2 是归一化 action，
# 并不表示“移动 0.2 米”。
action[0] = 0.2

# 夹爪保持打开方向
action[6] = 1.0


print("\n========== Test action ==========")
print(action)


# ============================================================
# 9. 执行一个 control step
# ============================================================

obs, reward, terminated, truncated, info = env.step(action)

env.render()


# ============================================================
# 10. 看看一个 action 之后 observation 怎么变化
# ============================================================

print_observation(
    obs,
    "Observation AFTER one control step"
)


print("\n========== Step result ==========")

print("Reward:")
print(reward)

print("Terminated:")
print(terminated)

print("Truncated:")
print(truncated)

print("Info:")
print(info)


# ============================================================
# 11. 接下来保持 TCP 不再主动产生位姿增量
#
#     action 前 6 维全部为 0
#     夹爪保持打开
#
#     一直运行环境。
#
#     每个 episode 结束以后 reset，
#     但程序不会退出。
#
#     在 Terminal 按 Ctrl+C 可以结束程序。
# ============================================================

hold_action = np.zeros(
    env.action_space.shape,
    dtype=np.float32,
)

hold_action[6] = 1.0


try:

    while True:

        obs, reward, terminated, truncated, info = env.step(
            hold_action
        )

        env.render()

        # tensor([True]) -> Python 的 True / False
        episode_finished = (
            bool(terminated.item())
            or
            bool(truncated.item())
        )

        if episode_finished:

            print("\n========== Episode ended ==========")

            print("Terminated:")
            print(bool(terminated.item()))

            print("Truncated:")
            print(bool(truncated.item()))

            print("Reset environment...")

            # 开始新的一局
            obs, info = env.reset()


# ============================================================
# 12. 用户在 Terminal 中按 Ctrl+C
# ============================================================

except KeyboardInterrupt:

    print("\nProgram stopped by user.")


# ============================================================
# 13. 最后释放模拟器和窗口资源
# ============================================================

finally:

    env.close()