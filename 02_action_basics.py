import time
import numpy as np
import gymnasium as gym
import mani_skill.envs


# ============================================================
# 1. 创建环境
# ============================================================

env = gym.make(
    "PickCube-v1",
    obs_mode="state_dict",
    control_mode="pd_ee_delta_pose",
    render_mode="human",
)

obs, info = env.reset(seed=0)


# ============================================================
# 2. 读取 TCP 位置
# ============================================================

def get_tcp_position(obs):
    tcp_pose = obs["extra"]["tcp_pose"]

    # 前 3 个数就是 TCP 的 x, y, z
    return tcp_pose[..., :3]


initial_tcp = get_tcp_position(obs)

print("\n初始 TCP position:")
print(initial_tcp)


# ============================================================
# 3. 创建 +x action
# ============================================================

action = np.zeros(
    env.action_space.shape,
    dtype=np.float32,
)

# +x 方向
action[0] = 0.2

# 夹爪保持打开
action[6] = 1.0

print("\nAction:")
print(action)


# ============================================================
# 4. 连续执行 10 个 +x control steps
# ============================================================

for i in range(10):

    obs, reward, terminated, truncated, info = env.step(action)

    env.render()

    tcp = get_tcp_position(obs)

    print("\n-------------------------")
    print("Control step:", i)
    print("TCP position:")
    print(tcp)

    print("相对初始位置的变化:")
    print(tcp - initial_tcp)

    # 只是让你肉眼看得清
    # 不会增加模拟时间
    time.sleep(0.5)


# ============================================================
# 5. 停在最终位置
# ============================================================

hold_action = np.zeros(
    env.action_space.shape,
    dtype=np.float32,
)

hold_action[6] = 1.0


for i in range(10):

    obs, reward, terminated, truncated, info = env.step(
        hold_action
    )

    env.render()

    time.sleep(0.2)


# ============================================================
# 6. 保持窗口
# ============================================================

input("\n观察完成后按 Enter 退出...")

env.close()