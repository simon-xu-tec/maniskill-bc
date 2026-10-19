# ManiSkill BC

基于 ManiSkill PickCube 的行为克隆实验，持续开发中。

| 模块 | 用途 |
| --- | --- |
| `01_env_basics.py` | 查看环境、观测与任务信息 |
| `02_action_basics.py` | 测试机械臂动作控制 |
| `03_scripted_pick.py` | 运行脚本专家抓取 |
| `03_scripted_pick_v2.py` | 调用公共专家策略测试抓取 |
| `04_collect_demos.py` | 采集并保存专家演示 |
| `04_collect_demos_v2.py` | 使用公共专家策略采集演示 |
| `05_inspect_dataset.py` | 检查演示数据与轨迹统计 |
| `06_train_bc.py` | 训练行为克隆模型 |
| `06_train_bc_v2.py` | 使用 v2 数据训练模型 |
| `07_eval_bc.py` | 加载模型并评估抓取 |
| `07_eval_bc_v2.py` | 加载 v2 模型并评估抓取 |
| `expert_v3_policy.py` | 公共专家策略及状态辅助函数 |
| `data/` | 已保存的专家演示及元数据 |
| `checkpoints/` | 已保存的模型权重 |
