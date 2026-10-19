# ManiSkill PickCube Behavior Cloning

基于 ManiSkill `PickCube-v1` 的脚本专家演示采集、状态行为克隆（BC）训练与评估实验。本文按当前目录中的实际文件整理，不将文件名中的版本号视为已完成的实验结果。

## 当前文件与流程

| 文件 | 用途 / 当前默认输入输出 |
| --- | --- |
| `01_env_basics.py` | 环境与 observation 入门 |
| `02_action_basics.py` | 动作控制入门 |
| `03_scripted_pick.py` | 原始脚本抓取实验 |
| `04_collect_demos.py` | 采集到 `data/expert_demos/pickcube_scripted.{h5,json}` |
| `06_train_bc.py` | 读取原始数据，保存 `checkpoints/bc_state_mlp.pt` |
| `07_eval_bc.py` | 加载原始 checkpoint 进行 rollout |
| `03_scripted_pick_v2.py` | 使用实际存在的 `expert_v3_policy.py` 测试专家 |
| `04_collect_demos_v2.py` | 使用同一专家，当前输出配置见下文 |
| `05_inspect_dataset.py` | 检查 `data/expert_demos_v2/pickcube_arc_scripted.{h5,json}` |
| `06_train_bc_v2.py` | 读取现有 v2 数据，保存 `checkpoints/bc_state_mlp_v2.pt` |
| `07_eval_bc_v2.py` | 加载现有 v2 checkpoint 进行 rollout |
| `expert_v3_policy.py` | 两个 `_v2` 专家/采集脚本直接导入的公共策略模块 |

### 已存在的实验文件

- `data/expert_demos/pickcube_scripted.h5` 与配套 JSON；JSON 中有 100 条 episode 记录。
- `data/expert_demos_v2/pickcube_arc_scripted.h5` 与配套 JSON；JSON 中有 100 条 episode 记录。
- `checkpoints/bc_state_mlp.pt`。
- `checkpoints/bc_state_mlp_v2.pt`。

两个数据集的 JSON 均声明 `PickCube-v1`、`state_dict` observation、`pd_ee_delta_pose` 控制和 300 步 episode 上限。以上数量来自 JSON 元数据，未据此推断模型成功率或 HDF5 内容校验结果。

### 文件名与采集配置的差异

`expert_v3_policy.py` 确实存在，保留原名以维持现有 import。当前 `04_collect_demos_v2.py` 将输出目录设为 `data/expert_demos_v3`，轨迹名为 `pickcube_canonical_arc_scripted`，但当前目录中没有这套输出，也没有 v3 训练脚本、评估脚本或 checkpoint。

因此，当前 `_v2` 采集脚本不应被描述为已有 v2 数据的可复现生成器；它与已保存 v2 数据之间的对应关系尚未验证。运行它会创建新的输出目录，不会自动切换 v2 训练脚本的输入。文档保留这一差异，不通过改名或替换数据掩盖现状。

## 环境准备

在项目根目录运行以下命令：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

原始环境未附带依赖锁文件，因此 `requirements.txt` 仅列出源码直接使用的依赖，不保证还原原始软件版本。ManiSkill/SAPIEN 仿真、图形驱动、任务资源和 GUI 支持需在实际运行机器上配置并验证。训练脚本包含设备选择逻辑；Apple Silicon 的 MPS 训练支持不代表仿真后端已验证。

## 使用现有数据

所有命令都应从项目根目录执行，脚本使用相对路径。参数主要配置在各脚本顶部，当前没有统一命令行接口。

```bash
# 检查现有 v2 数据
python 05_inspect_dataset.py

# 使用现有 checkpoint 评估（需要仿真与 GUI）
python 07_eval_bc.py
python 07_eval_bc_v2.py
```

重新训练会写入相应的现有 checkpoint 路径；需要保留旧模型时请先备份或修改脚本顶部路径：

```bash
python 06_train_bc.py
python 06_train_bc_v2.py
```

重新采集前应检查输出路径并另选目录，以保留已有数据：

```bash
python 03_scripted_pick.py
python 03_scripted_pick_v2.py
python 04_collect_demos.py
python 04_collect_demos_v2.py
```

v2 训练与评估固定使用相同顺序的 8 个 observation 字段（总计 42 维）：`agent/qpos`、`agent/qvel`、`extra/is_grasped`、`extra/tcp_pose`、`extra/goal_pos`、`extra/obj_pose`、`extra/tcp_to_obj_pos`、`extra/obj_to_goal_pos`。修改输入时需同步检查训练和评估代码。

## 仓库范围与验证状态

保留现有脚本、两套数据和两个 checkpoint，忽略 Python 缓存、虚拟环境、编辑器设置和运行日志。未添加实验成绩、许可证或未实际存在的版本产物。

整理时仅执行 Python 静态语法检查、源码路径核对和 JSON 元数据检查。当前系统 Python 缺少运行依赖，未执行仿真、训练、rollout 或 checkpoint 加载；现有模型表现需在配置好的环境中实测。
