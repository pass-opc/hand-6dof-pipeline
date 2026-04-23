# 仿真回放 Pipeline 技术文档

## 0. 文档定位

本文档是本 repo `sim/` 模块的设计与实施指南。回答三个问题：
- **为什么**做仿真回放（价值边界）
- **怎么**做（知识点 + 架构 + 文件清单）
- **现在做到什么程度**（Milestone + 未来一句话延申）

对照 `CLAUDE.md` 红线：sim 是下游消费者（消费 LeRobotDataset），和真机回放 `04_replay_on_arm.py` 并列，不替代、不越位。

---

## 1. 为什么需要仿真回放

### 1.1 用户痛点
- **采数完成，上真机前不知道轨迹好不好**：关节会不会越界、工作空间够不够、有没有自碰撞——只看数字和曲线看不出来
- **L1/L2 数据门禁失败时不直观**：门禁只输出通过/失败 + 指标，定位不到具体是哪一帧出问题
- **重定向调参没有对照系**：dex-retargeting 的参数改了，末端在哪、差多少，没法肉眼验证

### 1.2 仿真回放的作用
- 任选 dataset 里的 episode，在 MuJoCo viewer 里还原机械臂动作
- 作为真机回放（`04_replay_on_arm.py`）的**前置安全检查**——先在仿真跑，再上真机
- L1 门禁的**可视化 debug 入口**（Phase 1.1+ 对接 Rerun 叠加原视频+HaMeR 手部信号）

### 1.3 显式不做的事（避免 scope creep）
- **不做训练平台选型**：Phase 2+ 再评估 ManiSkill 3 / Isaac Lab，现在选就是过早优化
- **Phase 1.0 不加接触物理**：只做运动学回放（`mj_kinematics`），Phase 1.1+ 再加物体和抓取
- **不替代真机验证**：sim 通过 ≠ 真机通过，只是降低真机事故率和调试成本

---

## 2. MuJoCo 基础知识

### 2.1 为什么选 MuJoCo
| 候选 | 选/弃 | 原因 |
|------|------|------|
| MuJoCo + TRS 同源资产 | ✅ | TheRobotStudio 官方 repo 提供 URDF+MJCF 同源对（`Simulation/SO101/`，onshape-to-robot 生成），零点和 LeRobot URDF 天然一致；Windows `pip install mujoco` 即用 |
| ManiSkill 3 | ❌ 当前 | Windows 仅 state-mode，无渲染；Phase 2+ 训练再上 |
| Isaac Lab | ❌ | Windows 不支持 |
| Genesis | ❌ | 无 SO-ARM 资产，生态未成 |
| Rerun | 辅助 | 纯可视化无物理，Phase 1.1 叠加用 |

### 2.2 核心概念

**MJCF vs URDF**：
- MJCF（MuJoCo XML）是 MuJoCo 原生格式，比 URDF 表达力强（支持 keyframe、actuator、sensor、contact 参数等）
- 项目里 `assets/so101_new_calib.urdf` 给 IK/dex-retargeting 用；`assets/mujoco/trs_so101/*.xml` 给仿真用——**两者同源**（都由 TheRobotStudio 的 Onshape CAD 经 onshape-to-robot 生成），关节名、零点、轴方向天然对齐，不需要额外 alias/offset 表

**mjModel vs mjData**：
- `MjModel`：静态结构（关节、link、actuator、几何体），`from_xml_path()` 加载后只读
- `MjData`：运行时状态（`qpos`, `qvel`, `ctrl`, `xpos`, `xquat`...），`mj_step()` 每步更新

**qpos vs ctrl —— 两个数组，不是"两种模式"**：
- 它们是 `mjData` 同时存在的两块内存：
  - `qpos`：广义坐标（关节位置），**真实姿态**的唯一来源；渲染、FK、viewer 右面板读数都看它
  - `ctrl`：actuator 输入缓冲区；对 position actuator 是"目标角度"，`mj_step` 通过 PD（`kp/kv`）把它驱动到 qpos
- loader 本身**不写**这两个数组，只建名字→下标的索引表（`lerobot_to_qpos_idx` / `lerobot_to_ctrl_idx`），回放每一帧按名字查下标再写值
- `len(qpos) != len(ctrl)` 是常见坑：自由关节（`<freejoint>`）占 7 维 qpos 但没有对应 ctrl；驱动关节则 1:1
- **关键区别**：`qpos` = "机械臂在哪里"，`ctrl` = "我让它去哪里"

**运动学 vs 物理 —— 为什么 kinematics 模式要同时写 ctrl+qpos**：
- `mj_step(model, data)`：完整物理步（ctrl → PD → 力 → 积分 → qpos / qvel / 接触解算），Phase 1.1+ 才需要
- `mj_kinematics(model, data)`：**只**从 qpos 算 xpos/xquat（正向运动学），设计上就完全忽略 ctrl——这不是 bug，是 MuJoCo 把"纯几何预览"和"动力学仿真"正交拆开
- 所以 Phase 1.0 回放采用**双写**策略（见 `mujoco_replay.py:134-139`）：
  - `apply_joint_positions_deg(..., target="ctrl")`：喂给 ctrl，让 viewer 右面板的 "Actuator" 滑条和轨迹一致（便于调试对照）
  - `apply_joint_positions_deg(..., target="qpos")` + `mj_kinematics`：直写 qpos 跳过 PD 追踪，让渲染**精确**贴合命令值（否则 kp/kv 会让预览滞后）
- 只写 ctrl 不写 qpos → `mj_kinematics` 不看 ctrl，手臂永远停在 0 位；只写 qpos 不写 ctrl → 右面板 actuator 读数和实际姿态对不上，调试晕

**"不 clamp" 策略**：
- 当 IK 输出的关节角超出 MJCF `jnt_range` 时，`apply_joint_positions_deg` **打 WARN 但不截断**（`mujoco_loader.py:130-135`）
- 如果默默 clamp，上游的 IK 过冲、插值越界、workspace scale 选错会被**掩盖**成"看起来正常的轨迹"；真机跑到物理限位时才炸
- 配套消费方：03 的 `--sim-check` 会通过 grep `[mujoco] WARN` 行数把这个 WARN 升级成"gate 候选信号"（统计、展示，但不阻断 build）

**批量 precompute vs 流式**：
- 回放链路（`05_replay_in_sim.py`）是**一次性把整条轨迹在循环前算完**（EE → retarget → IK 全部做完），再交给 `mujoco_replay` 按帧 `for t in range(T)` 喂进仿真
- 这是行业标准做法：LeRobot `replay_episode`、ROS `bag play`、Isaac Lab 的 dataset replay 都是这样——回放的前提是"轨迹已知"，批量才能统一做可视化/梯度/gate 评估
- 流式只在**真机遥操**（实时 IK）场景才必要；Phase 1.0 不做
- 每帧 `for t` 的循环变量会被下一帧覆盖，没有 list 累积，**不存在爆内存**——跑 10 万帧也只占用单帧的几十字节

**`scene` 是什么**：
- 不是 MJCF 文件，是 `sim.mujoco_loader.LoadedScene` 这个 dataclass，一次性打包：
  - `model` / `data`：MuJoCo 两个原生对象
  - `lerobot_to_qpos_idx` / `lerobot_to_ctrl_idx`：LeRobot 关节名 → 数组下标
  - `joint_range_rad`：`jnt_range` 查表（用于 WARN 判断）
- 下游（replay、tests）只需持有 `scene` 一个变量就能做"按名字写 ctrl/qpos、查 range、做 reset"全部操作

### 2.3 Python API 核心片段

```python
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("assets/mujoco/trs_so101/scene.xml")
data = mujoco.MjData(model)

# Passive viewer: 用户手动控制渲染循环，适合 replay
with mujoco.viewer.launch_passive(model, data) as viewer:
    for t in range(T):
        data.ctrl[:] = action[t]       # position actuators
        mujoco.mj_step(model, data)    # 或 mj_kinematics 不算物理
        viewer.sync()                  # 推一帧到 GUI
        time.sleep(dt)
```

### 2.4 SO-ARM101 同源资产

- **路径**：`assets/mujoco/trs_so101/`（从 `TheRobotStudio/SO-ARM100` 的 `Simulation/SO101/` 拷贝，Apache-2.0）
- **同源性**：`so101_new_calib.urdf`（IK 用）和 `so101_new_calib.xml`（MJCF）由同一份 Onshape CAD 经 `onshape-to-robot` 生成——关节名、零点、旋转轴**完全一致**。IK 输出 `shoulder_lift=+15°` 语义上就等于 MJCF qpos `shoulder_lift=+15°`，不需要 offset 校准。
- **Joint 顺序**：`shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`（和 LeRobot 约定完全一致，loader 里的 `LEROBOT_TO_MJCF` 是 identity 映射）
- **Actuator**：6 个 position actuator（`class="sts3215"`，`kp=998.22`, `kv=2.731`），`ctrl_range` 与 `jnt_range` 对齐
- **Scene**：`scene.xml` 包含机械臂 + 地面 + 灯光；**本地增补**了 keyframe "home"（全 0 姿态，对应校准后的中点位置），不是 upstream 的
- **License**：Apache 2.0，来源标注到 `LICENSE_THIRD_PARTY.md`

### 2.5 已知坑
- **Position actuator 的 kp 默认值可能太软**，导致回放看起来"滞后"——如果有这现象，`<position kp="...">` 调大（当前 TRS 默认 998.22 够用）
- **`launch_passive` 在 Windows 偶现刷新率过低**：设置 `viewer.sync()` 前的 sleep 控制到 ≥ 30 FPS
- **如果将来换 MJCF 来源**（例如切到 Menagerie 的 SO-100 资产），它的关节名是 `Rotation/Pitch/Elbow/Wrist_Pitch/Wrist_Roll/Jaw`，和 LeRobot 不一致——届时只需改 `LEROBOT_TO_MJCF` 一张表，但零点约定也会变，要一起核验
- **GUI 会在轨迹跑完后自动消失**（`launch_passive` with-block 默认行为）。`mujoco_replay.py` 已补了两条出路：
  - 单次回放结束后进入"保持最后一帧直到用户关窗"的循环，方便肉眼检查终态
  - `--loop` 可以无限循环到用户关窗（适合反复对比同一动作）

### 2.6 Viewer 交互速查

Phase 1.0 回放默认用 `mujoco.viewer.launch_passive`，这是 MuJoCo 官方 GLFW viewer。**不要把我说的键位当成权威**——权威来源永远是：

1. 启动后按 **F1** 打开 help overlay，里面列全部键位
2. 右侧面板 Tab 切换：**Rendering**（开关阴影/反射/惯性 visualization）、**Visualization**（参考系、contact 点、COM 标记等滑条 / 复选框）、**Simulation**（暂停 / 单步 / 回放速度）、**Group**（切换 visual / collision 几何体组）
3. 直接在右面板 Actuator / Joint 区拖滑条能手动覆盖 ctrl / qpos，便于排查"哪一个关节出问题"

**确认过的键位**（其他的请 F1 自查）：
- `Space`：暂停 / 继续仿真时钟
- `右键拖` / `中键拖` / `滚轮`：视角旋转 / 平移 / 缩放
- `左键双击 body`：选中，`Ctrl+左键拖` 可施加 perturb force（仅 mj_step 模式有效）

**RGB → XYZ 颜色约定**（确认）：
- **红 = X，绿 = Y，蓝 = Z**
- 来源：MuJoCo 源码 `mjvisualize.h` 的坐标系 gizmo 约定，和 ROS rviz / Gazebo / Blender **完全一致**，属于 3D 图形学通用惯例
- 自查方法：在空场景打开 viewer，`世界坐标系` visualization 开关打开，按右手系心里验证一下三轴指向

### 2.7 换机械臂分层评估

当前 sim 模块是**"LeRobot 5+1 DoF 专用"**设计，不是通用框架。换一个机械臂要动的东西按复用难度分三层：

| 层级 | 文件 | 是否通用 | 换臂要做什么 |
|------|------|---------|--------------|
| L1 MuJoCo 原生 API | `mj_step`, `mj_kinematics`, `launch_passive`, `mjData.ctrl/qpos` | ✅ 100% 通用 | 无 |
| L2 Loader/Replay 骨架 | `mujoco_loader.py`（数据类 + 索引表模式）、`mujoco_replay.py`（per-frame 循环 + viewer lifecycle） | ⚠️ 骨架通用，常量要改 | 改 `LEROBOT_TO_MJCF` 表、`ARM_JOINT_NAMES` 列表；如果新臂 DoF 数不是 5+1 gripper，还要改调用方拼列的逻辑 |
| L3 业务定制 | `scripts/05_replay_in_sim.py`（调用 04 的 IK + retargeting + placement） | ❌ 强绑定 SO-ARM101 | 基本重写；dex-retargeting 参数、IK chain、wrist_chain、auto_placement 都是针对 SO-ARM101 几何标定的 |

**结论**：换臂时 sim/ 模块本身只需改一张映射表 + 一个常量列表；真正要重写的是 `scripts/05` 和上游 04——这就是为什么现在没把 04 的函数抽到 utils：在第二款臂落地前，抽象还不知道该怎么抽。

---

## 3. 架构图

```
LeRobotDataset (v3.0)  [已有]
    │  episode_{id}.parquet: action (6,), observation.state (6,) ...
    ▼
scripts/05_replay_in_sim.py  [入口 CLI]
    │  --dataset <repo_id> --episode <id> [--hz 30] [--no-gui] [--scene <path>]
    ▼
sim/mujoco_loader.py
    │  加载 MJCF、建立 joint_name → qpos_idx 映射、暴露 model/data
    ▼
sim/mujoco_replay.py
    │  for t in range(T):
    │      data.ctrl[:] = action[t] (经过 joint_map)
    │      mj_kinematics(model, data)  # Phase 1.0 纯运动学
    │      viewer.sync()
    ▼
MuJoCo viewer (交互式 3D)
    │  空格暂停 / 右键拖动视角 / 双击 body 查看状态

─────── Phase 1.1 扩展（虚线表示暂未实现）───────
    ▼
sim/rerun_logger.py
    │  同步推送：原视频帧 + HaMeR 手部关节 + 仿真机械臂关节曲线
    ▼
Rerun Viewer (多模态 debug)
```

---

## 4. 脚本文件清单

### Phase 1.0（本次交付）

| 文件 | 作用 | 输入 | 输出 |
|------|------|------|------|
| `sim/__init__.py` | 模块标识 | — | — |
| `sim/mujoco_loader.py` | 加载 MJCF + 场景；构建 `dataset_joint_name → qpos_idx` 映射；封装 model/data 初始化 | `scene_xml_path`, `joint_name_map` | `(MjModel, MjData, joint_idx_list)` |
| `sim/mujoco_replay.py` | 核心回放：逐帧 ctrl 驱动 + viewer 同步；支持暂停/步进/速率控制 | `MjModel, MjData, action_sequence, hz` | 副作用：渲染 |
| `scripts/05_replay_in_sim.py` | CLI 入口：解析 args、读 LeRobotDataset 的指定 episode、调用 replay | CLI args | stdout 日志 |
| `assets/mujoco/trs_so101/` | SO-ARM101 URDF+MJCF 同源资产（从 TheRobotStudio `Simulation/SO101/` 拷贝） | — | — |
| `LICENSE_THIRD_PARTY.md` | 第三方来源 + license 汇总（TheRobotStudio SO-ARM101，Apache-2.0） | — | — |

### Phase 1.1 扩展（本次不做）

| 文件 | 作用 |
|------|------|
| `sim/rerun_logger.py` | 可视化叠加层：原视频 + HaMeR 关节 + 仿真状态同步推 Rerun |
| `assets/mujoco/scene_table.xml` | 自定义桌面场景（加物体） |
| `tests/test_sim_replay.py` | 单元测试：loader 映射正确性、replay 不崩 |

### Phase 2+ 一句话延申

> **接训练平台**：等 Linux 或 WSL2 环境就绪后，评估 ManiSkill 3（SO100 原生）或 Isaac Lab，把 sim 从"回放可视化工具"升级为"策略训练 + 评估环境"。

---

## 5. 现阶段 Milestone（Phase 1.0）

### M1 环境 + 资产就位 ✅ 已完成
- [x] `pip install mujoco` 到 `lerobot` conda env
- [x] `assets/mujoco/trs_so101/` 从 TheRobotStudio `Simulation/SO101/` sparse-checkout 拷贝
- [x] 更新 `LICENSE_THIRD_PARTY.md`（TheRobotStudio SO-ARM101 条目，Apache-2.0）
- [x] `python -c "import mujoco; print(mujoco.__version__)"` 冒烟通过

### M2 viewer 冒烟 ✅ 已完成
- [x] 打开 `trs_so101/scene.xml` 的 viewer，看到机械臂渲染正常
- [x] 在 viewer 里手动拖 slider，6 个 joint 都能动到边界

### M3 Episode 回放 ✅ 已完成
- [x] 实现 `sim/mujoco_loader.py` + `sim/mujoco_replay.py` + `scripts/05_replay_in_sim.py`
- [x] 读 HaMeR v3 数据集 episode 2 并回放（`--scale 0.5` 按 memory 要求）
- [x] `tests/test_sim_replay.py` 14 项全通过（loader + replay + script subprocess smoke）
- [x] GUI lifecycle：跑完不消失、支持 `--loop`、响应关窗

### M4 和真机对比（可选）
- [ ] 同一 episode 仿真 + 真机各跑一遍（`04_replay_on_arm.py --scale 0.5`）
- [ ] 末端位置曲线（sim FK vs real FK）叠加看偏差

### M5 03 pipeline 集成 ✅ 已完成
- [x] `03_build_dataset.py --sim-check` 跑完 build 后批量 headless replay 前 N 个 episode
- [x] 通过 grep `[mujoco] WARN` 分类 PASS / WARN(n) / FAIL
- [x] 设计为**软信号**：不阻断 build，只在终端展示"建议复查"列表

---

## 6. 运行命令速查

```bash
conda activate lerobot
cd hand-6dof-pipeline

# ---- 人工预览：GUI，跑完保持最后一帧 ----
python scripts/05_replay_in_sim.py \
    --dataset-root output/03_dataset_v3 \
    --episode 2 --scale 0.5

# 循环播放（反复对比同一动作）
python scripts/05_replay_in_sim.py \
    --dataset-root output/03_dataset_v3 \
    --episode 2 --scale 0.5 --loop

# ---- 批量 / CI 冒烟：headless 加速，用于门禁 ----
python scripts/05_replay_in_sim.py \
    --dataset-root output/03_dataset_v3 \
    --episode 2 --scale 0.5 \
    --speed 20.0 --no-gui

# ---- 建 dataset 的同时跑 sim-check（03 集成入口） ----
python scripts/03_build_dataset.py \
    --processed output/02_processed.pkl \
    --r3d-dir data/raw \
    --output-dir output/03_dataset_v3 \
    --repo-id <user>/demo_v3 --task "pick up cup" --hand right \
    --sim-check --sim-check-episodes 3 --sim-check-scale 0.5
```

**`--sim-check` 输出样例**：

```
=== Sim-check: 3 episode(s), scale=0.5 ===

  [sim-check] Episode 0 ...
    -> PASS
  [sim-check] Episode 1 ...
    -> 12 out-of-range joint commands
  [sim-check] Episode 2 ...
    -> PASS

=== Sim-check summary ===
  Episode 0: PASS
  Episode 1: WARN (12)
  Episode 2: PASS
  2/3 clean
  NOTE: sim-check is a pre-real-arm filter, not a blocker.
```

---

## 7. 参考资料

- MuJoCo 官方：https://mujoco.readthedocs.io/
- MuJoCo Python API：https://mujoco.readthedocs.io/en/stable/python.html
- TheRobotStudio SO-ARM100/101 官方同源 URDF+MJCF：https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101
- MuJoCo Menagerie（备选 SO-ARM100 资产，注意零点和 LeRobot URDF 不对齐）：https://github.com/google-deepmind/mujoco_menagerie/tree/main/trs_so_arm100
- 参考实现（LeRobot + MuJoCo + Qt UI）：https://github.com/lachlanhurst/so100-mujoco-sim
- Rerun（Phase 1.1 用）：https://github.com/rerun-io/rerun
