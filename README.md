# hand-6dof-pipeline

**iPhone RGB-D 采集 → 裸手 6DoF 追踪 → LeRobot v3 数据集 → SO-ARM 101 真机 / MuJoCo 仿真回放**

为具身智能策略（DP / ACT 等）提供"人类操作演示"训练数据的端到端离线流水线。上游单目 RGB-D + LiDAR（Record3D / ARKit），下游产出 [LeRobot v3](https://github.com/huggingface/lerobot) Parquet + MP4 数据集。

---

## 1. Pipeline 主线

```
   .r3d  (Record3D, ARKit VIO + LiDAR @ 60 fps)
     │
     ▼
 ┌──────────────────────────────────────────────────────────────┐
 │ [01] 01_hand_track.py         (感知：HaMeR + MediaPipe + LiDAR)│
 │      cam-frame tracking.pkl                                   │
 └──────────────────────────────────────────────────────────────┘
     │
     ▼
 ┌──────────────────────────────────────────────────────────────┐
 │ [02] 02_process.py            (世界系 + 质量门控 + 滤波)       │
 │      processed.pkl （episode-local world frame, 7D state）     │
 └──────────────────────────────────────────────────────────────┘
     │
     ▼
 ┌──────────────────────────────────────────────────────────────┐
 │ [03] 03_build_dataset.py      (LeRobot v3 打包 ± sim-check)    │
 │      Parquet + MP4（可直接训 DP / ACT）                        │
 └──────────────────────────────────────────────────────────────┘
     │
     ├──────────────────────┬──────────────────────┐
     ▼                      ▼                      ▼
  策略训练              [04] 真机回放           [05] 仿真回放
  (LeRobot)         04_replay_on_arm.py     05_replay_in_sim.py
                    SO-ARM 101 + IK         MuJoCo + same-source MJCF
```

| 步骤 | 输入 | 输出 | 关键能力 |
|:---|:---|:---|:---|
| 01 | `.r3d` 目录 | `01_tracking/*.pkl` | HaMeR 裸手 6DoF 检测，MediaPipe bbox 先验，LiDAR 深度透视修正 |
| 02 | 01 pkl | `02_processed/*.pkl` | ARKit 世界系变换，robust anchor，质量门控，NaN 插值，One-Euro 滤波 |
| 03 | 02 pkl + `.r3d` | `03_dataset_v3/` | 流式读 r3d → resize → LeRobot v3 打包；可选 `--sim-check` 触发 05 烟测 |
| 04 | 03 数据集 | 真机动作 | retarget → IK（wrist sub-chain）→ wrist_roll anchor → SafeHome |
| 05 | 03 数据集 | MuJoCo 窗口 / headless | same-source URDF+MJCF，`ctrl = IK 解`，`qpos = 原始 dataset`（双写） |

详细技术文档：
- **[docs/HaMeR_Pipeline_Guide.md](docs/HaMeR_Pipeline_Guide.md)** — 01 → 04 主线全栈
- **[docs/Sim_Replay_Guide.md](docs/Sim_Replay_Guide.md)** — 05 仿真回放 + sim-check 集成
- **[docs/keywords.md](docs/keywords.md)** — 术语速查

---

## 2. 环境配置

### 2.1 依赖 & Python

- Python **3.10+**（推荐 miniconda `lerobot` env，Python 3.12）
- 核心：`numpy`, `opencv-contrib-python`, `scipy`, `pandas`, `pyarrow`
- 管线：`ikpy`（IK）、`lerobot`（数据集 + SO100Follower 驱动）、`record3d`、`pyliblzfse`（LiDAR LZFSE 解压）
- 感知：`torch`, `detectron2`, `smplx`, `timm`, `pytorch-lightning`（HaMeR），`mediapipe`
- 仿真：`mujoco`（≥ 3.x）

```bash
# 推荐使用 lerobot 预置环境
conda activate lerobot
pip install ikpy record3d pyliblzfse mediapipe mujoco
# HaMeR 依赖（torch、detectron2 请按官方说明匹配 CUDA 版本）
```

### 2.2 资产下载

| 资产 | 路径 | 获取方式 |
|:---|:---|:---|
| SO-ARM 101 URDF | `assets/so101_new_calib.urdf` | ✅ 仓库自带（Apache-2.0） |
| SO-ARM 101 MJCF | `assets/mujoco/trs_so101/` | ✅ 仓库自带（Apache-2.0，同源于 URDF） |
| MediaPipe hand_landmarker | `assets/mediapipe/hand_landmarker.task` | ✅ 仓库自带 |
| HaMeR 模型权重 + MANO | `assets/hamer/_DATA/` | ⚠️ 需单独下载（~2GB） |

```bash
# 下载 HaMeR 权重
python scripts/_download_hamer.py

# 验证环境
python scripts/_verify_hamer_env.py
```

### 2.3 数据准备

把 Record3D 导出的 `.r3d` 文件放进 `data/` 目录（`data/` 已 gitignore）：

```
data/
├── raw/
│   ├── 2026-04-15--episode-00.r3d
│   ├── 2026-04-15--episode-01.r3d
│   └── ...
```

### 2.4 硬件（可选）

| 设备 | 用途 |
|:---|:---|
| iPhone 15 Pro（带 LiDAR） | 60fps RGB-D 采集，ARKit VIO 提供 `T_world_cam` |
| SO-ARM 101 | 5-DoF + 夹爪真机回放目标（04 阶段） |

无硬件也可跑：01/02/03 + 05（仿真）完整闭环。

---

## 3. 快速上手（一条完整链路）

以下命令假定工作目录为仓库根 `hand-6dof-pipeline/`，使用 `data/raw/*.r3d` 作为原始数据：

```bash
# --- Step 1/4  感知（cam-frame，HaMeR + MediaPipe + LiDAR） ---
python scripts/01_hand_track.py \
    --r3d-dir ./data/raw \
    --output output/01_tracking/tracking.pkl

# --- Step 2/4  处理（world-frame + 质量门控 + 滤波） ---
python scripts/02_process.py \
    --input output/01_tracking/tracking.pkl \
    --output output/02_processed/processed.pkl

# --- Step 3/4  打包 LeRobot v3（可选 sim-check 烟测） ---
python scripts/03_build_dataset.py \
    --processed output/02_processed/processed.pkl \
    --r3d-dir ./data/raw \
    --output-dir output/03_dataset_v3 \
    --repo-id <user>/demo_v3 \
    --task "pick up cup" \
    --hand right \
    --sim-check --sim-check-episodes 3 --sim-check-scale 0.5

# --- Step 4a  真机回放（SO-ARM 101） ---
python scripts/04_replay_on_arm.py \
    --robot so101 --port COM5 \
    --dataset-root output/03_dataset_v3 \
    --episode 0 --speed 0.3 --scale 0.5 --flip-lateral

# --- Step 4b  仿真回放（MuJoCo） ---
python scripts/05_replay_in_sim.py \
    --dataset-root output/03_dataset_v3 \
    --episode 0 --scale 0.5
```

`--sim-check` 在 03 打包结束后自动调起 05 做前 N 集 headless 烟测——只要 IK + MJCF 能解出姿态、无数值爆炸，就通过；不通过会在 stderr 给出具体 episode / frame。

---

## 4. 仓库布局

```
hand-6dof-pipeline/
├── README.md                     本文件
├── LICENSE_THIRD_PARTY.md        第三方资产来源 & license 汇总
│
├── docs/                         技术文档（主线）
│   ├── HaMeR_Pipeline_Guide.md       01→04 全栈
│   ├── Sim_Replay_Guide.md           05 + sim-check
│   └── keywords.md                   术语速查
│
├── assets/                       模型权重 / URDF / MJCF
│   ├── so101_new_calib.urdf          IK 用
│   ├── mujoco/trs_so101/             MJCF + scene.xml（05 用，同源于 URDF）
│   ├── hamer/                        HaMeR 权重（需单独下载）
│   └── mediapipe/                    hand_landmarker.task
│
├── scripts/                      可执行入口
│   ├── 01_hand_track.py              Step 1/4 感知
│   ├── 02_process.py                 Step 2/4 处理
│   ├── 03_build_dataset.py           Step 3/4 打包（± sim-check）
│   ├── 04_replay_on_arm.py           Step 4/4 真机回放
│   ├── 05_replay_in_sim.py           仿真回放（验证 04 解）
│   ├── _download_hamer.py            HaMeR 权重下载
│   └── _verify_hamer_env.py          环境自检
│
├── robots/                       机械臂驱动层
│   ├── base.py                       RobotArm 抽象接口
│   └── so101.py                      SO-ARM 101（LeRobot SO100Follower 封装）
│
├── sim/                          仿真层
│   ├── mujoco_loader.py              MJCF 加载 + LeRobot joint-name 映射
│   └── mujoco_replay.py              ctrl/qpos 双写 + viewer 循环
│
├── utils/                        共享工具
│   ├── r3d_reader.py                 .r3d 流式读取 + ARKit 位姿
│   ├── hand_tracker/                 HandDetectorBase + HaMeR + MediaPipe
│   ├── depth_correction.py           LiDAR 透视修正
│   ├── interpolation.py              NaN 补帧 + Slerp
│   ├── one_euro_filter.py            One-Euro 时序滤波（Pos + Slerp-Rot）
│   ├── spatial_tracker.py            MediaPipe 左右手翻转纠正
│   ├── pose_util.py                  pose ↔ mat ↔ rot6d
│   └── safe_home.py                  断连前回零安全上下文
│
├── tests/                        pytest 主线
│   ├── test_hand_track.py / test_process.py / test_build_dataset.py
│   ├── test_replay_on_arm.py（含 IK + compute_T_arm_world + wrist_roll）
│   ├── test_sim_replay.py
│   └── test_r3d_reader / test_interpolation / test_one_euro_filter /
│       test_depth_correction / test_pose_util / test_hand_tracker /
│       test_spatial_tracker
│
└── output/                       运行产出（内容已 gitignore，目录保留）
    ├── 01_tracking/                  01 输出 pkl
    ├── 02_processed/                 02 输出 pkl
    ├── 03_dataset_v3/                03 输出 LeRobot v3
    └── replay_start.json             04 使用的 IK-友好初始姿态
```

---

## 5. 关键设计决策（速览）

### 5.1 `wrist_flex` 作为 dataset 锚点（不是 gripper_frame）

Dataset 的 `eef_pos` 对齐 HaMeR 手腕关键点（= 机械臂 **wrist_flex** 关节），不是 gripper_frame（再向外 ~10cm 的工具末端）。

- 02 用首 N 帧 wrist 中位数做 robust anchor（抗首帧抖动）
- 04 IK 跑 wrist sub-chain（截断到 wrist_flex），world 原点 = replay_start 时的 wrist_flex 位置
- `auto_placement_from_home` 用 FK 自动算 `(distance, table_height)`，无需额外标定文件

### 5.2 世界系映射与 flip

ARKit 世界系：+Y 重力朝上。SO-ARM 101 base：+X 前、+Y 左、+Z 上。

变换链：`R_arm_world = Rz(rotate_deg) @ Rx(90°) @ M_flip`

- `--flip`：对称 world X（右手采集 → 左手工作空间复用）
- `--flip-lateral`：对称 world Z（采集方向 / 部署方向左右相反时用，`det(R) = -1`）

### 5.3 wrist_roll anchor（Option A）

HaMeR 全局方向不完美对齐机械臂，轴 5 绝对角易漂移。04 只锚首帧：

```
delta = replay_start.wrist_roll − extracted_wrist_roll[0]
wrist_roll_t += delta   # for all t
```

保持帧间相对旋转，首帧贴齐 replay_start，后续自然演化。

### 5.4 replay_start vs SafeHome

- **SafeHome**：连接时记录"上电安全姿态"，退出自动回到（关节紧凑、重心低）
- **replay_start**（外部 JSON）：独立"IK-友好展开姿态"，每次 episode 开始前从 safe → replay_start

两者解耦后，安全姿态可独立于 IK 起点。

### 5.5 LiDAR 深度透视修正

HaMeR 单目估计 z 精度差（焦距虚拟值 5000）。iPhone LiDAR 提供真实深度：

```
scale = z_lidar / z_hamer
corrected = [x_hamer * scale, y_hamer * scale, z_lidar]
```

所有 3D 数据（wrist + 21 joints）同步缩放，`eef_rot` 不变（射线方向本身准）。

### 5.6 same-source URDF + MJCF

`so101_new_calib.urdf` 和 `assets/mujoco/trs_so101/so101_new_calib.xml` 都是由同一份 Onshape CAD 经 onshape-to-robot 同时生成，**零点、轴向、关节名完全一致**。所以 04（ikpy + URDF）解出的 joint 角可以直接喂给 05（MuJoCo + MJCF）的 `mjData.ctrl`，不需要 per-joint offset。

---

## 6. 输出数据集结构（LeRobot v3）

| Feature | dtype | shape | 说明 |
|:---|:---|:---|:---|
| `observation.images.rgb` | video | (480, 640, 3) | MP4 (AV1) |
| `observation.images.depth` | video | (480, 640, 3) | 可选，LiDAR uint8 cm |
| `observation.state` | float32 | (7,) | `[x, y, z, rx, ry, rz, gripper]`（episode-local world） |
| `action` | float32 | (7,) | `state[t+1]`，绝对位姿 |

Per-episode 元数据附加 `center_offset_world`（episode 原点在 ARKit 绝对世界系的位置），下游可恢复全局坐标。

---

## 7. 测试

```bash
# 主线测试（legacy 默认已在 conftest 排除）
pytest tests/ -v

# 单模块
pytest tests/test_replay_on_arm.py -v
pytest tests/test_sim_replay.py -v
```

---

## 8. Limitations

诚实说明当前阶段的边界。这些不是 bug，是 scope 选择——后续会逐项推进。

### 数据 / 规模
- **单视角**：只测过 iPhone LiDAR 俯拍桌面场景；多视角融合未实现
- **小样本**：目前验证集 ~835 帧（3 episodes），尚未发布公开数据集（计划 W6+ Orbbec 头戴采集满 5h 后发 v1）
- **硬件依赖**：depth back-projection 路径依赖 iPhone LiDAR；无深度传感器的设备只能跑 HaMeR 虚拟深度（精度显著下降，见 §5.5）

### 方法 / 精度
- **动作复现 ≠ 动作推理**：当前 pipeline 把人手轨迹映射并回放到机械臂，**不是学习 policy**。下游策略训练（DP / ACT）只跑过教学级验证，非 production
- **wrist_roll Option A 简化**：首帧锚定 + 保留相对旋转，未实现完整 SO(3) orientation anchor。对大方向跟踪够用，精细操作（threading / insertion）可能不够
- **retarget 仅测过 SO-ARM 101**：其他形态机器人（7-DoF / 双臂 / 灵巧手）需重新适配 IK chain 和 wrist 对应

### 许可证
- **MANO 非商用**：HaMeR 依赖 MANO 模型，MANO 官方 license 仅限学术 / 非商用。商用需联系 MPI 另行授权，详见 [LICENSE_THIRD_PARTY.md](LICENSE_THIRD_PARTY.md)

### 工程 / 平台
- **Windows multiprocessing 未解**：L2 训练在 Windows 下需 `if __name__ == '__main__'` guard，当前脚本未完整适配，推荐在 Linux / WSL 跑训练
- **work-in-progress**：单人开发，API 可能迭代。Breaking change 会在 CHANGELOG 说明，但不保证向后兼容（Phase 0）

### 不做什么
- 不追求 SOTA 精度，追求**可复现 + 可扩展**
- 不造轮子，依赖成熟组件（HaMeR / MediaPipe / LeRobot / ikpy）
- 不做 GUI 工具链，一切 CLI + 文档

欢迎 issue 拍砖 · 具体场景不确定能不能用，开 Discussion 一起看。

---

## 9. License

- 本仓库代码：[MIT](LICENSE)
- 第三方资产（URDF / MJCF / HaMeR / MANO / MediaPipe）：见 [LICENSE_THIRD_PARTY.md](LICENSE_THIRD_PARTY.md)
