# opc_data_pipeline

**人类裸手操作数据采集与验证**——iPhone Pro RGB-D + 完整人手运动（MANO 21 关节 + 6DoF wrist）→ 多本体 retargeting → LeRobot v3 标准化打包 → 仿真 / 真臂回放闭环验证。

OPC 为中腰部机器人公司提供具身智能策略（DP / ACT / VLA 等）训练用的人类裸手操作演示数据。本仓库是采集 → 处理 → 打包 → retarget → sim/real 回放的端到端离线工具链。

---

## 1. 录制与执行的两层结构

**录制层**（`scripts/`）按设备分线，stages 01-03，输出 raw `.processed.npz` + 交付用 LeRobot v3。
**执行层**（`retarget/` `replay/`）录制无关，按"目标机器人 × 仿真/真机"分派，stages 05-06。

```
[录制层 — iPhone Pro]                    [执行层 — recording-agnostic]

(iPhone) .r3d → 01 → 02 → 03_source.lerobot
                  ↓
               .processed.npz                 retarget/                  replay/
                  └──────→ ──────→  (per-robot+env)  ──→  (per-robot+env)
                                    so101 / shadow / leap     mujoco / rerun / real
                                    python -m retarget         python -m replay
                                    → .qpos.npz                → viewer / .mp4 / 真臂
```

### 1.1 录制层（iPhone Pro）

| 步骤 | 输入 / 输出 | 说明 |
|:---|:---|:---|
| 01 | `.r3d` → `01_hand_track.py` → `01_tracking/<sid>/<sid>.tracking.npz` | HaMeR cam-frame 跟踪（双手 21 关节 + 6DoF） |
| 02 | `02_process.py` → `02_processed/<sid>/<sid>.processed.npz` | trim + quality 门控（**不改信号值**） |
| 03 | `03_build_source.py` → `03_source/` | LeRobot v3 数据集（**交付物**） |

**raw-first 原则**：02_process 只做 trim + quality 门控，**不修改任何信号值**（不填 NaN、不滤波、不平滑）。所有可选处理推到下游。这与 DexYCB / HumanPlus / DROID 等业界标杆数据集的发布范式一致。

### 1.2 执行层（recording-agnostic）

| 步骤 | 输入 | 输出 | 说明 |
|:---|:---|:---|:---|
| 05 retarget | `.processed.npz`（schema 校验决定，不绑文件名前缀） | `output/<line>/<batch>/05_qpos_<robot>/<sid>/<sid>.qpos.npz` + `.qpos.meta.json` | `python -m retarget --robot <r> --hand <h>`；conda env 视后端而定（dex 后端要 `opc-dex`） |
| 06 replay | `.qpos.npz` + meta | viewer 窗口 / `06_replay_<robot>/<sid>/<sid>.replay.mp4` / 真臂 | `python -m replay --qpos-root ... --output viewer\|mp4\|real`；后端从 meta 读 `(robot, env)` 自动派发 |

**执行层零代码改动地新增能力**：retarget 通过 schema 校验（不绑定文件名前缀）自动接受合规 `.npz` 输入；NaN 太多会失败并提示先做平滑。replay 只灌 qpos 给 MuJoCo / 真臂。客户交付的 per-embodiment LeRobot v3（含 state/action）放 `0X_build_<robot>`（task #22 待开发）。

详细技术文档：
- **[docs/keywords.md](docs/keywords.md)** — 术语速查
- **[docs/HaMeR_Pipeline_Guide.md](docs/HaMeR_Pipeline_Guide.md)** — iPhone 路线技术细节
- **[docs/Sim_Replay_Guide.md](docs/Sim_Replay_Guide.md)** — sim 集成

---

**iPhone-line 注释**：iPhone 多一列 ARKit `T_world_cam` per-frame 外参（透传到 03_source 的 LeRobot feature 列），下游可重建任意坐标系。

**SO-101 后端**（`retarget/so101.py` + `replay/sim/mujoco_so101.py` + `replay/real/so101.py`）已就绪。`python -m retarget --robot so101 --hand right` 跑 IK + wrist_roll 反解 + thumb-index 夹爪映射，输出 `.qpos.npz` (radians)；`python -m replay --output {viewer,mp4,real}` 派发到 sim viewer / 离屏 mp4 / SO-101 真臂三选一。**dex hands 后端**（Shadow / Leap / Allegro 等）`python -m retarget --robot shadow` + `python -m replay` 走 mujoco_dex / rerun_dex。

---

## 2. 环境

| Conda env | 用途 | 关键依赖 |
|:---|:---|:---|
| `lerobot` | 主环境（01-03 + so101 retarget + sim 回放） | `numpy`, `lerobot`, `mujoco`, `mink`, HaMeR 全家桶 |
| `opc-dex` | dex hands retarget 步骤（独立） | `dex_retargeting`, `pinocchio`（与 `lerobot` 的 numpy ABI 冲突，故隔离） |

```bash
conda activate lerobot
pip install -e .   # （仓库还未 packaging，按需安装）
```

### 2.1 资产下载

```bash
# HaMeR 权重（~3.4GB）
python scripts/_download_hamer.py
python scripts/_verify_hamer_env.py

# MuJoCo Menagerie（Shadow Hand 等 MJCF；submodule）
git submodule update --init --recursive

# dex_retargeting 配套 URDF（dex hands 后端必需，~80MB）
git clone https://github.com/dexsuite/dex-urdf.git assets/dex-urdf
```

---

## 3. 快速上手（iPhone Pro）

```bash
conda activate lerobot
cd code/opc_data_pipeline

# Step 1: HaMeR 跟踪（默认输入 output/iphone/<batch>/00_record/，输出 01_tracking/）
python scripts/01_hand_track.py --r3d-dir output/iphone/<batch>/00_record/

# Step 2: trim + quality
python scripts/02_process.py \
    --track output/iphone/<batch>/01_tracking/<sid>/<sid>.tracking.npz

# Step 3: 打 LeRobot v3 数据集（交付物）
python scripts/03_build_source.py \
    --repo-id opc/iphone_source_v1 --task pour_coffee_bean
```

**SO-arm 101 路径**（单一 `lerobot` env 即可）：

```bash
conda activate lerobot

# Step 5: retarget
python -m retarget --robot so101 --hand right \
    --source-root output/iphone/<batch>/02_processed
# 输出 → output/iphone/<batch>/05_qpos_so101/<sid>/<sid>.qpos.npz

# Step 6a: sim 预览 (mp4 / viewer / rerun)
python -m replay --qpos-root output/iphone/<batch>/05_qpos_so101 --output mp4
python -m replay --qpos-root output/iphone/<batch>/05_qpos_so101            # viewer
python -m replay --qpos-root output/iphone/<batch>/05_qpos_so101 --output rerun

# Step 6b: 真臂回放
python -m replay --qpos-root output/iphone/<batch>/05_qpos_so101 \
    --output real --port COM5 --speed 0.3
# 加 --dry-run 预跑（不连串口）
```

**Dex hands 路径**（Shadow Hand / Leap / Allegro 等）：

```bash
# Step 5: retarget — opc-dex env (numpy>=2 + dex_retargeting)
conda activate opc-dex
python -m retarget --robot shadow --hand right \
    --source-root output/iphone/<batch>/02_processed
# 输出 → output/iphone/<batch>/05_qpos_shadow/<sid>/<sid>.qpos.npz

# Step 6: replay — lerobot env (mujoco)
conda activate lerobot
python -m replay --qpos-root output/iphone/<batch>/05_qpos_shadow
```

---

## 4. 仓库布局

```
opc_data_pipeline/
├── README.md                    本文件
├── LICENSE / LICENSE_THIRD_PARTY.md
│
├── docs/                        技术文档
│   ├── keywords.md                  术语速查
│   ├── HaMeR_Pipeline_Guide.md      iPhone 路线技术细节
│   └── Sim_Replay_Guide.md          sim 集成
│
├── assets/                      模型权重 / URDF / MJCF
│   ├── hamer/                       HaMeR 权重（gitignored，需下载）
│   ├── mediapipe/                   MediaPipe hand_landmarker.task
│   ├── menagerie/                   MuJoCo Menagerie（submodule，Shadow Hand 等 MJCF）
│   ├── dex-urdf/                    dex_retargeting 配套 URDF（gitignored，需 clone）
│   ├── mujoco/trs_so101/            SO-ARM 101 MJCF
│   └── so101_new_calib.urdf         SO-ARM 101 URDF
│
├── scripts/                     iPhone-line 录制层 (stages 01-03)
│   ├── 01_hand_track.py / 02_process.py / 03_build_source.py
│   ├── _schema.py / _download_hamer.py / _verify_hamer_env.py
│   └── legacy/
│
├── utils/                       共用工具
│   ├── common/                      pose_util.py / geometry.py（数学）
│   ├── hand_tracker/                HaMeR + MediaPipe + depth_correction
│   │                                + spatial_tracker + io + overlay
│   ├── iphone/                      r3d_reader.py（.r3d 解码）
│   ├── process/                     core.py — trim + quality + 旋转诊断
│   │                                world_frame.py — ARKit 外参 → world frame
│   └── dataset/                     core.py + iphone_writer
│                                    LeRobot v3 写入封装
│
├── retarget/                    执行层 stage 5 — recording-agnostic
│   ├── __init__.py                  注册表 + (robot, env) 派发
│   ├── __main__.py                  CLI: python -m retarget
│   ├── loader.py                    任意 .npz schema 校验 + 加载
│   ├── dex_hands.py                 dex_retargeting 后端 (shadow/leap/...)
│   └── so101.py                     SO-arm 101 后端（mink IK + pinch midpoint + 夹爪）
│
├── replay/                      执行层 stage 6 — recording-agnostic
│   ├── __init__.py                  注册表 + (robot, env) 派发
│   ├── __main__.py                  CLI: python -m replay
│   ├── sim/
│   │   ├── mujoco_loader.py         SO-101 MJCF 加载
│   │   ├── mujoco_mesh.py           可视 mesh 提取（rerun 共用）
│   │   ├── mujoco_dex.py            dex hand sim 后端（mp4 / viewer）
│   │   ├── mujoco_so101.py          SO-101 sim 后端（mp4 / viewer）
│   │   ├── rerun_dex.py             dex hand rerun.io AR-overlay 后端
│   │   ├── rerun_so101.py           SO-101 rerun.io AR-overlay 后端
│   │   └── scenes/                  生成的 MJCF（gitignored）
│   └── real/
│       └── so101.py                 SO-101 真臂后端（含驱动 + SafeHome）
│
├── tests/                       pytest
│
└── output/                      运行产出（gitignored，目录保留）
    └── iphone/<batch>/
        ├── 00_record/ 01_tracking/ 02_processed/ 03_source/
        ├── 05_qpos_<robot>/<sid>/    qpos.npz + qpos.meta.json
        └── 06_replay_<robot>/<sid>/  replay.mp4
```

**Batch-major 布局**：每个 capture campaign（如 `pour_coffee_bean_4_30`）一个根目录，下面有完整 stages 产物。打包/归档/删除/对比都以 batch 为单元。脚本 `--batch <name>` 可显式指定，多数情况能从输入路径自动派生。

**Stage 编号**：01-03 是录制层（per-line），05-06 是执行层（per-robot+env，公共）。`05_qpos_<robot>` / `06_replay_<robot>` 结构一致；其中 robot 来自 retarget CLI 的 `--robot`。客户交付的 per-embodiment LeRobot v3（带 retarget 输出 + state/action）将作为 `0X_build_<robot>` 在录制层并存（task #22 待开发）。

---

## 5. 输出数据集结构（LeRobot v3）

### 5.1 Source 数据集（`iphone/<batch>/03_source/`）—— 交付物

| Feature | dtype | shape | 说明 |
|:---|:---|:---|:---|
| `observation.images.rgb` | video | (480, 640, 3) | MP4 (AV1) |
| `observation.depth` | uint16 | (480, 640) | **无损 mm 量纲**（Array2D，0..65535 mm）。可选 `--depth-encoding uint8_cm` 降级（120× 节省存储，截断 2.55 m） |
| `observation.wrist_pose_left/right` | float32 | (7,) | xyz + scipy xyzw quat，cam frame |
| `observation.mano_joints_left/right` | float32 | (21, 3) | MANO 21 关键点 cam frame |
| `observation.left/right_valid` | float32 | (1,) | 1.0 = 该帧有效（trim 内 + quality 过 + finite） |
| `observation.left/right_confidence` | float32 | (1,) | HaMeR 检测器原始置信度 |
| `observation.T_world_cam` | float32 | (4, 4) | ARKit per-frame 外参，下游可重建任意坐标系 |

**没有 `observation.state` / `action`**——那是下游 per-embodiment 数据集的工作。Source 数据集只发"裸手观测"，机器人侧动作交给客户 / 下游脚本组装。

### 5.2 Per-embodiment 客户交付数据集（`<line>/<batch>/0X_build_<robot>/`，task #22 待开发）

下一轮要重建的客户交付层。预期 schema（参考 LeRobot 训练 SDK）：

| Feature | dtype | shape | 说明 |
|:---|:---|:---|:---|
| `observation.images.rgb` | video | (H, W, 3) | passthrough |
| `observation.depth` | uint16 | (H, W) | passthrough |
| `observation.state` | float32 | (N,) | embodiment-specific（SO-101 = 7：xyz + rx ry rz + gripper） |
| `action` | float32 | (N,) | 同上；`action[t] = state[t+1]` |
| `observation.qpos` | float32 | (N_qpos,) | retarget 输出 qpos（来自 `05_qpos_<robot>`） |

数据流：`03_source` → 平滑/插值/gripper 归一化 → 拼 state/action → 拉 `05_qpos_<robot>` 的 qpos 列 → LeRobot v3 写盘。

### 5.3 中间产物：`05_qpos_<robot>/`（执行层，非交付）

`python -m retarget` 输出。**内部 QA 链路用**，不交付客户：

```
<sid>.qpos.npz:
  timestamps_us  (T,)    int64
  <hand>_qpos    (T, N)  float32   # NaN 在无效帧
  <hand>_qpos_valid (T,) bool

<sid>.qpos.meta.json:
  robot, env, hand, joint_names, n_joints,
  trim, n_frames_total / _in_trim / _retarget_succeeded,
  K_flat (intrinsics), backend, schema_version, run_timestamp_iso
```

---

## 6. 测试

```bash
pytest tests/ -v
```

---

## 7. License

- 仓库代码：[MIT](LICENSE)
- 第三方资产（URDF / MJCF / HaMeR / MANO / MediaPipe）：见 [LICENSE_THIRD_PARTY.md](LICENSE_THIRD_PARTY.md)
- **MANO 非商用**：HaMeR 依赖 MANO，MANO license 仅限学术 / 非商用，商用需联系 MPI 另行授权
