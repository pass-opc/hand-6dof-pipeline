# HaMeR 裸手追踪 Pipeline 技术文档

## 1. HaMeR 是什么

HaMeR (Hand Mesh Recovery) 是 CMU Pavlakos 团队 2024 CVPR 的工作，从单张 RGB 图像重建手部 3D mesh 和关节位姿。

**核心能力**：输入一张照片 → 输出手腕 6DoF 位姿 + 21 个关节 3D 坐标 + 778 顶点 mesh。

**技术栈**：
- **检测**：ViTDet (Vision Transformer + Mask R-CNN, detectron2 框架) — 找到图中手的 bounding box
- **重建**：ViT-H backbone + Transformer decoder → MANO 参数回归
- **手部模型**：MANO (Max Planck 的参数化手部模型) — 用少量参数表示任意手型和手势

**论文**：Pavlakos et al., "Reconstructing Hands in 3D with Transformers", CVPR 2024
**代码**：https://github.com/geopavlakos/hamer

---

## 2. 单帧推理全流程

```
.r3d 文件 (ZIP 包)
    │  iter_r3d_frames() 逐帧解包
    ▼
RGB 图片 (H, W, 3) uint8
    │  HandDetectorBase.detect_hands(rgb) → list[HandBox]
    │  当前实现: MediaPipe HandLandmarker (tight bbox from 21 landmarks)
    │  未来实现: ViTPose (mmpose, from ViTDet person crops)
    ▼
手部 bounding box [x1, y1, x2, y2] + handedness + 置信度
    │  ViTDetDataset: 扩大2倍裁剪 → 256×256 → ImageNet归一化
    │  左手会水平翻转成右手（MANO只建模右手）
    ▼
手部 crop tensor (3, 256, 256)
    │  HaMeR model.forward_step(batch, train=False)
    ▼
MANO 参数:
  - global_orient (1, 3, 3): 手腕旋转矩阵
  - hand_pose (15, 3, 3): 15个关节旋转
  - betas (10,): 手型参数
  - pred_cam (3,): weak perspective params [s, tx, ty] (crop space)
    │  cam_crop_to_full(pred_cam, box_center, box_size, img_size)
    ▼
  - pred_cam_t_full (3,): 手腕在全图相机坐标系的位置 [x, y, z]
    │  MANO forward kinematics
    ▼
21 个关节 3D 坐标 (21, 3) + 778 mesh 顶点 (778, 3)
    │  提取 6DoF + gripper 信号
    ▼
HandDetection:
  - wrist_pos (3,)       ← pred_cam_t_full
  - wrist_rot (3,)       ← Rotation.from_matrix(global_orient).as_rotvec()
  - joints_3d (21, 3)    ← MANO joints + wrist_pos offset
  - gripper_width         ← ||joint[4] - joint[8]||  拇指尖-食指尖距离
  - confidence            ← 检测器分数
  - handedness            ← "left" / "right"
```

### 2.1 手部检测层 (HandDetectorBase)

本 pipeline 将手部检测与 3D 重建解耦为两个独立接口：

```python
# utils/hand_tracker/base.py
class HandDetectorBase(ABC):
    def detect_hands(self, rgb) -> list[HandBox]: ...
    def get_detector_name(self) -> str: ...

class HandBox:
    bbox: np.ndarray  # (4,) [x1, y1, x2, y2]
    is_right: bool
    confidence: float
```

**当前检测器**: MediaPipe HandLandmarker
- 21 landmark → tight bbox，自带 handedness
- 适合俯拍桌面（手为主体、无完整人体）

**HaMeR 原版 3 步流程**: ViTDet (person detection) → ViTPose (hand keypoints → bbox) → HaMeR
- ViTDet 是 COCO person detector (class 0)，俯拍只看到手时检测率 0%
- ViTPose 需要 mmpose + mmcv（OpenMMLab 生态），尚未安装
- 后续实现 ViTPose 检测器后，通过 `create_detector("vitpose")` 即可切换，下游代码不变

### 2.2 MANO 模型

MANO 是手部的参数化模型，类似 SMPL 之于人体：

```
输入参数:
  pose (45D)     = global_orient(3D) + hand_pose(15×3D)  → 控制手势
  shape (10D)    = betas                                 → 控制手型（胖瘦大小）

输出:
  joints (21, 3) = 21 个关节的 3D 坐标
  vertices (778, 3) = mesh 顶点
```

MANO 只建模了**右手**。左手通过水平翻转 RGB → 当作右手推理 → 翻转结果回来。

### 2.3 MANO 21 关节定义

```
         0 (wrist) ← wrist_pos 的来源
        /  |  \
       1   5   9   13  17      ← 各手指根部 (MCP)
       |   |   |   |   |
       2   6   10  14  18      ← 近端指间关节 (PIP)
       |   |   |   |   |
       3   7   11  15  19      ← 远端指间关节 (DIP)
       |   |   |   |   |
       4   8   12  16  20      ← 指尖 (tip)
     thumb index mid ring pinky
       ↑   ↑
   gripper_width = ||joint[4] - joint[8]||
```

### 2.4 cam_crop_to_full 坐标变换

HaMeR 模型输出 `pred_cam` 是 crop 空间的 weak perspective 参数 `[s, tx, ty]`。需要用 `cam_crop_to_full()` 转换到全图相机坐标系：

```python
# hamer.utils.renderer.cam_crop_to_full
bs = box_size * s + 1e-9
tz = 2 * focal_length / bs
tx = (2 * (cx - w/2) / bs) + cam_tx
ty = (2 * (cy - h/2) / bs) + cam_ty
# 输出: (tx, ty, tz) — 全图相机坐标系下的 3D 位置
```

注意：`focal_length=5000` 是虚拟值，z 方向不是真实米制深度。需要 LiDAR 深度修正才能获得真实尺度。

---

## 3. Pipeline 架构（4 层解耦）

主线由 4 个脚本组成，每一层有独立契约，中间产物是 pickle / LeRobot v3 数据集。

```
[01] 感知 (01_hand_track.py)
   .r3d → cam-frame tracking.pkl
   coord_frame = "camera"
        │
        ▼
[02] 处理 (02_process.py)
   cam-frame → trim → world transform → robust anchor →
   mark bad → interpolate → One-Euro → gripper 归一化 → 7D state/action
   coord_frame = "episode_local"
        │
        ▼
[03] 打包 (03_build_dataset.py)
   processed.pkl + .r3d RGB/Depth → LeRobot v3 parquet + MP4
        │
        ▼
[04] 回放 (04_replay_on_arm.py)
   LeRobot v3 → retarget → IK (wrist sub-chain) → wrist_roll anchor → SafeHome
```

**分层原则**：上游各自实现（HaMeR/WiLoR/ArUco），下游（02/03/04）统一消费 `episode_local` world-frame 契约，不跨上游强行抽象。

### 3.1 抽象层设计（感知层）

所有手部追踪后端（HaMeR、WiLoR、未来新算法）实现同一个接口：

```python
# utils/hand_tracker/base.py
class HandTracker(ABC):
    @abstractmethod
    def detect(self, rgb: np.ndarray) -> list[HandDetection]: ...
    @abstractmethod
    def get_backend_name(self) -> str: ...

@dataclass
class HandDetection:
    handedness: str       # "left" or "right"
    wrist_pos: np.ndarray # (3,) 相机坐标系
    wrist_rot: np.ndarray # (3,) axis-angle
    joints_3d: np.ndarray # (21, 3) MANO joints
    confidence: float

    @property
    def gripper_width(self) -> float:
        return float(np.linalg.norm(self.joints_3d[4] - self.joints_3d[8]))
```

工厂创建 + lazy import：只有真正创建 HaMeR tracker 时才 import torch/detectron2。

```python
tracker = create_tracker("hamer", detector="mediapipe")
```

### 3.2 感知层流程（01_hand_track.py）

**纯感知，不做时序处理**。输出 cam-frame 原始几何 + ARKit 位姿透传给 02：

```
每一帧:
  RGB + LiDAR depth + T_world_cam[t]  ← iter_r3d_frames() + read_poses()
       ↓
  tracker.detect(rgb) → list[HandDetection]  (0~2 手，camera frame)
       ↓
  SpatialHandTracker 纠正 MediaPipe 左右手翻转
       ↓
  LiDAR 深度修正：按 bbox 中心采 real K 深度，透视缩放 wrist + joints
       ↓
  存入 left_hand / right_hand 数组（cam frame, metric scale）
```

**不在 01 做的事（移到 02）**：world transform、trim、NaN 补帧、时序滤波、质量门控。

### 3.3 处理层（02_process.py）

处理层把 cam-frame 原始追踪转成 **episode-local world-frame 7D state/action**。双手独立处理（各自的 trim 窗口和 anchor），逐手 10 步流水线：

```
[1] trim leading/trailing NaN       → trim_slice into r3d frame range
[2] world transform                 → p_world = R_wc @ p_cam + t_wc
                                       R_hand_world = R_wc @ R_hand_cam
[3] center to robust anchor         → 减去 first-N valid wrists 的中位数
                                       （抗首帧抖动；结果 = episode-local）
[4] mark bad frames                 → 位置跳变 > max_pos_jump_m → NaN
[5] quality check                   → detection rate / max gap / duration
[6] fill interior NaN               → 位置线性 + 旋转 Slerp
[7] One-Euro filter                 → Pos (Vector) + Rot (slerp-OneEuro)
[8] rotation jump warning
[9] gripper normalize               → (gw - ep_min) / (ep_max - ep_min) ∈ [0,1]
[10] build state/action             → 7D 绝对，action[t] = state[t+1]
```

**Anchor 的作用**：把 world frame 原点从 ARKit 绝对 0 搬到"本 episode 的首 N 帧 wrist 中位数"。下游 04 回放层的 world 原点同时对齐这个锚点（= replay_start 时 wrist_flex 的位置），实现"人类手腕 ↔ 机器人手腕"物理对应。

### 3.4 打包层（03_build_dataset.py）

**纯格式写入**，不做任何运动学处理：

```
[1] 校验 input contract (coord_frame == 'episode_local')
[2] 按 --hand 选单手
[3] 跳过 02 标记 quality_passed=False 的 hand
[4] 按 trim_slice 流式读 r3d → resize → LeRobotDataset.add_frame
[5] 把 center_offset_world 合并进 meta/episodes/*.parquet
```

LeRobot v3 字段：

| 字段 | Shape | 用途 |
|:-----|:------|:-----|
| `observation.images.rgb` | video (480,640,3) | MP4 (AV1) |
| `observation.depth` | uint16 (480,640) | 可选，LiDAR raw mm |
| `observation.state` | (7,) | `[x, y, z, rx, ry, rz, gripper]` (episode-local) |
| `action` | (7,) | `state[t+1]` |

Per-episode 附加 `center_offset_world`（ARKit 绝对世界系下的 episode 原点），下游需要时可复原绝对坐标。

### 3.5 回放层（04_replay_on_arm.py）

把 episode-local world-frame EEF 轨迹跑到 SO-arm 101 实机上。核心步骤：

```
[1] 加载 episode actions + replay_start.json
[2] auto_placement_from_home: 用 wrist sub-chain FK 算 T_arm_world 的
    (distance, table_height)；world 原点 = replay_start wrist_flex 位置
[3] compute_T_arm_world: R = Rz(rotate) @ Rx(90°) @ M_flip
    - flip        → 对称 world X（右↔左手空间复用）
    - flip_lateral → 对称 world Z（采集和部署左右相反时用）
[4] retarget: p_arm = R @ (scale * p_world) + t_arm
[5] IK (wrist sub-chain, 截断到 wrist_flex) → joints 1-4
[6] wrist_roll 从 orientation 用 FK 残差反解；首帧 anchor 到 replay_start
    delta = replay_start.wrist_roll - extracted_wrist_roll[0]
    wrist_roll += delta
[7] smooth_joint_trajectory: 跳变 clamp + 滑动平均
[8] workspace_check: IK err / 跳变阈值校验
[9] safe_home → replay_start → episode start → replay
    SafeHome 退出时自动回到上电安全姿态
```

**wrist_flex 锚点原则**：dataset `eef_pos` 对齐人类手腕关键点，不是夹爪末端。IK 跑 wrist sub-chain（`chain.links[:wrist_flex_idx+1]`），target 落到 wrist_flex，保证"wrist ↔ wrist"对应。若跑全链会落到 gripper_frame，多 10cm 偏移。

**safe_home vs replay_start**：两者分工
- `SafeHome`（上下文管理器）：记录连接时的安全姿态（上电重心低、关节紧凑）
- `replay_start`（外部 JSON）：IK 友好的展开姿态（关节远离极限，利于 IK 收敛）
- 04 执行流程：safe → replay_start → episode → （退出时）回 safe

---

## 4. Gripper 信号：拇指-食指距离

### 4.1 原理

裸手没有物理夹爪，用拇指尖 (joint 4) 到食指尖 (joint 8) 的 3D 欧氏距离模拟 gripper 开合：

```python
gripper_width = np.linalg.norm(joints_3d[4] - joints_3d[8])
# 捏合 → ~0.01m → gripper 关闭
# 张开 → ~0.08m → gripper 打开
```

### 4.2 行业参考

这是行业标准做法：
- **EasyMimic** (2024)：thumb tip - index tip 距离
- **Robotic Telekinesis** (Stanford)：同上
- **AnyTeleop** (清华)：thumb-index 用于简单夹爪，全关节用于灵巧手
- **H2O** (CMU)：双手场景也用 thumb-index

### 4.3 归一化

手追踪没有标定文件（不像 ArUco 有 gripper_range.json），04 自动从当前 episode 的 min/max 归一化到 [0, 1]：

```python
if source in ("hamer", "wilor"):
    ep_min, ep_max = np.nanmin(gripper_width), np.nanmax(gripper_width)
    gripper_width = (gripper_width - ep_min) / (ep_max - ep_min)
```

---

## 5. LiDAR 深度修正

### 5.1 为什么需要

HaMeR 从单目 RGB 估计深度（z 方向），使用虚拟 focal_length=5000，精度差。iPhone LiDAR 提供真实深度（±1-2cm）。

### 5.2 修正算法（透视缩放）

```python
# 1. 投影 3D → 2D 像素坐标 (pinhole camera model)
px = fx * x / z + cx
py = fy * y / z + cy

# 2. 从 LiDAR depth map 读真实深度
z_lidar = depth_map[py, px]       # wrist_point 模式
z_lidar = median(depth_map[patch]) # patch_median 模式 (5×5, 推荐)

# 3. 透视缩放
scale = z_lidar / z_hamer
corrected = [x * scale, y * scale, z_lidar]
```

**为什么 x, y 也要乘 scale？**

pinhole 模型下，相机射线方向 (x/z, y/z) 是准的（由像素位置决定），只有射线上的深度 z 不准。修正 z 后，x 和 y 按同比例缩放才能保持在同一条射线上。

### 5.3 修正范围

开启 `--read-depth` 时，**所有 3D 数据**都会同步缩放：
- `eef_pos` ×scale
- `joints_3d` ×scale（21 个关节全部）
- `gripper_width` 从修正后的 joints 重新计算

`eef_rot` 不变（射线方向不受 scale 影响）。

### 5.4 两种模式

| 模式 | 做法 | 优缺点 |
|:-----|:-----|:-------|
| `wrist_point` | 读投影像素处单点深度 | 快，但对深度噪声敏感 |
| `patch_median` | 取 5×5 邻域中位数 | 抗 outlier，推荐使用 |

---

## 6. One-Euro 时序滤波

### 6.1 为什么需要

HaMeR 逐帧独立推理，帧间没有时序约束，输出有高频抖动 (jitter)。ArUco 基于 PnP 几何解算，抖动小，不需要滤波。

### 6.2 原理

One-Euro filter 是自适应低通滤波器，根据信号速度调整平滑强度：

```
慢动作 → 低 cutoff → 强平滑 → 去抖动
快动作 → 高 cutoff → 弱平滑 → 少延迟
```

核心公式：
```python
speed = ||dx/dt||                          # 信号速度
cutoff = min_cutoff + beta * speed          # 自适应截止频率
alpha = 2π * cutoff * dt / (2π * cutoff * dt + 1)  # 平滑因子
x_filtered = alpha * x + (1 - alpha) * x_prev      # 指数平滑
```

### 6.3 参数

| 参数 | 默认值 | 含义 |
|:-----|:-------|:-----|
| `min_cutoff` | 1.0 Hz | 静止时的平滑强度。越小越平滑 |
| `beta` | 0.007 | 速度敏感度。越大，快动作时越跟手 |
| `d_cutoff` | 1.0 Hz | 速度估计本身的平滑，一般不调 |

### 6.4 Rotation 滤波

旋转不能直接做线性平滑（axis-angle 有 2π 不连续性）。PoseOneEuroFilter 的做法：

```
axis-angle → quaternion → One-Euro 滤波 (slerp-based) → 归一化 → axis-angle
```

quaternion 空间连续，适合插值和平滑。

### 6.5 实测效果（2026-04-15，3 episodes）

| 指标 | RAW 均值 | Filtered 均值 | 改善 |
|:-----|:---------|:-------------|:-----|
| 最大速度 | 3.22 m/s | 0.93 m/s | -71% |
| 最大加速度 | 177 m/s² | 11.2 m/s² | -94% |
| 平均 Jerk | 2933 m/s³ | 166 m/s³ | -94% |
| SPARC | -6.21 | -3.55 | +43% |
| Jitter | 10.4mm | 5.2mm | -50% |

---

## 7. 质量评估体系

### 7.1 评估指标 (utils/eval_metrics.py)

融合 UMI、DIME、AnyTeleop、运动科学文献的标准：

| 类别 | 指标 | 阈值 | 来源 |
|:-----|:-----|:-----|:-----|
| 检测 | detection_rate (全 episode) | >70% | — |
| 检测 | detection_rate_active (有效区间) | >95% | DIME/ACE |
| 检测 | max_consecutive_dropout | <10 frames | DIME |
| 轨迹 | speed_max | <2 m/s | UMI/AnyTeleop |
| 轨迹 | acc_max | <20 m/s² | 运动科学 |
| 轨迹 | SPARC (spectral arc length) | >-4.0 | Balasubramanian 2015 |
| 轨迹 | jitter (displacement std) | <3mm | 手部追踪社区 |
| 轨迹 | normalized_jerk (dimensionless) | — | Flash & Hogan 1985 |
| Gripper | toggle_rate | <2 Hz | 遥操作文献 |
| 置信度 | confidence_mean | >0.5 | — |
| 手性 | handedness_consistency | >80% | — |

### 7.2 使用方式

```python
from utils.eval_metrics import evaluate_trajectory, print_evaluation_report

metrics = evaluate_trajectory(positions, timestamps, gripper_widths, confidences,
                               detection_mask=mask)
print_evaluation_report(metrics, label="Episode 01")
```

---

## 8. joints_3d 的用途

本 pipeline 当前只用 wrist 6DoF + gripper_width (7D) 训练，因为目标机器人是简单夹爪 (SO-arm 101)。joints_3d 存在 dataset 里但不参与当前训练。

| 客户机器人类型 | 用什么数据 |
|:-------------|:----------|
| 简单夹爪 | state 7D (wrist 6DoF + gripper) |
| 灵巧手 (Allegro, LEAP Hand) | joints_3d → retarget → 手指关节角 |
| 人形机器人 (宇树 H1) | 全部：wrist 6DoF + joints_3d |

**本 pipeline 的定位是数据服务**——存完整数据让下游自己按需取用，不丢信息。

---

## 9. 环境配置

### 已完成配置

HaMeR 已装入 `lerobot` conda 环境 (Python 3.10, PyTorch 2.7.1+cu128)。

关键依赖：detectron2, smplx, timm, pytorch-lightning, mmcv, mediapipe

### 资产文件

```
assets/
  hamer/_DATA/
    hamer_ckpts/checkpoints/hamer.ckpt    # HaMeR model checkpoint
    data/mano/MANO_RIGHT.pkl              # MANO right hand model
    vitdet_model/model_final_61ccd1.pkl   # ViTDet weights (future use)
  mediapipe/
    hand_landmarker.task                   # MediaPipe model (auto-downloaded)
```

### Windows 注意事项

- detectron2 需要 `--no-build-isolation` 安装
- pyrender 需要 mock（`_mock_pyrender()` in hamer_backend.py）避免 OpenGL 依赖

### 代码隔离

hamer_backend.py 用 lazy import，其他文件永远不 import hamer/torch/detectron2。

### 环境策略

**优先**: 装到 lerobot env（当前方案，已验证可用）

**备选**: 单独 hamer env（如果依赖冲突）
```bash
# 01_hand_track.py 在 hamer env 跑, 输出 pkl, 04 在 lerobot env 消费
```

---

## 11. CLI 使用

### 01_hand_track.py — 感知（cam frame）

```bash
python scripts/01_hand_track.py \
    --r3d-dir ./data/raw \
    --output output/01_tracking/tracking.pkl

# 关闭 LiDAR 深度修正
python scripts/01_hand_track.py \
    --r3d-dir ./data/raw --output tracking.pkl --no-depth
```

| 参数 | 默认值 | 说明 |
|:-----|:-------|:-----|
| `--r3d-dir` | 必填 | .r3d 文件目录 |
| `--output` | 必填 | 输出 pkl 路径 |
| `--backend` | hamer | 追踪后端 |
| `--detector` | mediapipe | 手部检测器 |
| `--no-depth` | 关闭 | 禁用 LiDAR 深度修正 |

### 02_process.py — 处理（world frame + 质量门控）

```bash
python scripts/02_process.py \
    --input output/01_tracking/tracking.pkl \
    --output output/02_processed/processed.pkl

# 禁用 One-Euro（debug 用）
python scripts/02_process.py --input ... --output ... --no-filter
```

### 03_build_dataset.py — 打包 LeRobot v3

```bash
python scripts/03_build_dataset.py \
    --processed output/02_processed/processed.pkl \
    --r3d-dir ./data/raw \
    --output-dir output/03_dataset_v3 \
    --repo-id demo/demo_v3 \
    --task "pick up cup" \
    --hand right

# 禁用 depth channel
python scripts/03_build_dataset.py ... --no-depth

# 可选：build 后自动在 MuJoCo headless 跑前 N 个 episode，
# 用 sim 侧的 joint-range WARN 作为"pre-real-arm"过滤信号
python scripts/03_build_dataset.py ... --sim-check --sim-check-episodes 3
```

### 04_replay_on_arm.py — 真机回放

```bash
# Dry-run（无机械臂）
python scripts/04_replay_on_arm.py --dry-run \
    --dataset-root output/03_dataset_v3 --episode 0

# 真机执行
python scripts/04_replay_on_arm.py \
    --robot so101 --port COM5 \
    --dataset-root output/03_dataset_v3 \
    --episode 0 --speed 0.3 --scale 0.5 --flip-lateral
```

| 参数 | 默认值 | 说明 |
|:-----|:-------|:-----|
| `--episode` | 0 | 选哪个 episode |
| `--speed` | 0.5 | 播放速度（0.3 = 30% 最安全）|
| `--scale` | 1.0 | 工作空间缩放（<1 缩小轨迹） |
| `--flip` | 关 | 对称 world X（左右手空间互换）|
| `--flip-lateral` | 关 | 对称 world Z（采集和部署左右相反时）|
| `--distance` / `--table-height` / `--rotate-deg` | auto | 覆盖自动放置 |
| `--replay-start` | output/replay_start.json | IK-友好初始姿态 JSON |

### 05_replay_in_sim.py — 仿真回放（MuJoCo）

详见 [Sim_Replay_Guide.md](Sim_Replay_Guide.md)。常用命令：

```bash
# 人工预览（GUI，跑完保持最后一帧）
python scripts/05_replay_in_sim.py --dataset-root output/03_dataset_v3 \
    --episode 2 --scale 0.5

# 批量 / CI 冒烟（headless 加速）
python scripts/05_replay_in_sim.py --dataset-root output/03_dataset_v3 \
    --episode 2 --scale 0.5 --speed 20.0 --no-gui
```

---

## 12. 实施进度

| Step | 内容 | 状态 |
|:-----|:-----|:-----|
| S1-S2 | HandTracker 抽象层 + 测试 | ✅ |
| S3-S4 | One-Euro filter + 测试 | ✅ |
| S5-S6 | Depth correction + 测试 | ✅ |
| S7-S8 | 01_hand_track.py（纯感知）+ 测试 | ✅ |
| S9 | 02_process.py（world transform + robust anchor + quality gate）+ 测试 | ✅ |
| S10 | 03_build_dataset.py（LeRobot v3 打包）+ 测试 | ✅ |
| S11 | HaMeR + MediaPipe + MANO + checkpoint 环境 | ✅ |
| S12 | HaMeR 真实实现 + MediaPipe 检测器 + cam_crop_to_full 修复 | ✅ |
| S13 | L1 格式：01→02→03 完整跑通 dataset_v3 | ✅ |
| S14 | L2 精度：裸手轨迹质量评估 | ✅ |
| S15 | L3 闭环：04 真机回放（wrist_flex 锚点 + flip_lateral + wrist_roll anchor） | ✅ 多 episode 成功 |

### 验证标准

| 级别 | 验证内容 | 通过标准 | 结果 |
|:-----|:---------|:---------|:-----|
| L1 格式 | 01→02→03 pipeline | pkl 格式正确, LeRobot v3 数据集可生成 | ✅ dataset_v3 261 / 294 帧 multi-episode |
| L2 精度 | 裸手轨迹质量 | 检测率>95%(active), SPARC>-4.0, 速度<2m/s | ✅ |
| L3 闭环 | 真机回放 | 机械臂复现裸手动作形态, gripper 开合正确 | ✅ IK 0.00mm, wrist_roll 首帧锚到 replay_start |

### 遗留项

1. **ViTPose 检测器**：HaMeR 原版 ViTDet→ViTPose→HaMeR 流程，需要安装 mmpose。MediaPipe 路径已跑通，ViTPose 作为后续优化
2. **Orientation 完整锚定**：当前 wrist_roll 用 Option A（constant delta）仅首帧对齐。若需要更严格的方向跟踪，可升级到完整 orientation anchor（SO(3) 相对旋转）

---

## 13. 关键设计决策

1. **wrist_flex = dataset 锚点**（不是 gripper_frame）：human wrist keypoint 对应机械臂 wrist_flex；02 robust anchor + 04 wrist sub-chain IK 保证"wrist ↔ wrist"物理对应
2. **4 层解耦**：01 感知 → 02 处理 → 03 打包 → 04 回放。上游追踪各自实现，下游在 `episode_local` 契约上统一消费
3. **双手追踪**：两只手独立 trim/filter/anchor；03 按 `--hand` 选择
4. **Gripper 信号**：thumb tip (joint 4) - index tip (joint 8) 距离（行业标准）；02 做 per-episode min/max 归一化
5. **深度修正**：透视缩放（pinhole camera model），精度 ±2-3cm
6. **时序滤波**：One-Euro filter（在 02 处理层，不在 01 感知层）
7. **wrist_roll Option A anchor**：`delta = replay_start - extracted[0]`；首帧严格对齐 replay_start，保持后续相对旋转
8. **flip_lateral**：iPhone 采集朝向和机器人部署左右相反时用（`M_flip = diag(1, 1, -1)`，世界 Z → arm Y 符号反转，det(R)=-1）
9. **safe_home vs replay_start 解耦**：SafeHome 管上电安全姿态，replay_start 是 IK 友好的展开姿态
10. **scale-around-anchor 而非 around-mean**：`positions *= scale`（anchor 固定在 0），不能 `(p - mean) * scale + mean`（会破坏 anchor 对齐）
