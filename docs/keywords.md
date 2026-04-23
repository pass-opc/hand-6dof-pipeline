# 关键词字典 — hand-6dof-pipeline

开发过程中遇到的数学和机器人学概念速查。

---

## Homogeneous Matrix（齐次矩阵）

**定义**：4x4 矩阵，把旋转 (R) 和平移 (t) 打包成一个可以做矩阵乘法的结构。

```
T = [ R  | t ]     R = 3x3 旋转矩阵（朝向）
    [ 0  | 1 ]     t = 3x1 平移向量（位置）
                    底部 [0,0,0,1] 是为了让矩阵乘法同时完成旋转+平移
```

**为什么不直接用 6D pose `[x,y,z,rx,ry,rz]`？**

6D pose 无法直接链式相乘。要组合两个变换必须先转矩阵。用 4x4 一行搞定：

```
T_world_marker = T_world_camera @ T_camera_marker
```

**为什么叫"齐次"？**

多出的底部那行让旋转+平移统一成一次矩阵乘法：
- 没有底部行：`p_new = R @ p + t`（两步）
- 有底部行：`[p_new; 1] = T @ [p; 1]`（一步）

**具体例子**：marker 在相机前方 30cm、右移 10cm、无旋转：

```
T = [ 1  0  0 | 0.10 ]     x = 0.10m（右移10cm）
    [ 0  1  0 | 0.00 ]     y = 0.00m
    [ 0  0  1 | 0.30 ]     z = 0.30m（前方30cm）
    [ 0  0  0 | 1.00 ]
```

同一个 marker，但绕 z 轴旋转了 90°：

```
T = [ 0 -1  0 | 0.10 ]     左上 3x3 变成了 90° 旋转矩阵
    [ 1  0  0 | 0.00 ]     右上 3x1 平移不变
    [ 0  0  1 | 0.30 ]
    [ 0  0  0 | 1.00 ]
```

**代码对应**：`pose_to_mat()` 把 6D 转成 4x4，`mat_to_pose()` 转回来。内部运算全用 4x4，输入输出用 6D。

**相关文件**：`utils/pose_util.py`

---

## SE(3) — 三维特殊欧几里得群

**定义**：三维空间中所有合法刚体变换（旋转+平移）的集合。代码里看到 SE(3) 就是指"合法的 4x4 齐次矩阵"。

**名字拆解**：
- **S**（Special）= R 的行列式为 +1（正旋转，不含镜像翻转）
- **E**（Euclidean）= 旋转 + 平移（保持距离不变）
- **(3)** = 三维空间

**相关的群**：
- **SO(3)** = 只有旋转（3x3 旋转矩阵）
- **SE(3)** = SO(3) + 平移 = 完整刚体变换

**对代码的意义**：

1. **求逆很便宜**：旋转矩阵是正交的，`R^{-1} = R^T`（转置就是逆）。所以不用通用的 `np.linalg.inv(T)`，可以直接：
   ```
   T_inv = [ R^T | -R^T @ t ]
           [ 0   |     1    ]
   ```
   `invert_transform()` 做的就是这件事——结果一样，速度更快、数值更稳定。

2. **组合就是乘法**：两个 SE(3) 矩阵相乘还是 SE(3)，坐标变换链能一直乘下去：
   ```
   T_world_tcp = T_world_cam @ T_cam_marker @ T_marker_tcp
   ```

**不需要学群论**——知道 SE(3) = "合法的刚体变换 4x4 矩阵"就够了。

**相关文件**：`utils/pose_util.py::invert_transform()`

---

## rot6d — 6D 旋转表示

**定义**：取 3x3 旋转矩阵的前两行，展平成 6 个数。

```
R = [ r1 ]        rot6d = [r1, r2] = 6 个数
    [ r2 ]    →   （第三行 r3 = r1 × r2，是冗余的，可以从前两行恢复）
    [ r3 ]
```

**为什么不用 axis-angle (3D) 或 quaternion (4D)？**

Zhou et al. (CVPR 2019) 证明了：任何少于 5 维的旋转表示都是**不连续的**——神经网络很难学不连续的映射。rot6d 是最小的连续表示。

- axis-angle (3D)：在 ±π 处不连续——网络输出会跳变
- quaternion (4D)：q 和 -q 表示同一个旋转——有歧义
- rot6d (6D)：连续、无歧义，多出的冗余度无所谓

**在我们 pipeline 中的位置**：数据存储用 axis-angle（紧凑，3D）。训练前通过 `mat_to_pose10d()` 转成 pos(3) + rot6d(6) = 9D。

> 注意：UMI 里叫 "pose10d"，但实际输出是 9D，不是 10D。这是 UMI 的命名错误，我们沿用了函数名但在测试中记录了这个问题。

**相关文件**：`utils/pose_util.py::mat_to_rot6d()`、`utils/pose_util.py::mat_to_pose9d()`

---

## rvec / tvec — OpenCV 的 6DoF 位姿输出格式

**定义**：OpenCV 的 ArUco 检测（`estimatePoseSingleMarkers`）返回的两个向量，合在一起描述目标在相机坐标系中的 6DoF 位姿（**T_camera_marker**）。

- **rvec**（rotation vector，旋转向量）：3D 向量，方向是旋转轴，长度是旋转角度（弧度）。也叫 Rodrigues 向量。
- **tvec**（translation vector，平移向量）：3D 向量 `[x, y, z]`，目标在相机坐标系下的位置，单位米。

**具体例子**：

```
rvec = [0, 0, 1.5708]
  → 旋转轴 = z 轴，旋转角度 = |rvec| = 1.5708 弧度 = 90°

rvec = [0.1, 0.2, 0.3]
  → 旋转轴 = normalize([0.1, 0.2, 0.3]) = [0.267, 0.535, 0.802]
  → 旋转角度 = sqrt(0.1² + 0.2² + 0.3²) = 0.374 弧度 ≈ 21.4°

tvec = [0.10, -0.05, 0.30]
  → 目标在相机右方 10cm、上方 5cm、前方 30cm
```

**OpenCV 相机坐标系**（注意 y 轴朝下，和直觉相反）：

```
        z（前方，指向场景）
       /
      /
     O ———— x（右）
     |
     y（下）
```

**对代码的意义**：

rvec 本质就是 axis-angle，所以 `rvec_tvec_to_pose()` 直接拼接成 `[tvec, rvec]` = `[x, y, z, rx, ry, rz]`，然后通过 `pose_to_mat()` 转成 4x4 齐次矩阵参与变换链。

`cv2.Rodrigues()` 可以在 rvec（3D）和 3x3 旋转矩阵之间互转，但我们用 `scipy.spatial.transform.Rotation` 统一处理，不直接调 Rodrigues。

**相关文件**：`utils/pose_util.py::rvec_tvec_to_pose()`、`utils/pose_util.py::rvec_tvec_to_mat()`、`scripts/01_aruco_detect.py::detect_aruco_markers()`

---

## 流式处理（Streaming Processing）— 逐帧读取-处理-丢弃，避免 OOM

**定义**：从 .r3d（zip 压缩包）中逐帧读取图片，当场做检测/处理，只保留结果（几十字节的 rvec/tvec），立刻丢弃原始图片（~8MB/帧）。

**为什么不一次性全读入内存？**

ArUco 检测和棋盘格检测都是逐帧独立的，不需要跨帧数据。1920×1440 的 iPhone 帧每张 ~8MB，1000 帧 = 8GB，直接 OOM。流式处理全程只有 1 张图片在内存中。

**内存对比**：

```
全量读取：1328 帧 × 8MB = ~10GB → OOM ✗
流式处理：1 帧 × 8MB + 1328 个检测结果 × ~56B = ~8MB → OK ✓
```

**处理流程**：

```
for 每帧 in zip:
    读 jpg → 解码 → 检测 marker/棋盘格 → 保存结果 → 丢弃图片
              ↑                                          ↑
         内存中唯一的图片                              立刻释放
```

**什么时候仍需保留帧**：

- `--visualize` 需要显示几帧图片 → 只保留检测到目标的前几帧
- `04_generate_dataset.py` 需要写 RGB 到数据集 → 仍需逐帧读取写入，但同样可以流式

**代码对应**：

- `03_calibrate_eef.py::detect_all_frames_streaming()` — 流式 ArUco 检测
- `02_calibrate_camera.py::main()` — 流式棋盘格检测，只保留有棋盘的帧

**相关文件**：`scripts/02_calibrate_camera.py`、`scripts/03_calibrate_eef.py`

---

## T_marker_wrist — marker 到手腕轴心的刚体偏移

**定义**：4×4 齐次矩阵，描述从某个手腕 marker 的坐标系到手腕旋转轴心的固定变换。贴上去就不变。

**为什么需要？**

手腕贴了多个 marker（每面一个），每个 marker 的 PnP 结果（T_cam_marker）都不同——因为它们贴在不同位置。但我们需要的是统一的手腕轴心位姿。偏移补偿了这个差异：

```
T_cam_wrist = T_cam_markerN @ T_markerN_wrist
              ├── 每帧不同（marker 在动）
              └── 固定常量（标定一次）
```

**标定方法（自动）**：

```
1. 手腕慢转一圈 → 相邻 marker 有共视帧
2. 同帧看到 A 和 B → T_A_B = inv(T_cam_A) @ T_cam_B（固定刚体关系）
3. 多对 marker 链式传递 → 所有 marker 统一到参考坐标系
4. 拟合旋转轴圆弧 → 找到轴心位置
5. 每个 marker 到轴心的 offset 就确定了
```

**对代码的意义**：运行时不管哪个面朝相机，检测到任意一个手腕 marker 就能算出统一的轴心位姿。多个可见时取平均更精确。

**相关文件**：`scripts/03_calibrate_eef.py::calibrate_wrist_offsets()`、`scripts/01_aruco_detect.py::compute_wrist_pose()`

---

## Task Space vs Joint Space — 两种控制空间

**Task Space（任务空间 / Cartesian Space）**：用末端位姿 `[x, y, z, rx, ry, rz]` 描述机械臂状态。

**Joint Space（关节空间）**：用各关节角度 `[θ1, θ2, θ3, θ4, θ5, θ6]` 描述机械臂状态。

```
Task Space:  "末端在世界坐标 (0.18, -0.12, -0.15)，朝向 ..."
Joint Space: "1号舵机 30°, 2号舵机 45°, 3号舵机 -20°, ..."
```

**本 pipeline 数据集用 Task Space**——上游追踪直接给出末端位姿，不知道关节角度。这也是 UMI 的做法：Task Space 数据跨机械臂通用，不同机械臂只需各自的 IK 求解器。

**LeRobot SO-arm 101 默认用 Joint Space**——舵机直接反馈关节角度。但 LeRobot 有 EE 模式，通过 FK/IK 处理器在两个空间之间转换。

**对下游的影响**：推理时 policy 输出 Task Space action → IK 转换为 Joint Space → 舵机执行。

---

## FK（Forward Kinematics，正运动学）— 关节角度 → 末端位姿

**定义**：已知各关节角度，计算末端在空间中的位置和朝向。

```
输入: [θ1=30°, θ2=45°, θ3=-20°, θ4=10°, θ5=60°, θ6=0°]
输出: [x=0.18, y=-0.12, z=-0.15, rx, ry, rz]
```

**特点**：
- **确定性**：给定关节角度，末端位姿有且只有一个解
- **计算方法**：沿运动链逐关节做齐次矩阵乘法 `T_base_ee = T_01 @ T_12 @ ... @ T_56`
- 每个 `T_{i,i+1}` 由 DH 参数（连杆长度、关节偏移等）和当前关节角度决定

**本 pipeline 中的用途**：推理时读取舵机角度反馈 → FK → 当前末端位姿 → 作为 observation.state 输入 policy。

**相关代码**：LeRobot `robot_kinematic_processor.py::ForwardKinematicsJointsToEE`

---

## IK（Inverse Kinematics，逆运动学）— 末端位姿 → 关节角度

**定义**：已知目标末端位姿，求解各关节应该转到什么角度。

```
输入: [x=0.20, y=-0.10, z=-0.13, rx, ry, rz]  ← policy 预测的目标
输出: [θ1=35°, θ2=42°, θ3=-18°, θ4=12°, θ5=58°, θ6=3°]  ← 发给舵机
```

**特点**：
- **可能多解**：多个关节组合可以到达同一个末端位姿（如肘朝上/朝下）
- **可能无解**：目标超出机械臂工作范围
- **奇异点**：某些位姿下关节冗余/锁死，IK 不稳定

**求解方法**：
- 解析解（6-DoF 机械臂有标准公式，精确但需要推导）
- 数值迭代（Jacobian 迭代，通用但可能收敛慢）
- LeRobot 用数值方法（适用于各种结构）

**本 pipeline 中的用途**：推理时 policy 输出 Task Space action → IK → 关节角度 → 舵机执行。

**相关代码**：LeRobot `robot_kinematic_processor.py::InverseKinematicsEEToJoints`

---

## Observation / State / Action — 训练和推理的核心数据结构

### Observation（观测）— Policy 的输入

**"机器人看到了什么 + 当前状态"**

```
observation = {
    "images.rgb":  (480, 640, 3) uint8      ← 相机画面
    "images.depth": (480, 640, 3) uint8     ← 深度图（可选）
    "state":       (7,) float32             ← 当前末端位姿 或 关节角度
}
```

训练时从数据集读取，推理时从相机+舵机反馈实时获取。Policy 通常看过去 N 帧的 observation（observation horizon，一般 N=2）。

### State（状态）— 机器人当前物理状态

**就是 `observation.state`，同一个东西的不同叫法。**

```
Task Space:        state = [x, y, z, rx, ry, rz, gripper]     (7D)
LeRobot Joint:     state = [θ1, θ2, θ3, θ4, θ5, gripper]     (6D)
```

### Action（动作）— Policy 的输出

**"机器人接下来要做什么"**

```
action = (7,) float32  ← 目标末端位姿 或 目标关节角度
```

本 pipeline 的约定（和 UMI 一致）：`action[t] = state[t+1]`，即绝对目标位姿。

**不是** 速度或力——是"你应该到达这个位置"。机械臂控制器（PD/PID）负责平滑地从当前位置移到目标位置。

### Action Horizon 和 Action Chunking

```
Policy 一次预测未来 H 步的 action 序列:

  action_chunk = [a_t, a_{t+1}, ..., a_{t+H-1}]    H = action horizon (通常 16)

执行时只用前 K 步 (K < H)，然后重新预测:
  → 执行 [a_t ... a_{t+K-1}]
  → 重新观测
  → 预测新的 [a_{t+K} ... a_{t+K+H-1}]
  → 执行前 K 步 ...（循环）
```

**为什么一次预测多步？** 时序一致性——如果每步独立预测，连续动作之间可能不协调（比如忽左忽右）。一次预测一整段轨迹保证了动作连贯性。这是 Diffusion Policy 和 ACT 的核心设计。

### 各环节的数据流

```
采集:
  iPhone RGB → ArUco → state/action (task space) → LeRobot dataset

训练:
  DataLoader 取样本:
    输入: obs_images (过去2帧RGB) + obs_state (过去2帧state)
    标签: action_chunk (未来16帧action)
  Loss = MSE(predicted_chunk, ground_truth_chunk)

推理 (实时循环, 30Hz):
  相机 → RGB ─────────────┐
  舵机 → FK → state ──────┤
                           ↓
                      Policy (GPU, ~10ms)
                           ↓
                    action_chunk [16, 7]
                           ↓ 取前8步
                    IK → 关节命令 [6]
                           ↓
                    舵机执行
                           ↓
                    执行完8步 → 回到顶部重新观测
                    (下一批 action 在执行期间已提前算好，无缝衔接)
```

---

## Receding Horizon（滑动窗口执行）— 预测多步，执行少步，循环修正

**定义**：Policy 一次预测 H 步（如 16 步），但只执行前 K 步（如 8 步），然后重新观测、重新预测。

```
预测①: [a1, a2, ..., a8, a9, ..., a16]
        执行 ──────────┘  丢弃 ─────────┘

  → 重新拍照 + 读 state

预测②:          [b1, b2, ..., b8, b9, ..., b16]
                 执行 ──────────┘  丢弃 ─────────┘
```

**为什么不全部执行 16 步？** 开环执行会累积误差——预测第 16 步时已经偏离现实了。只执行前 8 步，然后用最新观测修正。

**会不会顿挫？** 不会。推理（~10ms）远快于执行周期（~33ms@30Hz），在执行第 5-6 步时下一批预测就算完了，等着无缝衔接。推理和执行是异步的。

**相关论文**：Diffusion Policy (Chi et al. 2023), ACT (Zhao et al. 2023)

---

## 坐标系对齐 — pipeline 世界坐标系 vs 机械臂基座坐标系

**问题**：pipeline 数据集里的位姿基于"棋盘格定义的世界坐标系"（02_calibrate_camera 的输出），而机械臂 FK/IK 用的是"机械臂基座坐标系"。两者不一定一致。

```
pipeline 数据: state = T_world_cam @ T_cam_marker     ← 世界坐标系
SO-arm FK:    state = FK(joints)                      ← 基座坐标系

如果 world ≠ base → IK 收到错误目标 → 机械臂动作不对
```

**解决方法**：测量或标定 `T_base_world`（机械臂基座在世界坐标系中的位姿），推理时做一次坐标变换。

**相关文件**：`scripts/02_calibrate_camera.py`（定义世界坐标系）

---

## PnP 跳变 vs Axis-Angle 2π 跳变 — 两种不同的"数值不连续"

**PnP 跳变（真错误）**：marker 半遮挡/模糊时 solvePnP 给出不准确的位姿。xyz 和旋转都可能跳。转任何格式都无法修复——数据本身就是错的。

**2π 跳变（表示歧义）**：axis-angle 的 +π 和 -π 表示同一个旋转，数值差 6.28 但物理上完全一样。转成旋转矩阵后自动消失，转 rot6d 后连续。

```
PnP 跳变:   帧100 xyz=[0.22, -0.09, -0.18]  ← PnP 算错了，需要过滤
2π 跳变:    帧100 ry=-3.13 (vs 帧99 ry=+3.13) ← 同一个旋转，不需要处理
```

**处理方式**：
- PnP 跳变：`mark_bad_frames()` 标记为 NaN → 插值修复
- 2π 跳变：训练时转 rot6d 自动解决，数据集阶段不处理

**相关文件**：`utils/interpolation.py::mark_bad_frames()`

---

<!-- 
新词条模板（复制到上方，填入内容）：

## 术语 — 一句话描述

**定义**：

**对代码的意义**：

**具体例子**：

**相关文件**：

---
-->
