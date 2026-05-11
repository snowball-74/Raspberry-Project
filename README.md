# 基于树莓派的深度学习手语识别系统

> 哈尔滨理工大学 计算机科学与技术学院 · 2026届本科毕业设计  
> 作者：黄婧怡

---

## 项目简介

本项目面向公共服务无障碍场景，在树莓派 5 等资源受限的边缘端设备上，实现了一套支持**静态手势**与**动态手语**双模式的实时孤立词手语识别系统。系统融合多维特征工程、时空平滑算法与轻量化模型部署，兼顾高精度与低延迟，具备良好的实用价值与可扩展性。

```
静态识别  →  MLP  (33类字母/数字，测试集准确率 99.28%)
动态识别  →  CNN-BiGRU  (CSL 100类常用词汇，测试集准确率 91.1%)
推理引擎  →  ONNX Runtime（CPU，毫秒级单帧推理）
界面框架  →  PyQt6 多线程异步架构，树莓派端帧率稳定 ~20 FPS
```

---

## 功能特性

- **双模式识别**：静态手势识别（字母 / 数字切换）+ 动态孤立词识别
- **双输入源**：摄像头实时录入 & 本地视频 / 图片文件导入
- **双去噪机制**：坐标 EMA 平滑 + 滑动窗口多数投票，有效抑制噪声闪烁
- **轻量化部署**：PyTorch 模型经 ONNX 转换，ONNX Runtime 在 ARM CPU 上实时推理
- **摄像头兼容**：自动切换 Picamera2（CSI）与 USB/内置摄像头，彻底解决 `libcamera` 设备残留问题
- **历史记录管理**：SQLite 持久化，支持按识别模式、输入方式、置信度区间多维筛选与批量删除
- **友好交互**：置信度阈值动态调节、识别序列自动拼接与一键清除

---

## 系统架构

```
Raspberry-Project/
├── main.py              启动器（模式选择 / 历史记录入口）
├── static.py            静态识别窗口
├── dynamic.py           动态识别窗口
├── camera.py            统一摄像头线程（CameraWorker）
├── db_manager.py        SQLite 数据库管理
├── history_ui.py        历史记录界面
├── model.py             CNN-BiGRU模型定义
├── requirements.txt     Python 依赖列表
│
├── train/               训练相关脚本（仅训练阶段使用）
│   ├── model.py         CNN-BiGRU 模型定义
│   ├── preprocess.py    动态模型数据预处理（165 维 → 172 维特征序列）
│   ├── augment.py       动态模型在线数据增强（10 种策略）
│   ├── train.py         动态模型训练主脚本
│   └── export_onnx.py   PyTorch → ONNX 转换与精度验证工具
│
└── model/               部署模型文件（训练完成后放入）
    ├── class.onnx           静态识别 ONNX 模型
    ├── class.json           静态识别类别名称
    ├── dynamic_model.onnx   动态识别 ONNX 模型
    ├── class_names.json     动态识别类别名称（100类）
    └── norm_params.json     动态特征归一化参数
```

**数据流**：`CameraWorker（采集）→ RecognitionThread（MediaPipe + 推理）→ PyQt6 UI（渲染 + 存库）`

---

## 硬件环境

| 组件 | 规格 |
|------|------|
| 主控板 | Raspberry Pi 5B（4GB / 8GB） |
| 摄像头 | CSI 广角摄像头 × 2 或 USB 摄像头 |
| 显示器 | 5 寸 HDMI 显示器（可选，也可 VNC 远程） |
| 操作系统 | Raspberry Pi OS (64-bit, Bookworm) |

> 本项目同样可运行于配备普通摄像头的 x86/x64 Linux / macOS / Windows 开发机，Picamera2 依赖会自动降级为 OpenCV。

---

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/snowball-74/Raspberry-Project.git
cd Raspberry-Project
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

> 树莓派端需额外安装 `picamera2`（系统包，通常已预装）：
> ```bash
> sudo apt install -y python3-picamera2
> ```

### 3. 放置模型文件

将训练好的模型文件放至 `model/` 目录（目录结构见上方）。

> 模型文件未随代码一并上传，如需获取请联系作者，或根据train文件夹内容自行训练。

### 4. 启动系统

```bash
python main.py
```

在启动器界面选择「静态识别」或「动态识别」即可开始使用。

---

## 模型训练教程

### 动态识别模型（CNN-BiGRU）

完整流程：**关键点提取 → 预处理 → 训练 → 导出 ONNX**

#### Step 1：准备原始数据

本项目动态模型使用 CSL 中国手语孤立词数据集。需先用 MediaPipe Holistic 对视频逐帧提取关键点，并保存为如下格式的 npz 文件（165 维 / 帧）：

```
dataset_keypoints_valid.npz
  ├── sequences    object 数组，每条为变长帧序列，每帧 165 维
  │                布局：Pose(39) + 右手槽位(63) + 左手槽位(63)
  ├── labels       (N,)  格式为 "词汇_视频ID"，如 "你好_001"
  └── seq_lengths  (N,)  各序列实际帧数（可选，缺失时自动推断）
```

165 维关键点布局：

| 区段 | 维度 | 内容 |
|------|------|------|
| `[0:39]` | 39 | Pose 13 个关键点（鼻、双肩、双肘、双腕等）× 3 |
| `[39:102]` | 63 | 右手 21 个关键点 × 3 |
| `[102:165]` | 63 | 左手 21 个关键点 × 3 |

#### Step 2：预处理

从原始关键点提取 172 维复合特征，并将序列对齐至固定 20 帧：

```bash
python train/preprocess.py \
  --input_npz  dataset_keypoints_valid.npz \
  --output_npz dataset_aligned.npz \
  --n_frames   20
```

预处理完成后输出 `dataset_aligned.npz`，包含：

| 键名 | 形状 | 说明 |
|------|------|------|
| `sequences` | (N, 20, 172) | 每条样本的帧级特征序列 |
| `start_poses` | (N, 40) | 起始手型向量（前 3 帧均值） |
| `end_poses` | (N, 40) | 终止手型向量（后 3 帧均值） |
| `labels` | (N,) | 整数类别标签 |

172 维特征布局（右手 86 维 + 左手 86 维，结构对称）：

```
右手 [0:86]
  [0:15]   R1(拇指尖) → 食/中/无名/小指尖、掌心（5×3）
  [15:27]  R2(掌心)   → 食/中/无名/小指尖（4×3）
  [27:45]  R2(掌心)   → 鼻/左腕/左掌心/左肩/右肩/左肘（6×3）
  [45:60]  R3(腕点)   → 鼻/左腕/左肘/左肩/右肩（5×3）
  [60:65]  五指弯曲角度
  [65:80]  五指尖相对掌心速度 Δ（5×3）
  [80:83]  掌心相对肩中心速度 Δ
  [83:86]  腕点相对肩中心速度 Δ
左手 [86:172]  结构与右手完全对称
```

#### Step 3：训练

```bash
python train/train.py
```

训练脚本顶部配置区可调整主要超参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `DATASET_PATH` | `dataset_aligned.npz` | 预处理后数据集路径 |
| `EPOCHS` | 200 | 训练轮数 |
| `BATCH_SIZE` | 16 | 批大小 |
| `LR` | 1e-3 | 初始学习率（AdamW + CosineAnnealing） |
| `BIDIRECTIONAL` | `True` | 是否使用双向 GRU |
| `GRU_HIDDEN` | 128 | GRU 隐层维度 |
| `AUGMENT` | `True` | 是否开启在线数据增强 |

训练采用**按人划分**策略（随机取 2 个视频 ID 作为验证集，其余为训练集），避免同一个人的不同样本同时出现在训练集和验证集中。

训练结束后自动保存以下文件：

| 文件 | 说明 |
|------|------|
| `dynamic_model.pth` | 最佳验证集权重（含归一化参数与类别名） |
| `class_names.json` | 类别名称列表 |
| `curves_bigru_aug.png` | 训练曲线图（Loss / Accuracy） |

#### Step 4：导出 ONNX

```bash
python train/export_onnx.py \
  --pth  dynamic_model.pth \
  --out  dynamic_model.onnx \
  --seq  20
```

可选参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--pth` | `dynamic_model.pth` | 输入权重路径 |
| `--out` | `dynamic_model.onnx` | 输出 ONNX 路径 |
| `--seq` | 20 | 序列帧数，需与训练一致 |
| `--opset` | 12 | ONNX opset 版本（兼容 onnxruntime 1.10+） |

导出完成后会自动进行精度验证，输出 PyTorch 与 ONNX Runtime 的最大误差（正常应 < 1e-3）。同时自动生成 `norm_params.json`（推理时必须，包含归一化均值、标准差等参数）。

将以下三个文件复制到 `model/` 目录即可：

```bash
cp dynamic_model.onnx  model/
cp class_names.json    model/
cp norm_params.json    model/
```

---

### 静态识别模型（MLP）

静态模型的输入为单帧 68 维特征，由 `static.py` 中的特征提取逻辑实时计算：

| 区段 | 维度 | 内容 |
|------|------|------|
| 归一化坐标 | 63 | 21 个手部关键点 × 3（坐标以腕关节为原点，整体缩放到 [-1, 1]） |
| 几何先验 | 5 | 关键指尖间欧氏距离（拇指尖-食指尖、食指尖-腕、中指尖-腕、食中指尖间距、拇指尖-腕） |

网络为多层感知器 MLP，使用自建数据集（33 类手势字母/数字，4700+ 样本）训练，测试准确率 99.28%。

训练好的模型需导出为 ONNX 并放置到 `model/` 目录：

```
model/class.onnx    静态识别 ONNX 模型
model/class.json    类别名称列表（33类）
```

> 静态模型训练脚本未包含在本仓库中，如需获取请联系作者。

---

### 数据增强说明

训练时内置 10 种在线数据增强策略（见 `augment.py`），按如下顺序依概率叠加应用：

| 增强方法 | 默认概率 | 说明 |
|----------|----------|------|
| `time_scale` | 0.5 | 时间轴缩放，模拟快/慢手语 |
| `time_warp` | 0.5 | 时间轴非线性扭曲，改变动作节奏 |
| `frame_drop` | 0.4 | 随机丢帧后插值，模拟追踪抖动 |
| `spatial_shift` | 0.7 | 相对坐标整体平移 |
| `spatial_zoom` | 0.9 | 距离模拟缩放（0.5×~2.2×），覆盖不同使用距离 |
| `random_erase` | 0.3 | 随机连续帧置零，模拟短暂遮挡 |
| `mirror` | 0.0 | 左右镜像（手语不对称，默认关闭） |
| `occlude_left_hand` | 0.35 | 左手全通道清零，提升单手鲁棒性 |
| `occlude_right_hand` | 0.10 | 右手全通道清零（低概率辅助） |
| `noise` | 0.8 | 全通道高斯噪声（速度/角度通道单独控制倍率） |

---

## 模型说明

### 静态识别

- **输入特征**：68 维（21 个手部关键点 3D 坐标 63 维 + 指尖欧氏距离几何先验 5 维）
- **网络**：多层感知器 MLP（PyTorch 训练，ONNX 部署）
- **数据集**：自建数据集，33 类手势，4700+ 样本
- **测试准确率**：99.28%

### 动态识别

- **输入特征**：172 维帧级复合特征（双手手型 + 双手轨迹 + 躯干姿态）+ 40 维起止手型（首/尾各 3 帧均值）
- **网络**：CNN（2 层 1D 卷积）+ BiGRU（双向双层）+ MLP 起止姿态分支，特征融合后分类头输出
- **数据集**：CSL 中国手语孤立词数据集，100 类常用词汇
- **测试准确率**：91.1%
- **序列长度**：固定 20 帧（线性插值对齐）

---

## 主要技术栈

| 类别 | 技术 |
|------|------|
| 视觉感知 | MediaPipe Hands / Holistic |
| 深度学习 | PyTorch → ONNX → ONNX Runtime |
| 界面框架 | PyQt6（多线程 QThread 异步架构） |
| 数据库 | SQLite（via Python sqlite3） |
| 摄像头 | Picamera2 / OpenCV VideoCapture |
| 语言 | Python 3.11+ |

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `main.py` | 系统主入口，启动器界面 |
| `static.py` | 静态手势识别窗口（字母/数字） |
| `dynamic.py` | 动态手语识别窗口（孤立词） |
| `camera.py` | 统一摄像头线程，支持 Picamera2 / OpenCV / 视频文件 |
| `db_manager.py` | SQLite 数据库 CRUD 操作封装 |
| `history_ui.py` | 历史记录查看与管理界面 |
| `train/model.py` | CNN-BiGRU 模型结构定义 |
| `train/preprocess.py` | 动态模型数据预处理（165 维关键点 → 172 维特征序列） |
| `train/augment.py` | 动态模型在线数据增强（10 种策略） |
| `train/train.py` | 动态模型训练脚本（CNN-BiGRU，含归一化与数据划分） |
| `train/export_onnx.py` | 将 `.pth` 权重导出为 ONNX，并自动精度验证 |

---

## 项目背景

据世界卫生组织统计，我国听力残疾人数已达 2780 万，听力障碍人群超过 7200 万。手语作为核心沟通方式，在公共服务、医疗、教育等场景中因专业翻译人员稀缺而面临严重障碍。本项目旨在以低成本嵌入式设备为载体，降低手语识别技术的部署门槛，为信息无障碍建设提供实用的技术方案。

---

## 致谢

感谢研究生指导老师和本校指导教师在整个毕业设计过程中给予的悉心指导与支持。  
感谢 MediaPipe、PyTorch、ONNX Runtime、PyQt6 等开源社区提供的优秀工具与框架。

---

## License

本项目为学术毕业设计，代码仅供学习与研究使用。 
