# LIBERO 数据集下载与负样本生成工具

本项目包含两个主要脚本，用于下载 LIBERO 机器人操作数据集并生成用于 VLM（视觉语言模型）训练的正负样本。

---

## 文件说明

| 文件 | 功能 |
|------|------|
| `libero_to_qwen.py` | 从 Hugging Face 下载 LIBERO 数据集，按 episode 组织数据 |
| `generate_negative_samples.py` | 生成训练样本，按照 2:1:1:1 比例生成正样本和三种负样本 |

---

## 1. 数据下载脚本 (libero_to_qwen.py)

从 Hugging Face 下载 LIBERO 数据集，每个样本对应一个完整的 episode，包含所有帧图像、动作序列、状态序列和任务指令。

### 使用方法

```bash
python libero_to_qwen.py [参数]
```

### 可选参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--max-samples` | 10 | 最大下载样本数（每个样本是一个完整 episode） |
| `--output-dir` | `libero_qwen` | 输出目录 |

### 示例

```bash
# 下载 100 个 episodes
python libero_to_qwen.py --max-samples 100

# 下载 50 个 episodes，输出到自定义目录
python libero_to_qwen.py --max-samples 50 --output-dir my_data
```

### 输出结构

```
libero_qwen/
├── packaged/                    # 打包后的 episodes
│   ├── episode_000000/          # 每个 episode 一个目录
│   │   ├── frames/              # 帧图像
│   │   │   ├── frame_0000_main.jpg
│   │   │   ├── frame_0000_wrist.jpg
│   │   │   └── ...
│   │   ├── metadata.json        # episode 元数据
│   │   └── video_info.txt       # 视频文件信息
│   ├── episode_000001/
│   └── ...
├── videos/                      # 视频文件（74个）
│   ├── file-000.mp4
│   ├── file-001.mp4
│   └── ...
├── meta_tasks.json              # 任务指令映射
└── meta_episodes.json           # episodes 统计信息
```

### metadata.json 结构

```json
{
  "episode_index": 0,
  "task_index": 0,
  "instruction": "pick up the alphabet soup and place it in the basket",
  "num_frames": 284,
  "actions_sequence": [...],
  "state_sequence": [...],
  "actions_detail": {
    "position_delta": [...],
    "rotation_delta": [...],
    "gripper": [...]
  },
  "state_detail": {
    "position": [...],
    "rotation": [...],
    "gripper": [...]
  },
  "gripper_stats": {...},
  "frames": [...],
  "files": {
    "video": "../videos/file-000.mp4",
    "video_index": 0,
    "video_file": "file-000.mp4"
  }
}
```

---

## 2. 负样本生成脚本 (generate_negative_samples.py)

按照 **2:1:1:1** 的比例生成训练样本，用于训练 VLM 判别视频质量（检测是否对齐或畸变）。

### 样本类型说明

| 类型 | 比例 | 说明 | 标签 |
|------|------|------|------|
| **正样本** | 2 | 原始数据，保持不变 | 1 |
| **视觉负样本** | 1 | 修改帧图像的色相、饱和度、亮度、对比度 | 0 |
| **视觉动作未对齐** | 1 | 反转部分连续帧的夹子开关状态 | 0 |
| **动作指令未对齐** | 1 | 随机替换任务指令为其他任务的指令 | 0 |

### 使用方法

```bash
python generate_negative_samples.py [参数]
```

### 可选参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input-dir` | `libero_qwen/packaged` | 输入目录（包含下载的 episodes） |
| `--output-dir` | `training_data` | 输出目录 |
| `--seed` | 42 | 随机种子，保证可复现 |

### 示例

```bash
# 使用默认参数生成训练数据
python generate_negative_samples.py

# 指定输入输出目录
python generate_negative_samples.py \
    --input-dir libero_qwen/packaged \
    --output-dir my_training_data \
    --seed 123
```

### 输出结构

```
training_data/
├── episode_000000_positive/          # 正样本 (label=1)
│   ├── frames/
│   ├── metadata.json                 # 包含 label, sample_type
│   └── ...
├── episode_000001_positive/          # 正样本 (label=1)
├── episode_000002_visual/            # 视觉负样本 (label=0)
│   └── metadata.json                 # 包含 negative_sample_info
├── episode_000003_gripper/           # 视觉动作未对齐 (label=0)
├── episode_000004_instruction/       # 动作指令未对齐 (label=0)
├── ...
└── statistics.json                   # 统计信息
```

### 负样本标记示例

**视觉负样本** (`metadata.json`):
```json
{
  "sample_type": "negative_visual",
  "label": 0,
  "negative_sample_type": "hue_modified",
  "negative_sample_info": {
    "type": "视觉负样本",
    "description": "帧图像的色相、饱和度、亮度被随机修改",
    "original_label": 1,
    "negative_label": 0
  }
}
```

**视觉动作未对齐** (`metadata.json`):
```json
{
  "sample_type": "negative_gripper",
  "label": 0,
  "negative_sample_type": "gripper_inverted",
  "negative_sample_info": {
    "type": "视觉动作未对齐",
    "description": "帧 50 到 200 的夹子状态被反转",
    "inverted_frames": [50, 51, ..., 199],
    "original_label": 1,
    "negative_label": 0
  }
}
```

**动作指令未对齐** (`metadata.json`):
```json
{
  "sample_type": "negative_instruction",
  "label": 0,
  "negative_sample_type": "instruction_shuffled",
  "negative_sample_info": {
    "type": "动作指令未对齐",
    "description": "任务指令被随机替换",
    "original_instruction": "pick up the alphabet soup...",
    "new_instruction": "put the white mug on the plate...",
    "original_label": 1,
    "negative_label": 0
  }
}
```

---

## 完整使用流程

### 步骤 1: 下载数据

```bash
# 下载 100 个 episodes（约 100 个样本）
python libero_to_qwen.py --max-samples 100 --output-dir libero_qwen
```

### 步骤 2: 生成训练样本

```bash
# 按照 2:1:1:1 比例生成正负样本
python generate_negative_samples.py \
    --input-dir libero_qwen/packaged \
    --output-dir training_data \
    --seed 42
```

### 步骤 3: 验证结果

```bash
# 查看统计信息
cat training_data/statistics.json

# 查看某样本的 metadata
cat training_data/episode_000000_positive/metadata.json | jq '.label, .sample_type'
```

---

## 依赖安装

```bash
pip install pillow numpy pyarrow huggingface-hub
```

---

## 注意事项

1. **视频文件**: LIBERO 数据集只有 74 个视频文件，包含 1693 个 episodes 的所有帧。视频文件按顺序存储，每个视频约包含 23 个 episodes。

2. **磁盘空间**: 
   - 下载 100 个 episodes 约需 5-10 GB 空间
   - 生成训练样本会复制数据，需要额外空间

3. **随机种子**: 设置 `--seed` 参数可以保证每次生成的负样本类型相同，便于实验复现。

4. **负样本比例**: 2:1:1:1 的比例意味着每 5 个样本中有 2 个正样本和 3 个负样本（来自三种不同类型）。

---

## 数据集统计

- **总 episodes**: 1693
- **总视频文件**: 74
- **总任务指令**: 40
- **视频分辨率**: 通常为主视角 1280x720，手腕视角 640x480
- **帧率**: 10 FPS
