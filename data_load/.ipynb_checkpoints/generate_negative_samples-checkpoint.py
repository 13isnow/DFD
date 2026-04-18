#!/usr/bin/env python3
"""
生成负样本用于训练VLM判别视频质量
按照 2:1:1:1 的比例生成样本：
- 2: 正样本（保持不变）
- 1: 视觉负样本（基于指令对局部物体做颜色替换）
- 1: 视觉动作未对齐负样本（反转夹子状态）
- 1: 动作指令未对齐负样本（随机调换任务指令）

使用方法:
    python generate_negative_samples.py --input-dir libero_qwen/packaged --output-dir training_data

参数:
    --input-dir: 输入目录（包含下载的episodes）
    --output-dir: 输出目录
"""

import argparse
import json
import random
import re
import shutil
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="生成负样本用于VLM训练")
    parser.add_argument("--input-dir", type=str, default="libero_qwen/packaged",
                       help="输入目录（包含episodes）")
    parser.add_argument("--output-dir", type=str, default="training_data",
                       help="输出目录")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    return parser.parse_args()


def load_episodes(input_dir: Path) -> List[Path]:
    """加载所有episode目录"""
    episode_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith("episode_")])
    print(f"找到 {len(episode_dirs)} 个episodes")
    return episode_dirs


def copy_episode(src_dir: Path, dst_dir: Path, new_name: str) -> Path:
    """复制episode到目标目录"""
    new_dir = dst_dir / new_name
    if new_dir.exists():
        shutil.rmtree(new_dir)
    shutil.copytree(src_dir, new_dir)
    return new_dir


COLOR_RANGES = {
    "white": [((0, 0, 170), (180, 70, 255))],
    "yellow": [((18, 70, 80), (42, 255, 255))],
    "red": [((0, 90, 70), (12, 255, 255)), ((168, 90, 70), (180, 255, 255))],
    "orange": [((8, 90, 80), (22, 255, 255))],
    "blue": [((92, 70, 50), (132, 255, 255))],
    "green": [((40, 60, 50), (88, 255, 255))],
    "black": [((0, 0, 0), (180, 255, 65))],
    "gray": [((0, 0, 70), (180, 45, 180))],
    "grey": [((0, 0, 70), (180, 45, 180))],
    "brown": [((5, 80, 40), (22, 220, 170))],
    "pink": [((140, 40, 100), (175, 255, 255))],
    "purple": [((125, 50, 60), (160, 255, 255))],
}

CONTRAST_COLOR_MAP = {
    "white": (255, 0, 255),
    "yellow": (0, 0, 255),
    "red": (0, 255, 255),
    "orange": (0, 120, 255),
    "blue": (255, 255, 0),
    "green": (255, 0, 255),
    "black": (0, 255, 255),
    "gray": (255, 0, 0),
    "grey": (255, 0, 0),
    "brown": (0, 255, 255),
    "pink": (0, 255, 0),
    "purple": (0, 255, 0),
}

OBJECT_KEYWORDS = [
    "mug", "plate", "bowl", "basket", "bottle", "cup", "can", "spoon",
    "fork", "knife", "pan", "pot", "box", "block", "cracker", "soup",
    "tomato", "apple", "banana", "lemon", "pear", "peach", "strawberry",
    "book", "tray", "rack", "cabinet", "drawer", "lid", "handle"
]

OBJECT_COLOR_PRIORS = {
    "mug": ["white", "yellow", "blue", "red"],
    "plate": ["white", "yellow", "blue"],
    "bowl": ["white", "yellow", "blue", "red"],
    "basket": ["brown", "yellow"],
    "bottle": ["blue", "green", "red", "white"],
    "cup": ["white", "yellow", "blue"],
    "box": ["red", "blue", "yellow", "white"],
    "butter": ["yellow", "white"],
    "cracker": ["red", "white"],
    "soup": ["red", "white", "yellow"],
    "tomato": ["red"],
    "apple": ["red", "green"],
    "banana": ["yellow"],
    "lemon": ["yellow"],
    "pear": ["green", "yellow"],
    "strawberry": ["red"],
    "book": ["blue", "red", "green", "white"],
    "tray": ["white", "blue", "gray"],
}

GENERIC_FALLBACK_COLORS = ["yellow", "red", "blue", "green", "white", "orange", "brown"]

STOPWORDS = {
    "put", "pick", "place", "move", "the", "a", "an", "on", "in", "into",
    "to", "and", "left", "right", "top", "bottom", "of", "with", "from"
}


def rgb_to_hsv_np(img_rgb: np.ndarray) -> np.ndarray:
    """将 RGB 图像转为 OpenCV 风格 HSV（H: 0-180, S/V: 0-255）"""
    rgb = img_rgb.astype(np.float32) / 255.0
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    delta = maxc - minc

    h = np.zeros_like(maxc)
    nonzero = delta > 1e-6

    mask = nonzero & (maxc == r)
    h[mask] = ((g[mask] - b[mask]) / delta[mask]) % 6

    mask = nonzero & (maxc == g)
    h[mask] = (b[mask] - r[mask]) / delta[mask] + 2

    mask = nonzero & (maxc == b)
    h[mask] = (r[mask] - g[mask]) / delta[mask] + 4

    h = (h * 30.0) % 180.0
    s = np.zeros_like(maxc)
    nonzero_max = maxc > 1e-6
    s[nonzero_max] = delta[nonzero_max] / maxc[nonzero_max]
    v = maxc

    hsv = np.stack([h, s * 255.0, v * 255.0], axis=-1)
    return np.clip(hsv, 0, 255).astype(np.uint8)


def extract_target_objects(instruction: str) -> List[Dict[str, str]]:
    """从指令中抽取候选物体及其颜色描述"""
    lowered = instruction.lower()
    normalized = re.sub(r"[^a-z\s-]", " ", lowered)
    normalized = normalized.replace("-", " ")
    tokens = [t for t in normalized.split() if t]

    color_words = [t for t in tokens if t in COLOR_RANGES]
    candidates = []

    for obj in OBJECT_KEYWORDS:
        if obj not in tokens:
            continue
        indices = [i for i, token in enumerate(tokens) if token == obj]
        for obj_idx in indices:
            start = max(0, obj_idx - 4)
            phrase_tokens = tokens[start:obj_idx + 1]
            phrase_colors = [t for t in phrase_tokens if t in COLOR_RANGES]
            descriptor_tokens = [t for t in phrase_tokens if t not in STOPWORDS]
            phrase = " ".join(descriptor_tokens).strip() or obj
            candidates.append({
                "object": obj,
                "phrase": phrase,
                "source_color": phrase_colors[-1] if phrase_colors else "",
            })

    if candidates:
        candidates.sort(key=lambda x: (0 if x["source_color"] else 1, len(x["phrase"])))
        return candidates

    if color_words:
        return [{
            "object": "colored_region",
            "phrase": f"{color_words[0]} region",
            "source_color": color_words[0],
        }]

    return []


def get_candidate_colors(target: Dict[str, str]) -> List[str]:
    """为目标物体生成可尝试的颜色候选"""
    colors = []
    source_color = target.get("source_color", "")
    obj_name = target.get("object", "")

    if source_color:
        colors.append(source_color)

    for color in OBJECT_COLOR_PRIORS.get(obj_name, []):
        if color not in colors:
            colors.append(color)

    for color in GENERIC_FALLBACK_COLORS:
        if color not in colors:
            colors.append(color)

    return colors


def compute_component_score(
    bbox: Tuple[int, int, int, int],
    area: int,
    frame_shape: Tuple[int, int, int],
    target: Dict[str, str],
    color_name: str,
) -> float:
    """综合面积、中心位置和颜色匹配优先级给连通域打分"""
    x1, y1, x2, y2 = bbox
    height, width = frame_shape[:2]
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    dist_x = abs(center_x - width / 2.0) / max(1.0, width / 2.0)
    dist_y = abs(center_y - height / 2.0) / max(1.0, height / 2.0)
    center_bonus = max(0.05, 1.35 - 0.8 * dist_x - 0.55 * dist_y)

    score = float(area) * center_bonus
    if color_name == target.get("source_color") and color_name:
        score *= 1.5
    elif color_name in OBJECT_COLOR_PRIORS.get(target.get("object", ""), []):
        score *= 1.2

    return score


def build_color_mask(hsv: np.ndarray, color_name: str) -> np.ndarray:
    """根据颜色名构建掩码"""
    ranges = COLOR_RANGES.get(color_name, [])
    if not ranges:
        return np.zeros(hsv.shape[:2], dtype=bool)

    mask = np.zeros(hsv.shape[:2], dtype=bool)
    for lower, upper in ranges:
        lower_arr = np.array(lower, dtype=np.uint8)
        upper_arr = np.array(upper, dtype=np.uint8)
        current = np.all(hsv >= lower_arr, axis=-1) & np.all(hsv <= upper_arr, axis=-1)
        mask |= current
    return mask


def largest_connected_component(mask: np.ndarray) -> Optional[Tuple[int, int, int, int, int]]:
    """返回最大连通域的边界框与面积 (x1, y1, x2, y2, area)"""
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best = None
    best_area = 0

    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue

            queue = deque([(x, y)])
            visited[y, x] = True
            min_x = max_x = x
            min_y = max_y = y
            area = 0

            while queue:
                cx, cy = queue.popleft()
                area += 1
                min_x = min(min_x, cx)
                max_x = max(max_x, cx)
                min_y = min(min_y, cy)
                max_y = max(max_y, cy)

                for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                    if 0 <= nx < width and 0 <= ny < height and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((nx, ny))

            if area > best_area:
                best_area = area
                best = (min_x, min_y, max_x + 1, max_y + 1, area)

    return best


def choose_target_frames(frame_files: List[Path]) -> List[Path]:
    """选择少量中间帧进行局部替换，避免改动整个视频"""
    if not frame_files:
        return []
    if len(frame_files) <= 3:
        return frame_files[:1]

    candidate_indices = sorted({
        max(0, len(frame_files) // 3),
        max(0, len(frame_files) // 2),
        min(len(frame_files) - 1, (2 * len(frame_files)) // 3),
    })
    return [frame_files[i] for i in candidate_indices]


def recolor_region(img_rgb: np.ndarray, mask: np.ndarray, target_rgb: Tuple[int, int, int]) -> np.ndarray:
    """只重着色掩码区域，其余区域保持不变"""
    result = img_rgb.copy().astype(np.float32)
    if not np.any(mask):
        return img_rgb

    target = np.array(target_rgb, dtype=np.float32)
    alpha = 0.82
    result[mask] = result[mask] * (1.0 - alpha) + target * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def modify_frames_local_object_color(frames_dir: Path, metadata: Dict) -> Tuple[bool, Dict]:
    """
    基于 metadata 中的 instruction 抽取物体与颜色，
    遍历所有帧，对检测到目标物体的局部区域做高反差颜色替换。
    """
    try:
        frame_files = sorted(frames_dir.glob("frame_*_main.jpg"))
        if not frame_files:
            frame_files = sorted(frames_dir.glob("*.jpg"))
        if not frame_files:
            print("    未找到帧图像")
            return False, {}

        instruction = metadata.get("instruction", "")
        target_objects = extract_target_objects(instruction)
        if not target_objects:
            print("    指令中未抽取到可定位物体或颜色")
            return False, {}

        modified_frames = []
        target_object_name = target_objects[0]["object"] if target_objects else "unknown"
        target_phrase = target_objects[0]["phrase"] if target_objects else "unknown"
        source_color_used = ""
        matched_colors = set()
        replacement_colors = set()
        total_mask_area = 0

        for frame_file in frame_files:
            with Image.open(frame_file) as img:
                rgb = np.array(img.convert("RGB"))

            hsv = rgb_to_hsv_np(rgb)
            best_candidate = None

            for target in target_objects:
                for color_name in get_candidate_colors(target):
                    color_mask = build_color_mask(hsv, color_name)
                    component = largest_connected_component(color_mask)
                    if component is None:
                        continue

                    x1, y1, x2, y2, area = component
                    box_area = max(1, (x2 - x1) * (y2 - y1))
                    frame_area = rgb.shape[0] * rgb.shape[1]

                    if area < 90 or box_area < 144:
                        continue
                    if area > frame_area * 0.45:
                        continue

                    score = compute_component_score((x1, y1, x2, y2), area, rgb.shape, target, color_name)
                    if best_candidate is None or score > best_candidate["score"]:
                        best_candidate = {
                            "frame_file": frame_file,
                            "rgb": rgb,
                            "mask": color_mask[y1:y2, x1:x2],
                            "bbox": (x1, y1, x2, y2),
                            "score": score,
                            "mask_area": area,
                            "matched_color": color_name,
                            "target": target,
                        }

            if best_candidate is None:
                continue

            x1, y1, x2, y2 = best_candidate["bbox"]
            source_color = best_candidate["target"].get("source_color") or best_candidate["matched_color"]
            target_color = CONTRAST_COLOR_MAP.get(source_color, (255, 0, 255))

            patch = best_candidate["rgb"][y1:y2, x1:x2].copy()
            recolored_patch = recolor_region(patch, best_candidate["mask"], target_color)
            best_candidate["rgb"][y1:y2, x1:x2] = recolored_patch
            Image.fromarray(best_candidate["rgb"]).save(frame_file, "JPEG", quality=95)

            frame_match = re.search(r"frame_(\d+)", frame_file.name)
            frame_index = int(frame_match.group(1)) if frame_match else -1

            modified_frames.append({
                "frame_index": frame_index,
                "frame_file": frame_file.name,
                "bbox_xyxy": [x1, y1, x2, y2],
                "mask_area": int(best_candidate["mask_area"]),
                "matched_color": best_candidate["matched_color"],
                "replacement_color_rgb": list(target_color),
            })
            total_mask_area += int(best_candidate["mask_area"])
            matched_colors.add(best_candidate["matched_color"])
            replacement_colors.add(tuple(target_color))
            source_color_used = source_color
            target_object_name = best_candidate["target"]["object"]
            target_phrase = best_candidate["target"]["phrase"]

        if not modified_frames:
            print("    未找到可靠的局部物体区域")
            return False, {}

        info = {
            "type": "视觉负样本",
            "description": "根据任务指令抽取目标物体，在所有检测到目标的帧中进行局部高反差颜色替换",
            "target_object": target_object_name,
            "target_phrase": target_phrase,
            "source_color": source_color_used,
            "matched_colors": sorted(matched_colors),
            "replacement_colors_rgb": [list(c) for c in sorted(replacement_colors)],
            "modified_frames": modified_frames,
            "modified_frame_count": len(modified_frames),
            "modified_frame_files": [item["frame_file"] for item in modified_frames],
            "total_mask_area": total_mask_area,
            "original_label": 1,
            "negative_label": 0,
        }

        print(
            f"    已修改 {len(modified_frames)} 帧，目标: {target_phrase}，"
            f"示例帧: {modified_frames[0]['frame_file']}"
        )
        return True, info

    except Exception as e:
        print(f"    局部物体颜色替换失败: {e}")
        return False, {}


def invert_gripper_state(metadata: Dict) -> Dict:
    """
    反转夹子开关状态，生成视觉动作未对齐负样本
    将部分连续帧中的夹子状态反转
    """
    modified = metadata.copy()

    actions_sequence = modified.get("actions_sequence", [])
    if not actions_sequence:
        return modified

    num_frames = len(actions_sequence)

    # 随机选择一段连续帧进行反转（占总帧数的30%-70%）
    segment_length = random.randint(int(num_frames * 0.3), int(num_frames * 0.7))
    start_idx = random.randint(0, num_frames - segment_length)
    end_idx = start_idx + segment_length

    # 反转这段帧的夹子状态
    modified_actions = [list(a) for a in actions_sequence]  # 深拷贝

    for i in range(start_idx, end_idx):
        if len(modified_actions[i]) >= 7:
            # 反转第7个元素（夹子状态）
            original_gripper = modified_actions[i][6]
            modified_actions[i][6] = -original_gripper  # -1 <-> 1

    modified["actions_sequence"] = modified_actions

    # 更新 actions_detail
    if "actions_detail" in modified:
        modified["actions_detail"] = {
            "position_delta": [[a[0], a[1], a[2]] for a in modified_actions if len(a) >= 3],
            "rotation_delta": [[a[3], a[4], a[5]] for a in modified_actions if len(a) >= 6],
            "gripper": [a[6] for a in modified_actions if len(a) >= 7],
            "description": "动作向量 [dx, dy, dz, droll, dpitch, dyaw, gripper] (夹子状态已反转)"
        }

    # 更新 frames 中的 action
    if "frames" in modified:
        for i, frame in enumerate(modified["frames"]):
            if start_idx <= i < end_idx and "action" in frame:
                frame["action"] = modified_actions[i]

    # 添加负样本标记
    modified["negative_sample_type"] = "gripper_inverted"
    modified["negative_sample_info"] = {
        "type": "视觉动作未对齐",
        "description": f"帧 {start_idx} 到 {end_idx} 的夹子状态被反转",
        "inverted_frames": list(range(start_idx, end_idx)),
        "original_label": 1,  # 正样本标签
        "negative_label": 0   # 负样本标签
    }

    return modified


def shuffle_task_instruction(metadata: Dict, all_instructions: List[str]) -> Dict:
    """
    随机调换任务指令，生成动作指令未对齐负样本
    """
    modified = metadata.copy()

    original_instruction = modified.get("instruction", "")
    original_task_idx = modified.get("task_index", 0)

    # 随机选择一个不同的指令
    available_instructions = [i for i in all_instructions if i != original_instruction]
    if not available_instructions:
        return modified

    new_instruction = random.choice(available_instructions)

    modified["instruction"] = new_instruction

    # 添加负样本标记
    modified["negative_sample_type"] = "instruction_shuffled"
    modified["negative_sample_info"] = {
        "type": "动作指令未对齐",
        "description": "任务指令被随机替换",
        "original_instruction": original_instruction,
        "new_instruction": new_instruction,
        "original_task_index": original_task_idx,
        "original_label": 1,
        "negative_label": 0
    }

    return modified


def generate_training_set(episode_dirs: List[Path], output_dir: Path, seed: int = 42):
    """
    按照 2:1:1:1 的比例生成训练集
    """
    random.seed(seed)
    np.random.seed(seed)

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 收集所有指令用于随机调换
    all_instructions = []
    for ep_dir in episode_dirs:
        metadata_path = ep_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                instruction = metadata.get("instruction", "")
                if instruction:
                    all_instructions.append(instruction)

    all_instructions = list(set(all_instructions))  # 去重
    print(f"收集了 {len(all_instructions)} 个不同的任务指令")

    # 按照比例分配样本类型
    # 2:1:1:1 = 正样本:视觉负样本:视觉动作未对齐:动作指令未对齐
    sample_types = []
    for i in range(len(episode_dirs)):
        remainder = i % 5
        if remainder < 2:
            sample_types.append("positive")  # 2个正样本
        elif remainder == 2:
            sample_types.append("visual")    # 1个视觉负样本
        elif remainder == 3:
            sample_types.append("gripper")   # 1个视觉动作未对齐
        else:
            sample_types.append("instruction")  # 1个动作指令未对齐

    # 打乱顺序
    combined = list(zip(episode_dirs, sample_types))
    random.shuffle(combined)

    # 统计
    stats = {
        "positive": 0,
        "visual": 0,
        "gripper": 0,
        "instruction": 0
    }

    print(f"\n开始生成训练集（共 {len(episode_dirs)} 个样本）...")

    for idx, (ep_dir, sample_type) in enumerate(combined):
        print(f"\n[{idx+1}/{len(combined)}] 处理 {ep_dir.name} -> {sample_type}")

        # 生成新名称
        new_name = f"{ep_dir.name}_{sample_type}"

        # 复制episode
        new_dir = copy_episode(ep_dir, output_dir, new_name)

        # 加载metadata
        metadata_path = new_dir / "metadata.json"
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # 根据样本类型进行处理
        if sample_type == "positive":
            # 正样本：只添加标签
            metadata["sample_type"] = "positive"
            metadata["label"] = 1
            stats["positive"] += 1
            print(f"  -> 正样本 (label=1)")

        elif sample_type == "visual":
            # 视觉负样本：根据指令只修改局部物体颜色
            frames_dir = new_dir / "frames"
            if frames_dir.exists():
                success, visual_info = modify_frames_local_object_color(frames_dir, metadata)
                if success:
                    metadata["sample_type"] = "negative_visual"
                    metadata["label"] = 0
                    metadata["negative_sample_type"] = "local_object_color_replaced"
                    metadata["negative_sample_info"] = visual_info
                    stats["visual"] += 1
                    print(f"  -> 视觉负样本 (label=0)")
                else:
                    # 如果修改失败，作为正样本
                    metadata["sample_type"] = "positive"
                    metadata["label"] = 1
                    stats["positive"] += 1
                    print(f"  -> 未找到可靠物体区域，转为正样本 (label=1)")
            else:
                metadata["sample_type"] = "positive"
                metadata["label"] = 1
                stats["positive"] += 1
                print(f"  -> 无帧图像，转为正样本 (label=1)")

        elif sample_type == "gripper":
            # 视觉动作未对齐负样本：反转夹子状态
            metadata = invert_gripper_state(metadata)
            metadata["sample_type"] = "negative_gripper"
            metadata["label"] = 0
            stats["gripper"] += 1
            print(f"  -> 视觉动作未对齐负样本 (label=0)")

        elif sample_type == "instruction":
            # 动作指令未对齐负样本：调换任务指令
            metadata = shuffle_task_instruction(metadata, all_instructions)
            metadata["sample_type"] = "negative_instruction"
            metadata["label"] = 0
            stats["instruction"] += 1
            print(f"  -> 动作指令未对齐负样本 (label=0)")

        # 保存修改后的metadata
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    # 保存统计信息
    stats_path = output_dir / "statistics.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump({
            "total_samples": len(episode_dirs),
            "positive_samples": stats["positive"],
            "negative_visual": stats["visual"],
            "negative_gripper": stats["gripper"],
            "negative_instruction": stats["instruction"],
            "ratio": "2:1:1:1",
            "seed": seed
        }, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print("生成完成！")
    print(f"{'='*60}")
    print(f"总样本数: {len(episode_dirs)}")
    print(f"正样本: {stats['positive']}")
    print(f"视觉负样本: {stats['visual']}")
    print(f"视觉动作未对齐负样本: {stats['gripper']}")
    print(f"动作指令未对齐负样本: {stats['instruction']}")
    print(f"比例: 2:1:1:1")
    print(f"输出目录: {output_dir}")
    print(f"{'='*60}")


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"错误: 输入目录不存在: {input_dir}")
        return

    # 加载episodes
    episode_dirs = load_episodes(input_dir)

    if not episode_dirs:
        print("错误: 未找到任何episodes")
        return

    # 生成训练集
    generate_training_set(episode_dirs, output_dir, args.seed)


if __name__ == "__main__":
    main()
