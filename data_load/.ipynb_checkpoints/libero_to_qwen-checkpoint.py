#!/usr/bin/env python3
"""
LIBERO数据集下载工具
每个样本对应一个完整的episode（包含该episode的所有帧、动作序列、状态序列）

使用方法:
    python libero_to_qwen.py --max-samples 100 --output-dir libero_qwen

参数:
    --max-samples: 最大下载样本数（每个样本是一个完整episode）
    --output-dir: 输出目录
"""

import argparse
import concurrent.futures
import io
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError, LocalEntryNotFoundError
from PIL import Image

from utils import write_json


def parse_args():
    parser = argparse.ArgumentParser(description="LIBERO数据集下载工具")
    parser.add_argument("--max-samples", type=int, default=100,
                        help="最大下载样本数（每个样本是一个完整episode）")
    parser.add_argument("--output-dir", default="libero_qwen", help="输出目录")
    parser.add_argument(
        "--hf-cache-dir",
        type=str,
        default="~/autodl-fs/cache/huggingface",
        help="HuggingFace Hub 缓存目录（建议放到 autodl-fs 以减少系统盘占用）",
    )
    parser.add_argument("--download-workers", type=int, default=8, help="并发下载线程数（parquet与视频）")
    parser.add_argument(
        "--episode-prefetch",
        type=int,
        default=32,
        help="每轮并发预取多少个episode parquet文件（建议>=download_workers）",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="每个chunk包含的episode parquet数量（用于生成 data/chunk-XYZ/episode_XXXXXX.parquet 路径）",
    )
    return parser.parse_args()


def save_image(raw: bytes, output_path: Path) -> str:
    """保存图像并返回文件名"""
    try:
        with Image.open(io.BytesIO(raw)) as img:
            img.load()
            path = output_path.with_suffix(f".{(img.format or 'png').lower()}")
            img.save(path)
            return path.name
    except Exception as e:
        print(f"保存图像失败: {e}")
        return None


def download_tasks(repo_id: str) -> Dict[int, str]:
    """下载任务指令"""
    print("\n[1/4] 下载任务指令...")

    import requests
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/meta/tasks.jsonl"

    try:
        print(f"  从 {url} 下载...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        content = response.text
        lines = content.strip().split('\n')

        tasks = {}
        for line in lines:
            try:
                data = json.loads(line)
                task_idx = data.get("task_index", 0)
                task_desc = data.get("task", "")
                if task_desc:
                    tasks[task_idx] = task_desc
            except Exception as e:
                continue

        print(f"  加载了 {len(tasks)} 个任务指令")
        return tasks

    except Exception as e:
        print(f"  下载 tasks.jsonl 失败: {e}")
        raise


def _hf_download(repo_id: str, filename: str) -> Tuple[Optional[str], str]:
    cache_dir = os.environ.get("HF_HUB_CACHE", None)

    try:
        local_file = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            cache_dir=cache_dir,
            local_files_only=True,
        )
        return local_file, "cache_hit"
    except LocalEntryNotFoundError:
        pass
    except Exception as e:
        return None, f"cache_check_failed: {type(e).__name__}: {e}"

    try:
        local_file = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            cache_dir=cache_dir,
        )
        return local_file, "downloaded"
    except HfHubHTTPError as e:
        return None, f"hf_http_error: {e}"
    except Exception as e:
        return None, f"download_failed: {type(e).__name__}: {e}"


def download_episodes(
    repo_id: str,
    output_dir: Path,
    max_episodes: int,
    download_workers: int,
    episode_prefetch: int,
    chunk_size: int,
) -> List[Dict]:
    """
    下载完整的episodes数据
    每个episode包含：
    - 所有帧的图像
    - 动作序列
    - 状态序列
    - 时间戳序列
    """
    print(f"\n[2/4] 下载episodes数据（最多{max_episodes}个）...")

    episodes_dir = output_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    episodes = []
    episode_count = 0

    next_idx = 0
    episode_prefetch = max(int(episode_prefetch), int(download_workers), 1)
    download_workers = max(int(download_workers), 1)
    chunk_size = max(int(chunk_size), 1)
    cache_hit = 0
    downloaded = 0
    failed = 0

    while episode_count < max_episodes:
        parquet_files = []
        for i in range(next_idx, next_idx + episode_prefetch):
            chunk_id = i // chunk_size
            parquet_files.append(f"data/chunk-{chunk_id:03d}/episode_{i:06d}.parquet")
        next_idx += episode_prefetch

        with concurrent.futures.ThreadPoolExecutor(max_workers=download_workers) as ex:
            fut_to_pf = {ex.submit(_hf_download, repo_id, pf): pf for pf in parquet_files}
            for fut in concurrent.futures.as_completed(fut_to_pf):
                if episode_count >= max_episodes:
                    break
                pf = fut_to_pf[fut]
                local_file, status = fut.result()
                if not local_file:
                    failed += 1
                    print(f"  跳过 {pf}: {status}")
                    continue
                if status == "cache_hit":
                    cache_hit += 1
                elif status == "downloaded":
                    downloaded += 1

                try:
                    table = pq.read_table(local_file)
                    rows = table.to_pylist()
                except Exception as e:
                    failed += 1
                    print(f"  跳过 {pf}: parquet_read_failed: {type(e).__name__}: {e}")
                    continue

                if len(rows) == 0:
                    continue

                # 提取episode信息
                episode_idx = episode_count

                # 收集该episode的所有帧数据
                frames = []
                for i, row in enumerate(rows):
                    frame_data = {
                        "frame_index": i,
                        "timestamp": row.get("timestamp", 0.0),
                        "actions": row.get("action", row.get("actions", [])),
                        "state": row.get("state", row.get("observation.state", [])),
                    }

                    # 提取图像数据 (LIBERO parquet中的字段名是 'image' 和 'wrist_image')
                    if "image" in row and row["image"]:
                        frame_data["image"] = row["image"]
                    if "wrist_image" in row and row["wrist_image"]:
                        frame_data["wrist_image"] = row["wrist_image"]

                    frames.append(frame_data)

                # 提取task_index（该episode的任务类型）
                task_idx = rows[0].get("task_index", rows[0].get("task", 0))

                episode_data = {
                    "episode_index": episode_idx,
                    "task_index": task_idx,
                    "num_frames": len(frames),
                    "frames": frames,
                    "parquet_file": pf
                }

                episodes.append(episode_data)
                episode_count += 1

                if episode_count % 10 == 0:
                    print(
                        f"    已获取 {episode_count}/{max_episodes} 个episodes"
                        f" (cache_hit={cache_hit}, downloaded={downloaded}, failed={failed})"
                    )

    print(f"  下载完成: {len(episodes)} 个episodes")
    return episodes


def download_all_videos(repo_id: str, output_dir: Path, download_workers: int) -> Dict[int, Path]:
    """
    下载所有可用视频（LIBERO只有74个视频）
    视频文件命名：file-000.mp4 到 file-073.mp4
    这些视频按顺序包含所有episodes的帧
    """
    print("\n[3/4] 下载视频...")
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    downloaded_videos = {}
    max_attempts = 74
    download_workers = max(int(download_workers), 1)

    def _download_video(i: int) -> Tuple[int, Optional[str], str]:
        video_path = f"videos/observation.images.image/chunk-000/file-{i:03d}.mp4"
        local_file, status = _hf_download(repo_id, video_path)
        return i, local_file, status

    to_download: List[int] = []
    skipped = 0
    failed = 0
    for i in range(max_attempts):
        dest_path = videos_dir / f"file-{i:03d}.mp4"
        if dest_path.exists():
            downloaded_videos[i] = dest_path
            skipped += 1
        else:
            to_download.append(i)

    with concurrent.futures.ThreadPoolExecutor(max_workers=download_workers) as ex:
        futs = [ex.submit(_download_video, i) for i in to_download]
        for fut in concurrent.futures.as_completed(futs):
            i, local_video, status = fut.result()
            if not local_video:
                failed += 1
                print(f"  跳过视频 file-{i:03d}.mp4: {status}")
                continue
            video_filename = f"file-{i:03d}.mp4"
            dest_path = videos_dir / video_filename
            if not dest_path.exists():
                shutil.copy(local_video, dest_path)
            downloaded_videos[i] = dest_path
            if len(downloaded_videos) % 10 == 0:
                print(f"  已下载 {len(downloaded_videos)} 个视频...")

    print(f"  视频下载完成: {len(downloaded_videos)} 个视频 (skip_existing={skipped}, failed={failed})")
    print(f"  注意: 74个视频文件包含1693个episodes的所有帧")
    return downloaded_videos


def save_frame_image(img_data, img_path: Path) -> bool:
    """保存帧图像，支持多种数据格式"""
    try:
        if img_data is None:
            return False

        # 如果是字典格式（LIBERO parquet存储方式）
        if isinstance(img_data, dict):
            img_bytes = img_data.get('bytes')
            if img_bytes:
                # 使用 PIL 打开并保存，确保格式正确
                try:
                    img = Image.open(io.BytesIO(img_bytes))
                    # 转换为 RGB（如果是RGBA）
                    if img.mode == 'RGBA':
                        img = img.convert('RGB')
                    img.save(img_path, 'JPEG', quality=95)
                    return True
                except Exception as e:
                    # 如果 PIL 失败，直接写入 bytes
                    with open(img_path, 'wb') as f:
                        f.write(img_bytes)
                    return True
            return False

        # 如果是 bytes，直接保存
        if isinstance(img_data, bytes):
            with open(img_path, 'wb') as f:
                f.write(img_data)
            return True

        # 如果是 numpy 数组
        if isinstance(img_data, np.ndarray):
            # 确保是 uint8 类型
            if img_data.dtype != np.uint8:
                img_data = np.clip(img_data, 0, 255).astype(np.uint8)
            # 保存为图像
            Image.fromarray(img_data).save(img_path)
            return True

        # 如果是 PIL Image
        if hasattr(img_data, 'save'):
            img_data.save(img_path)
            return True

        return False

    except Exception as e:
        print(f"    保存图像失败: {e}")
        return False


def save_episode_frames(episode: Dict, episode_dir: Path) -> List[str]:
    """保存episode的所有帧图像"""
    frames = episode["frames"]

    # frames 目录直接在 episode_dir 下
    frames_dir = episode_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    saved_images = []
    saved_count = 0

    for i, frame in enumerate(frames):
        frame_files = {}

        # 保存主视角图像
        if "image" in frame and frame["image"]:
            img_path = frames_dir / f"frame_{i:04d}_main.jpg"
            if save_frame_image(frame["image"], img_path):
                frame_files["main"] = f"frames/frame_{i:04d}_main.jpg"
                saved_count += 1

        # 保存手腕视角图像
        if "wrist_image" in frame and frame["wrist_image"]:
            img_path = frames_dir / f"frame_{i:04d}_wrist.jpg"
            if save_frame_image(frame["wrist_image"], img_path):
                frame_files["wrist"] = f"frames/frame_{i:04d}_wrist.jpg"
                saved_count += 1

        saved_images.append(frame_files)

    print(f"    保存了 {saved_count} 张图像")
    return saved_images


def package_episodes(episodes: List[Dict], tasks: Dict[int, str],
                     downloaded_videos: Dict[int, Path], output_dir: Path):
    """打包episodes数据（每个样本是一个完整episode）"""
    print(f"\n[4/4] 打包episodes...")

    packaged_dir = output_dir / "packaged"
    packaged_dir.mkdir(parents=True, exist_ok=True)

    episode_list = []

    for episode in episodes:
        episode_idx = episode["episode_index"]
        task_idx = episode["task_index"]

        # 获取任务指令
        instruction = tasks.get(task_idx, "perform the task")

        # 创建episode目录
        episode_dir = packaged_dir / f"episode_{episode_idx:06d}"
        episode_dir.mkdir(exist_ok=True)

        # 保存所有帧图像
        print(f"  保存episode {episode_idx} 的 {episode['num_frames']} 帧图像...")
        saved_images = save_episode_frames(episode, episode_dir)

        # 计算该episode对应的视频文件
        # LIBERO: 74个视频文件按顺序包含1693个episodes
        # 每个视频平均包含约23个episodes (1693/74 ≈ 23)
        video_index = episode_idx // 23  # 计算该episode属于哪个视频
        video_path = downloaded_videos.get(video_index)

        if video_path and video_path.exists():
            # 创建指向视频的引用文件（不复制大视频文件）
            video_relative = f"../videos/file-{video_index:03d}.mp4"
            # 创建一个小文件记录该episode在视频中的位置
            with open(episode_dir / "video_info.txt", 'w') as f:
                f.write(f"Video file: file-{video_index:03d}.mp4\n")
                f.write(f"Episode index in video: {episode_idx % 23}\n")
                f.write(f"Frame range: {episode_idx % 23 * episode['num_frames']} - {(episode_idx % 23 + 1) * episode['num_frames']}\n")
        else:
            video_relative = None

        # 提取动作序列和状态序列
        actions_sequence = [frame.get("actions", []) for frame in episode["frames"]]
        state_sequence = [frame.get("state", []) for frame in episode["frames"]]
        timestamps = [frame.get("timestamp", 0.0) for frame in episode["frames"]]

        # 解析动作和状态的详细信息
        # actions: [dx, dy, dz, droll, dpitch, dyaw, gripper]
        # state: [x, y, z, roll, pitch, yaw, gripper, timestamp]
        actions_detail = {
            "position_delta": [[a[0], a[1], a[2]] for a in actions_sequence if len(a) >= 3],
            "rotation_delta": [[a[3], a[4], a[5]] for a in actions_sequence if len(a) >= 6],
            "gripper": [a[6] for a in actions_sequence if len(a) >= 7],
            "description": "动作向量 [dx, dy, dz, droll, dpitch, dyaw, gripper]"
        }

        state_detail = {
            "position": [[s[0], s[1], s[2]] for s in state_sequence if len(s) >= 3],
            "rotation": [[s[3], s[4], s[5]] for s in state_sequence if len(s) >= 6],
            "gripper": [s[6] for s in state_sequence if len(s) >= 7],
            "description": "状态向量 [x, y, z, roll, pitch, yaw, gripper, ...]"
        }

        # 计算统计信息
        if actions_detail["gripper"]:
            gripper_values = actions_detail["gripper"]
            gripper_stats = {
                "min": min(gripper_values),
                "max": max(gripper_values),
                "mean": sum(gripper_values) / len(gripper_values),
                "open_frames": sum(1 for g in gripper_values if g > 0.5),
                "close_frames": sum(1 for g in gripper_values if g <= 0.5)
            }
        else:
            gripper_stats = {}

        # 构建episode元数据
        episode_metadata = {
            "episode_index": episode_idx,
            "task_index": task_idx,
            "instruction": instruction,
            "num_frames": episode["num_frames"],

            # 原始序列数据
            "actions_sequence": actions_sequence,
            "state_sequence": state_sequence,
            "timestamps": timestamps,

            # 详细解析的动作信息
            "actions_detail": actions_detail,

            # 详细解析的状态信息
            "state_detail": state_detail,

            # 夹爪统计信息
            "gripper_stats": gripper_stats,

            # 帧图像路径
            "frames": [
                {
                    "frame_index": i,
                    "timestamp": timestamps[i],
                    "images": saved_images[i] if i < len(saved_images) else {},
                    "action": actions_sequence[i] if i < len(actions_sequence) else [],
                    "state": state_sequence[i] if i < len(state_sequence) else []
                }
                for i in range(episode["num_frames"])
            ],

            # 文件信息
            "files": {
                "video": video_relative,
                "video_index": video_index,
                "video_file": f"file-{video_index:03d}.mp4" if video_relative else None,
                "parquet_source": episode["parquet_file"]
            }
        }

        write_json(episode_dir / "metadata.json", episode_metadata)

        episode_list.append(episode_metadata)

    print(f"  打包完成: {len(episode_list)} 个episodes")
    return episode_list


def save_dataset_info(episode_list: List[Dict], output_dir: Path):
    """保存数据集信息"""
    print(f"\n[5/5] 保存数据集信息...")

    # 计算统计信息
    total_frames = sum(ep["num_frames"] for ep in episode_list)
    avg_frames = total_frames / len(episode_list) if episode_list else 0

    write_json(output_dir / "sample_episode.json", {"examples": episode_list[:2]})

    # 保存数据集信息
    dataset_info = {
        "dataset_name": "LIBERO Episodes",
        "description": "LIBERO数据集，每个样本是一个完整episode（包含所有帧、动作序列、状态序列、详细解析信息）",
        "total_episodes": len(episode_list),
        "total_frames": total_frames,
        "avg_frames_per_episode": round(avg_frames, 2),
        "format": "Episode-based",
        "structure": {
            "episode": {
                "description": "一个完整的任务执行过程",
                "contains": [
                    "instruction: 任务指令",
                    "actions_sequence: 原始动作序列 [num_frames, 7]",
                    "actions_detail: 解析后的动作信息",
                    "state_sequence: 原始状态序列 [num_frames, 8]",
                    "state_detail: 解析后的状态信息",
                    "gripper_stats: 夹爪开合统计",
                    "timestamps: 时间戳序列",
                    "frames: 所有帧的详细信息（图像+动作+状态）",
                    "video: 完整视频文件"
                ]
            }
        },
        "universal_fields": {
            "actions_sequence": {
                "description": "原始动作序列（每个帧的动作向量）",
                "shape": "[num_frames, 7]",
                "format": "[dx, dy, dz, droll, dpitch, dyaw, gripper]",
                "importance": "⭐⭐⭐ 核心训练数据"
            },
            "actions_detail": {
                "description": "解析后的动作信息",
                "fields": {
                    "position_delta": "位置变化 [num_frames, 3] (dx, dy, dz)",
                    "rotation_delta": "旋转变化 [num_frames, 3] (droll, dpitch, dyaw)",
                    "gripper": "夹爪开合 [num_frames] (0=关闭, 1=打开)"
                },
                "importance": "⭐⭐⭐ 便于分析"
            },
            "state_sequence": {
                "description": "原始状态序列（每个帧的机器人状态）",
                "shape": "[num_frames, 8]",
                "importance": "⭐⭐⭐ 理解状态变化"
            },
            "state_detail": {
                "description": "解析后的状态信息",
                "fields": {
                    "position": "末端位置 [num_frames, 3] (x, y, z)",
                    "rotation": "末端旋转 [num_frames, 3] (roll, pitch, yaw)",
                    "gripper": "夹爪状态 [num_frames]"
                },
                "importance": "⭐⭐⭐ 便于分析"
            },
            "gripper_stats": {
                "description": "夹爪开合统计信息",
                "fields": ["min", "max", "mean", "open_frames", "close_frames"],
                "importance": "⭐⭐ 任务分析"
            },
            "frames": {
                "description": "所有帧的详细信息",
                "fields": ["frame_index", "timestamp", "images", "action", "state"],
                "importance": "⭐⭐⭐ 逐帧分析"
            },
            "instruction": {
                "description": "自然语言任务指令",
                "example": "pick up the bowl and place it on the plate",
                "importance": "⭐⭐⭐ 任务理解"
            },
            "video": {
                "description": "完整视频（包含该episode的所有帧）",
                "format": "MP4",
                "importance": "⭐⭐⭐ VLM视觉输入"
            },
            "timestamps": {
                "description": "时间戳序列（用于动作-视频对齐）",
                "importance": "⭐⭐ 对齐检查"
            }
        },
        "note": "每个样本是一个完整episode，包含详细解析的动作、状态、夹爪信息，适合训练需要理解完整任务过程的VLM/VLA模型"
    }

    write_json(output_dir / "dataset_info.json", dataset_info)

    print(f"  示例episode: {output_dir / 'sample_episode.json'}")
    print(f"  数据集信息: {output_dir / 'dataset_info.json'}")
    print(f"  总episodes: {len(episode_list)}")
    print(f"  总frames: {total_frames}")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hf_cache_dir = Path(args.hf_cache_dir).expanduser().resolve()
    hf_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_cache_dir)
    os.environ["HF_HUB_CACHE"] = str(hf_cache_dir / "hub")
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.environ["HF_HUB_CACHE"]
    (hf_cache_dir / "hub").mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LIBERO数据集下载工具（Episode-based）")
    print("=" * 60)
    print(f"输出目录: {output_dir.absolute()}")
    print(f"目标episodes: {args.max_samples}")
    print(f"HF缓存目录: {hf_cache_dir}")
    print("=" * 60)

    # 数据源配置
    data_repo = "physical-intelligence/libero"
    video_repo = "lerobot/libero_video"

    # 下载任务指令
    tasks = download_tasks(data_repo)

    # 下载episodes数据
    episodes = download_episodes(
        data_repo,
        output_dir,
        args.max_samples,
        download_workers=int(args.download_workers),
        episode_prefetch=int(args.episode_prefetch),
        chunk_size=int(args.chunk_size),
    )

    # 下载所有视频
    downloaded_videos = download_all_videos(video_repo, output_dir, download_workers=int(args.download_workers))

    if not episodes:
        print("错误: 没有加载到episodes数据")
        return

    # 打包episodes
    episode_list = package_episodes(episodes, tasks, downloaded_videos, output_dir)

    # 保存数据集信息
    save_dataset_info(episode_list, output_dir)

    print("\n" + "=" * 60)
    print("全部完成!")
    print("=" * 60)
    print(f"输出目录: {output_dir.absolute()}")
    print(f"总episodes: {len(episode_list)}")
    print(f"\n目录结构:")
    print(f"  {output_dir.name}/")
    print(f"    ├── packaged/")
    print(f"    │   ├── episode_000000/")
    print(f"    │   │   ├── frames/")
    print(f"    │   │   │   ├── frame_0000_main.jpg")
    print(f"    │   │   │   ├── frame_0000_wrist.jpg")
    print(f"    │   │   │   └── ...")
    print(f"    │   │   ├── video.mp4")
    print(f"    │   │   └── metadata.json")
    print(f"    │   ├── episode_000001/")
    print(f"    │   └── ...")
    print(f"    ├── videos/")
    print(f"    ├── sample_episode.json")
    print(f"    └── dataset_info.json")
    print("\n使用示例:")
    print(f"  python libero_to_qwen.py --max-samples 100")
    print("=" * 60)


if __name__ == "__main__":
    main()
