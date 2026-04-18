#!/usr/bin/env python3
"""
将 LIBERO training_data 中的样本改写为便于构建 VLM 微调数据集的格式：
1) 生成精炼后的 metadata.json（仅包含 instruction、actions_sequence）
2) 将离散 frames 合成为视频（main / wrist 两个视角）

默认输入:  ~/autodl-tmp/data/libero/training_data
默认输出:  ~/autodl-tmp/data/libero/processed_data

附加功能：
- 将指定视频转写为 GIF，便于直接预览
"""

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

from utils import SampleId, iter_sample_dirs, load_json, parse_views, slice_by_limit

FRAME_INDEX_RE = re.compile(r"%0(\d+)d")
VIEWS_TO_FRAME_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("main", "frame_%04d_main.jpg"),
    ("wrist", "frame_%04d_wrist.jpg"),
)
EVEN_SCALE_FILTER = "scale=trunc(iw/2)*2:trunc(ih/2)*2"  # 避免编码器因奇数尺寸而失败


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LIBERO training_data -> processed_data")
    parser.add_argument(
        "--transcode-video",
        type=str,
        default="",
        help="转写模式：输入视频路径（设置后将只执行转写，不处理数据集）",
    )
    parser.add_argument(
        "--transcode-dir",
        type=str,
        default="",
        help="批量转写模式：输入目录（包含 episode_XXXXXX_{type} 子目录），将其 videos/*.mp4 批量转为 gif",
    )
    parser.add_argument(
        "--transcode-output",
        type=str,
        default="",
        help="转写模式：输出文件路径（默认同目录同名 .gif）",
    )
    parser.add_argument(
        "--transcode-views",
        type=str,
        default="main,wrist",
        help="批量转写模式：需要转写的视角（逗号分隔），默认 main,wrist",
    )
    parser.add_argument("--gif-fps", type=int, default=10, help="转写 GIF 的帧率")
    parser.add_argument("--max-size", type=int, default=512, help="转写 GIF 的最长边（0 表示不缩放）")
    parser.add_argument("--stride", type=int, default=1, help="每隔多少帧取一帧（降采样）")
    parser.add_argument("--max-frames", type=int, default=0, help="最多写入多少帧（0 表示不限制）")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="~/autodl-tmp/data/libero/training_data",
        help="输入目录（包含 episode_XXXXXX_{type} 目录）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="~/autodl-tmp/data/libero/processed_data",
        help="输出目录",
    )
    parser.add_argument("--fps", type=int, default=20, help="输出视频帧率")
    parser.add_argument("--crf", type=int, default=23, help="H.264 CRF (越小越清晰/体积越大)")
    parser.add_argument("--preset", type=str, default="veryfast", help="ffmpeg x264 preset")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若输出文件已存在则覆盖（默认跳过已存在输出）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="最多处理多少个样本（0 表示不限制）",
    )
    return parser.parse_args()


def write_refined_metadata(metadata: Dict, output_path: Path) -> None:
    """
    仅保留 VLM 微调所需的字段，避免把原始 metadata 的冗余内容带入下游。
    """
    refined = {
        "instruction": metadata.get("instruction", ""),
        "actions_sequence": metadata.get("actions_sequence", []),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(refined, f, ensure_ascii=False, indent=2)


def _pattern_to_first_frame_name(frame_pattern: str) -> str:
    """
    将 ffmpeg 的序列帧 pattern（如 frame_%04d_main.jpg）转为首帧文件名。
    """
    match = FRAME_INDEX_RE.search(frame_pattern)
    if not match:
        return frame_pattern
    width = int(match.group(1))
    return FRAME_INDEX_RE.sub("0" * width, frame_pattern, count=1)




def frames_to_video_ffmpeg(
    frames_dir: Path,
    frame_pattern: str,
    output_path: Path,
    fps: int,
    crf: int,
    preset: str,
    overwrite: bool,
) -> bool:
    if shutil.which("ffmpeg") is None:
        return False

    input_pattern = frames_dir / frame_pattern
    first_frame = frames_dir / _pattern_to_first_frame_name(frame_pattern)
    if not first_frame.exists():
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-framerate",
        str(fps),
        "-i",
        str(input_pattern),
        "-vf",
        EVEN_SCALE_FILTER,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(crf),
        "-preset",
        preset,
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0 and output_path.exists()


def frames_to_video_cv2(
    frames_dir: Path,
    view_suffix: str,
    output_path: Path,
    fps: int,
    overwrite: bool,
) -> bool:
    try:
        import cv2
    except Exception:
        return False

    first_frame = frames_dir / f"frame_0000_{view_suffix}.jpg"
    if not first_frame.exists():
        return False

    if output_path.exists() and not overwrite:
        return True

    img0 = cv2.imread(str(first_frame), cv2.IMREAD_COLOR)
    if img0 is None:
        return False

    height, width = img0.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        return False

    idx = 0
    try:
        while True:
            frame_path = frames_dir / f"frame_{idx:04d}_{view_suffix}.jpg"
            if not frame_path.exists():
                break
            img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if img is None:
                return False
            if img.shape[0] != height or img.shape[1] != width:
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(img)
            idx += 1
    finally:
        writer.release()

    return output_path.exists() and idx > 0


def video_to_gif_cv2(
    video_path: Path,
    output_path: Path,
    gif_fps: int,
    max_size: int,
    stride: int,
    max_frames: int,
    overwrite: bool,
) -> Tuple[bool, str]:
    try:
        import cv2
    except Exception:
        return False, "missing dependency: cv2"

    try:
        from PIL import Image
    except Exception:
        return False, "missing dependency: PIL"

    if not video_path.exists():
        return False, f"video not found: {video_path}"

    if output_path.exists() and not overwrite:
        return True, "skip (already exists)"

    if gif_fps <= 0:
        return False, "gif_fps must be > 0"
    if stride <= 0:
        return False, "stride must be > 0"
    if max_size < 0:
        return False, "max_size must be >= 0"
    if max_frames < 0:
        return False, "max_frames must be >= 0"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, f"failed to open video: {video_path}"

    images: List[Image.Image] = []
    read_idx = 0
    kept = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if read_idx % stride != 0:
                read_idx += 1
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            if max_size and max(img.size) > max_size:
                w, h = img.size
                if w >= h:
                    new_w = max_size
                    new_h = max(1, int(round(h * (max_size / float(w)))))
                else:
                    new_h = max_size
                    new_w = max(1, int(round(w * (max_size / float(h)))))
                img = img.resize((new_w, new_h), resample=Image.BILINEAR)

            images.append(img)
            kept += 1
            read_idx += 1
            if max_frames and kept >= max_frames:
                break
    finally:
        cap.release()

    if not images:
        return False, "no frames decoded"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(round(1000.0 / float(gif_fps)))
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )

    if not output_path.exists():
        return False, "gif write failed"

    return True, "ok"


def transcode_dir_to_gifs(
    transcode_dir: Path,
    views: List[str],
    limit: int,
    gif_fps: int,
    max_size: int,
    stride: int,
    max_frames: int,
    overwrite: bool,
) -> Tuple[int, int, List[str]]:
    if not transcode_dir.exists():
        raise FileNotFoundError(f"transcode_dir not found: {transcode_dir}")

    sample_dirs = slice_by_limit(list(iter_sample_dirs(transcode_dir)), limit)

    ok_count = 0
    fail_count = 0
    failures: List[str] = []

    for _, sample_dir in sample_dirs:
        videos_dir = sample_dir / "videos"
        for view in views:
            video_path = videos_dir / f"{view}.mp4"
            output_path = video_path.with_suffix(".gif")
            ok, msg = video_to_gif_cv2(
                video_path=video_path,
                output_path=output_path,
                gif_fps=gif_fps,
                max_size=max_size,
                stride=stride,
                max_frames=max_frames,
                overwrite=overwrite,
            )
            if ok:
                ok_count += 1
            else:
                fail_count += 1
                failures.append(f"{sample_dir.name}/{view}: {msg}")

    return ok_count, fail_count, failures


def build_video_for_view(
    frames_dir: Path,
    view: str,
    frame_pattern: str,
    output_path: Path,
    fps: int,
    crf: int,
    preset: str,
    overwrite: bool,
) -> bool:
    """
    优先用 ffmpeg（速度快、兼容性好）合成视频；缺少 ffmpeg 时回退到 cv2。
    """
    ok = frames_to_video_ffmpeg(
        frames_dir=frames_dir,
        frame_pattern=frame_pattern,
        output_path=output_path,
        fps=fps,
        crf=crf,
        preset=preset,
        overwrite=overwrite,
    )
    if ok:
        return True
    return frames_to_video_cv2(
        frames_dir=frames_dir,
        view_suffix=view,
        output_path=output_path,
        fps=fps,
        overwrite=overwrite,
    )


def process_one_sample(
    sample_dir: Path,
    output_dir: Path,
    fps: int,
    crf: int,
    preset: str,
    overwrite: bool,
) -> Tuple[bool, str]:
    metadata_path = sample_dir / "metadata.json"
    frames_dir = sample_dir / "frames"

    if not metadata_path.exists():
        return False, f"missing metadata.json: {metadata_path}"
    if not frames_dir.exists():
        return False, f"missing frames/: {frames_dir}"

    metadata = load_json(metadata_path)

    out_sample_dir = output_dir / sample_dir.name
    out_metadata_path = out_sample_dir / "metadata.json"
    out_videos_dir = out_sample_dir / "videos"
    out_videos = {view: out_videos_dir / f"{view}.mp4" for view, _ in VIEWS_TO_FRAME_PATTERNS}

    if not overwrite and out_metadata_path.exists() and all(p.exists() for p in out_videos.values()):
        return True, "skip (already processed)"

    write_refined_metadata(metadata, out_metadata_path)

    results: Dict[str, bool] = {}
    for view, frame_pattern in VIEWS_TO_FRAME_PATTERNS:
        results[view] = build_video_for_view(
            frames_dir=frames_dir,
            view=view,
            frame_pattern=frame_pattern,
            output_path=out_videos[view],
            fps=fps,
            crf=crf,
            preset=preset,
            overwrite=overwrite,
        )

    if not all(results.values()):
        status = ", ".join(f"{k}={v}" for k, v in results.items())
        return False, f"video build failed ({status})"

    return True, "ok"


def run_transcode_video(args: argparse.Namespace) -> None:
    video_path = Path(args.transcode_video).expanduser()
    output_path = Path(args.transcode_output).expanduser() if args.transcode_output else video_path.with_suffix(".gif")

    ok, msg = video_to_gif_cv2(
        video_path=video_path,
        output_path=output_path,
        gif_fps=args.gif_fps,
        max_size=args.max_size,
        stride=args.stride,
        max_frames=args.max_frames,
        overwrite=args.overwrite,
    )
    if not ok:
        raise RuntimeError(msg)
    print(f"转写完成: {output_path}")


def run_transcode_dir(args: argparse.Namespace) -> None:
    transcode_dir = Path(args.transcode_dir).expanduser()
    views = parse_views(args.transcode_views)
    if not views:
        raise ValueError("transcode_views is empty")

    ok_count, fail_count, failures = transcode_dir_to_gifs(
        transcode_dir=transcode_dir,
        views=views,
        limit=args.limit,
        gif_fps=args.gif_fps,
        max_size=args.max_size,
        stride=args.stride,
        max_frames=args.max_frames,
        overwrite=args.overwrite,
    )

    print("=" * 60)
    print("转写完成")
    print(f"输入目录: {transcode_dir}")
    print(f"视角: {','.join(views)}")
    print(f"成功: {ok_count}")
    print(f"失败: {fail_count}")
    if failures:
        print("-" * 60)
        for line in failures[:50]:
            print(line)
        if len(failures) > 50:
            print(f"... (剩余 {len(failures) - 50} 条失败记录未展示)")
    print("=" * 60)

    if fail_count:
        raise RuntimeError("some videos failed to transcode")


def run_process_dataset(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    samples = slice_by_limit(list(iter_sample_dirs(input_dir)), args.limit)

    processed = 0
    skipped_or_ok = 0
    failed = 0
    failures: List[str] = []

    for _, sample_dir in samples:
        processed += 1
        ok, msg = process_one_sample(
            sample_dir=sample_dir,
            output_dir=output_dir,
            fps=args.fps,
            crf=args.crf,
            preset=args.preset,
            overwrite=args.overwrite,
        )
        if ok:
            skipped_or_ok += 1
        else:
            failed += 1
            failures.append(f"{sample_dir.name}: {msg}")

        if processed % 25 == 0:
            print(f"[{processed}/{len(samples)}] ok_or_skip={skipped_or_ok} failed={failed}")

    print("=" * 60)
    print("处理完成")
    print(f"输入: {input_dir}")
    print(f"输出: {output_dir}")
    print(f"总样本: {len(samples)}")
    print(f"成功/跳过: {skipped_or_ok}")
    print(f"失败: {failed}")
    if failures:
        print("-" * 60)
        for line in failures[:50]:
            print(line)
        if len(failures) > 50:
            print(f"... (剩余 {len(failures) - 50} 条失败记录未展示)")
    print("=" * 60)

def main() -> None:
    args = parse_args()

    if args.transcode_video:
        run_transcode_video(args)
        return

    if args.transcode_dir:
        run_transcode_dir(args)
        return

    run_process_dataset(args)


if __name__ == "__main__":
    main()
