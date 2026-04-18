#!/usr/bin/env python3
"""
将 processed_data 转为 QwenVL / Qwen2.5-VL 微调所需的 jsonl 格式。

输入目录结构（每个样本目录）：
- metadata.json: 至少包含 instruction、actions_sequence；可选 cot / cot_structured
- videos/{view}.mp4: 视角视频（默认 main / wrist）

输出 jsonl 单条结构：
{
  "id": "episode_000123_positive",
  "videos": ["/abs/path/to/main.mp4", ...],
  "messages": [
    {"role": "user", "content": "...含 <video> 占位符与上下文JSON..."},
    {"role": "assistant", "content": "...监督信号(JSON字符串)..."}
  ]
}
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils import SampleId, iter_samples, load_json, summarize_actions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate QwenVLM finetune jsonl from processed_data")
    parser.add_argument("--input-dir", type=str, default="~/autodl-tmp/data/libero/processed_data")
    parser.add_argument("--output-path", type=str, default="~/autodl-tmp/data/libero/train.jsonl")
    parser.add_argument("--val-output-path", type=str, default="~/autodl-tmp/data/libero/val.jsonl")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample-types",
        type=str,
        default="",
        help="参与构建的数据类型（逗号分隔），如 positive,visual；为空表示不过滤",
    )
    parser.add_argument(
        "--type-ratios",
        type=str,
        default="",
        help="各类型采样比例（逗号分隔 type=ratio），如 positive=1,visual=1；未指定的类型默认=1",
    )
    parser.add_argument("--max-actions", type=int, default=20)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--fps-max-frames", type=int, default=8)
    parser.add_argument(
        "--assistant-label-only",
        action="store_true",
        help='监督信号只输出 {"label":0|1}，不包含cot/evidence等字段；同时不再要求metadata.json里必须有cot',
    )
    return parser.parse_args()


def sample_frame_indices_by_fps(video_path: Path, fps: float, max_frames: int) -> List[int]:
    """
    以“按时间等间隔”的方式抽帧，并返回帧索引列表。

    - fps > 0：以给定 fps 采样，最多 max_frames 帧
    - fps <= 0：在 [0, frame_count-1] 上等分取 max_frames 个点
    """
    import cv2

    if max_frames <= 0:
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    try:
        video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()

    if frame_count <= 0:
        return []

    if fps <= 0:
        k = min(max_frames, frame_count)
        if k <= 1:
            return [0]
        step = (frame_count - 1) / float(k - 1)
        indices = [int(round(i * step)) for i in range(k)]
    else:
        if video_fps <= 0:
            indices = list(range(min(max_frames, frame_count)))
        else:
            indices = []
            for i in range(max_frames):
                t = i / float(fps)
                idx = int(round(t * video_fps))
                indices.append(idx)

    cleaned: List[int] = []
    prev = -1
    for idx in indices:
        if idx < 0:
            continue
        if idx >= frame_count:
            break
        if idx <= prev:
            idx = prev + 1
        if idx >= frame_count:
            break
        cleaned.append(idx)
        prev = idx

    return cleaned


def issue_focus_from_sample_type(sample_type: str) -> str:
    if sample_type == "positive":
        return "ok"
    if sample_type == "visual":
        return "image_quality"
    if sample_type == "gripper":
        return "action_alignment"
    if sample_type == "instruction":
        return "instruction_alignment"
    return "unknown"


def allowed_issue_focuses_from_types(sample_types: List[str]) -> List[str]:
    focuses = {issue_focus_from_sample_type(t) for t in sample_types}
    order = ["ok", "image_quality", "action_alignment", "instruction_alignment"]
    return [f for f in order if f in focuses]


def label_from_sample_type(sample_type: str) -> int:
    return 1 if sample_type == "positive" else 0


def build_user_text(
    instruction: str,
    action_summary: Dict[str, Any],
    sampled_frame_indices: List[int],
    sampling_fps: float,
    sampling_max_frames: int,
    include_views: List[str],
    include_action_context: bool,
    allowed_issue_focuses: Optional[List[str]] = None,
    assistant_label_only: bool = False,
) -> str:
    """
    构造 user 消息文本：采用结构化 JSON（类似 data_cot_process.py 的输入风格），并追加 <video> 占位符。

    这里不直接内嵌视频路径，而是通过 jsonl 的 videos 字段提供给多模态引擎。
    """
    video_lines = ["Main view: <video>"]

    issue_focuses = list(allowed_issue_focuses or []) or ["none", "temporal_error", "physical_violation", "semantic_mismatch", "low_perceptibility"]
    issue_focus_str = "|".join(issue_focuses)

    if assistant_label_only:
        fields = {"label": "0 (drop) or 1 (keep)"}
    else:
        fields = {
            "thought": "Analyze the trajectory step-by-step. Compare the action_summary with the visual feedback.",
            "issue_focus": f"Strictly one of: {issue_focus_str}. Use 'none' ONLY if label is 1.",
            "label": "1 (keep) if issue_focus is 'none'; 0 (drop) otherwise. Consistency is MANDATORY.",
            "evidence": "Cite specific views and [Frame ID]. E.g., '[Frame 20] Gripper penetrates the cube'.",
            "decision": "keep|drop"
        }

    payload: Dict[str, Any] = {
        "role": "Balanced Data Quality Auditor",
        "task": "Classify robot manipulation data for VLM training. Be strict but fair.",
        "instruction": instruction,
        "policy": {
            "keep_criteria": [
                "The core interaction is visible and follows the instruction.",
                "Minor motion blur or lighting shifts that do not obscure the object/gripper are ACCEPTABLE.",
                "Small trajectory jitters that don't break physical laws are ACCEPTABLE."
            ],
            "drop_criteria": [
                "Total failure: Gripper misses target, or object teleports.",
                "Severe occlusion: The main interaction point is completely invisible.",
                "Instruction violation: Robot does A when told to do B."
            ]
        },
        "requirements": {
            "output_format": "JSON_ONLY",
            "logic_constraint": "If issue_focus is NOT 'none', label MUST be 0. If issue_focus is 'none', label MUST be 1.",
            "fields": fields
        }
    }
    if include_action_context:
        payload["action_summary"] = action_summary

    return json.dumps(payload, ensure_ascii=False) + "\n" + "\n".join(video_lines)


def build_assistant_text(
    label: int,
    issue_focus: str,
    cot: str,
    cot_structured: Optional[Dict[str, Any]],
    assistant_label_only: bool = False,
) -> str:
    if assistant_label_only:
        return json.dumps({"label": int(label)}, ensure_ascii=False)

    decision = "keep" if int(label) == 1 else "drop"

    cot_text = str(cot or "").strip()
    evidence: List[str] = []
    quality_score: Optional[int] = None
    if isinstance(cot_structured, dict):
        maybe_cot = cot_structured.get("cot")
        if isinstance(maybe_cot, str) and maybe_cot.strip():
            cot_text = maybe_cot.strip()

        ev = cot_structured.get("evidence")
        if isinstance(ev, list):
            evidence = [str(x).strip() for x in ev if str(x).strip()][:4]

        qs = cot_structured.get("quality_score")
        if isinstance(qs, (int, float)):
            quality_score = int(round(float(qs)))

    if quality_score is None:
        quality_score = 100 if decision == "keep" else 0

    payload: Dict[str, Any] = {
        "decision": decision,
        "label": int(label),
        "issue_focus": issue_focus,
        "quality_score": int(quality_score),
        "evidence": evidence,
        "cot": cot_text,
    }
    return json.dumps(payload, ensure_ascii=False)


def stratified_split(
    samples: List[Tuple[SampleId, Path]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[SampleId, Path]], List[Tuple[SampleId, Path]]]:
    """
    按 sample_type 分组后分别 shuffle 并切分，尽量保持 train/val 的类型分布一致。
    """
    if val_ratio <= 0:
        return samples, []
    if val_ratio >= 1:
        return [], samples

    rng = random.Random(seed)
    groups: Dict[str, List[Tuple[SampleId, Path]]] = {}
    for sid, p in samples:
        groups.setdefault(sid.sample_type, []).append((sid, p))

    train: List[Tuple[SampleId, Path]] = []
    val: List[Tuple[SampleId, Path]] = []

    for _, items in sorted(groups.items(), key=lambda x: x[0]):
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        n_val = int(round(n * val_ratio))
        if n_val <= 0:
            train.extend(items)
        elif n_val >= n:
            val.extend(items)
        else:
            val.extend(items[:n_val])
            train.extend(items[n_val:])

    train.sort(key=lambda x: (x[0].episode_index, x[0].sample_type, x[1].name))
    val.sort(key=lambda x: (x[0].episode_index, x[0].sample_type, x[1].name))
    return train, val


def parse_sample_types(sample_types_raw: str) -> List[str]:
    return [s.strip() for s in (sample_types_raw or "").split(",") if s.strip()]


def parse_type_ratios(type_ratios_raw: str) -> Dict[str, float]:
    ratios: Dict[str, float] = {}
    raw = (type_ratios_raw or "").strip()
    if not raw:
        return ratios
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"invalid type ratio item: {part} (expected type=ratio)")
        k, v = part.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            raise ValueError(f"invalid type ratio item: {part}")
        try:
            ratio = float(v)
        except Exception:
            raise ValueError(f"invalid ratio value: {part}")
        if ratio <= 0:
            raise ValueError(f"ratio must be > 0: {part}")
        ratios[k] = ratio
    return ratios


def filter_and_resample_by_type(
    samples: List[Tuple[SampleId, Path]],
    allowed_types: List[str],
    type_ratios: Dict[str, float],
    seed: int,
) -> List[Tuple[SampleId, Path]]:
    """
    先按 allowed_types 过滤，再按 type_ratios 做下采样（downsample）以匹配比例。

    该策略不会复制样本，只会在每个类型内随机取子集。
    """
    if allowed_types:
        allowed_set = set(allowed_types)
        samples = [s for s in samples if s[0].sample_type in allowed_set]

    if not samples:
        return []

    rng = random.Random(seed)
    groups: Dict[str, List[Tuple[SampleId, Path]]] = {}
    for sid, p in samples:
        groups.setdefault(sid.sample_type, []).append((sid, p))

    ratios: Dict[str, float] = {}
    for t in groups.keys():
        ratios[t] = float(type_ratios.get(t, 1.0))

    base = None
    for t, items in groups.items():
        r = ratios[t]
        cap = len(items) / r
        base = cap if base is None else min(base, cap)
    if base is None:
        return []

    selected: List[Tuple[SampleId, Path]] = []
    for t, items in groups.items():
        r = ratios[t]
        target = int(round(base * r))
        target = max(0, min(target, len(items)))
        items = list(items)
        rng.shuffle(items)
        selected.extend(items[:target])

    selected.sort(key=lambda x: (x[0].episode_index, x[0].sample_type, x[1].name))
    return selected


def write_jsonl(
    output_path: Path,
    samples: List[Tuple[SampleId, Path]],
    include_views: List[str],
    fps: float,
    fps_max_frames: int,
    max_actions: int,
    allowed_issue_focuses: List[str],
    include_action_context: bool,
    assistant_label_only: bool,
) -> Tuple[int, int, int, List[str], Dict[str, int], Dict[str, int], Dict[str, int]]:
    """
    将样本列表写为 jsonl，并返回 (written, skipped, failed, failures)。
    """
    written = 0
    skipped = 0
    failed = 0
    failures: List[str] = []
    example_record: Optional[Dict[str, Any]] = None
    written_by_type: Dict[str, int] = {}
    skipped_by_type: Dict[str, int] = {}
    failed_by_type: Dict[str, int] = {}

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out_f:
        for sample_id, sample_dir in samples:
            metadata_path = sample_dir / "metadata.json"
            videos_dir = sample_dir / "videos"

            if not metadata_path.exists():
                failed += 1
                failed_by_type[sample_id.sample_type] = failed_by_type.get(sample_id.sample_type, 0) + 1
                failures.append(f"{sample_dir.name}: missing metadata.json")
                continue
            if not videos_dir.exists():
                failed += 1
                failed_by_type[sample_id.sample_type] = failed_by_type.get(sample_id.sample_type, 0) + 1
                failures.append(f"{sample_dir.name}: missing videos/")
                continue

            metadata = load_json(metadata_path)
            cot = str(metadata.get("cot", "")).strip()
            if (not assistant_label_only) and (not cot):
                skipped += 1
                skipped_by_type[sample_id.sample_type] = skipped_by_type.get(sample_id.sample_type, 0) + 1
                continue
            cot_structured = metadata.get("cot_structured") if (not assistant_label_only) else None

            instruction = str(metadata.get("instruction", ""))
            actions_sequence = metadata.get("actions_sequence", [])

            missing_video = False
            sampling_video_path: Optional[Path] = None
            videos: List[str] = []
            selected_views = ["main"]
            video_path = videos_dir / "main.mp4"
            if not video_path.exists():
                missing_video = True
            else:
                sampling_video_path = video_path
                videos.append(video_path.resolve().as_posix())

            if missing_video:
                failed += 1
                failed_by_type[sample_id.sample_type] = failed_by_type.get(sample_id.sample_type, 0) + 1
                failures.append(f"{sample_dir.name}: missing video for views=main")
                continue
            if sampling_video_path is None:
                failed += 1
                failed_by_type[sample_id.sample_type] = failed_by_type.get(sample_id.sample_type, 0) + 1
                failures.append(f"{sample_dir.name}: sampling_view video not found: main")
                continue

            sampled_frame_indices = sample_frame_indices_by_fps(
                video_path=sampling_video_path,
                fps=float(fps),
                max_frames=int(fps_max_frames),
            )
            sampled_action_sequence: List[Any] = []
            if isinstance(actions_sequence, list):
                for i in sampled_frame_indices:
                    if 0 <= i < len(actions_sequence):
                        sampled_action_sequence.append(actions_sequence[i])

            action_summary = summarize_actions(sampled_action_sequence, max_actions=int(max_actions))

            label = label_from_sample_type(sample_id.sample_type)
            issue_focus = issue_focus_from_sample_type(sample_id.sample_type)

            user_text = build_user_text(
                instruction=instruction,
                action_summary=action_summary,
                sampled_frame_indices=sampled_frame_indices,
                sampling_fps=float(fps),
                sampling_max_frames=int(fps_max_frames),
                include_views=selected_views,
                include_action_context=bool(include_action_context),
                allowed_issue_focuses=allowed_issue_focuses,
                assistant_label_only=bool(assistant_label_only),
            )

            assistant_text = build_assistant_text(
                label=label,
                issue_focus=issue_focus,
                cot=cot,
                cot_structured=cot_structured if isinstance(cot_structured, dict) else None,
                assistant_label_only=bool(assistant_label_only),
            )

            record = {
                "id": sample_dir.name,
                "videos": videos,
                "messages": [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": assistant_text},
                ],
            }

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            written_by_type[sample_id.sample_type] = written_by_type.get(sample_id.sample_type, 0) + 1
            if example_record is None:
                example_record = record

    if example_record is not None:
        example_path = output_path.parent / "example.json"
        with open(example_path, "w", encoding="utf-8") as f:
            json.dump(example_record, f, ensure_ascii=False, indent=2)

    return written, skipped, failed, failures, written_by_type, skipped_by_type, failed_by_type


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser()
    output_path = Path(args.output_path).expanduser()
    val_output_path = Path(args.val_output_path).expanduser()

    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    val_output_path.parent.mkdir(parents=True, exist_ok=True)

    include_views = ["main"]
    if not include_views:
        raise ValueError("include_views is empty")

    samples = iter_samples(input_dir)
    allowed_types = parse_sample_types(args.sample_types)
    type_ratios = parse_type_ratios(args.type_ratios)
    samples = filter_and_resample_by_type(
        samples=samples,
        allowed_types=allowed_types,
        type_ratios=type_ratios,
        seed=int(args.seed),
    )
    if not samples:
        raise RuntimeError("no samples after applying sample_types/type_ratios")

    from collections import Counter

    selected_dist = Counter([sid.sample_type for sid, _ in samples])
    print(f"[INFO] Selected sample_type distribution: {dict(selected_dist)}")

    sample_types_present = sorted({sid.sample_type for sid, _ in samples})
    allowed_issue_focuses = allowed_issue_focuses_from_types(sample_types_present)
    include_action_context = "action_alignment" in allowed_issue_focuses
    train_samples, val_samples = stratified_split(samples, val_ratio=float(args.val_ratio), seed=int(args.seed))

    train_written, train_skipped, train_failed, train_failures, train_written_by_type, train_skipped_by_type, train_failed_by_type = write_jsonl(
        output_path=output_path,
        samples=train_samples,
        include_views=include_views,
        fps=float(args.fps),
        fps_max_frames=int(args.fps_max_frames),
        max_actions=int(args.max_actions),
        allowed_issue_focuses=allowed_issue_focuses,
        include_action_context=include_action_context,
        assistant_label_only=bool(args.assistant_label_only),
    )
    val_written, val_skipped, val_failed, val_failures, val_written_by_type, val_skipped_by_type, val_failed_by_type = write_jsonl(
        output_path=val_output_path,
        samples=val_samples,
        include_views=include_views,
        fps=float(args.fps),
        fps_max_frames=int(args.fps_max_frames),
        max_actions=int(args.max_actions),
        allowed_issue_focuses=allowed_issue_focuses,
        include_action_context=include_action_context,
        assistant_label_only=bool(args.assistant_label_only),
    )

    print("=" * 60)
    print("jsonl 生成完成")
    print(f"输入: {input_dir}")
    print(f"划分: val_ratio={args.val_ratio} seed={args.seed}")
    print(f"训练集: {output_path} 写入={train_written} 跳过={train_skipped} 失败={train_failed}")
    print(f"验证集: {val_output_path} 写入={val_written} 跳过={val_skipped} 失败={val_failed}")
    print(f"训练集写入分布: {train_written_by_type}")
    print(f"训练集跳过分布: {train_skipped_by_type}")
    print(f"训练集失败分布: {train_failed_by_type}")
    print(f"验证集写入分布: {val_written_by_type}")
    print(f"验证集跳过分布: {val_skipped_by_type}")
    print(f"验证集失败分布: {val_failed_by_type}")
    failures = train_failures + val_failures
    failed = train_failed + val_failed
    if failures:
        print("-" * 60)
        for line in failures[:50]:
            print(line)
        if len(failures) > 50:
            print(f"... (剩余 {len(failures) - 50} 条失败记录未展示)")
    print("=" * 60)

    if failed:
        raise RuntimeError("some samples failed")


if __name__ == "__main__":
    main()
