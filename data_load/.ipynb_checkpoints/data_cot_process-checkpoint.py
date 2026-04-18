#!/usr/bin/env python3
"""
为 processed_data 中的每个样本生成 COT（筛选数据的理由与逻辑），并写回 metadata.json。

输入目录默认:  ~/autodl-tmp/data/libero/processed_data
样本目录形如: episode_000000_positive
每个样本包含: metadata.json, videos/main.mp4, videos/wrist.mp4
"""

import argparse
import base64
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils import (
    SampleId,
    extract_json_object,
    iter_samples,
    load_json,
    parse_views,
    summarize_actions,
    write_json,
)

DEFAULT_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

ANALYSIS_GUIDE_BY_TYPE: Dict[str, Dict[str, Any]] = {
    "positive": {
        "goal": "Explain why this sample is high-quality and suitable for VLM fine-tuning (aligned, clear, and explainable).",
        "checklist": [
            "Multi-view frames are clear with natural colors; no obvious color cast / overexposure / underexposure",
            "The instruction matches what is happening in the frames (semantic alignment)",
            "The action summary matches visual changes (e.g., gripper open/close timing aligns with grasp/release)",
            "Action/state statistics look reasonable (gripper transitions exist when expected; motion is smooth without sudden jumps)",
        ],
    },
    "visual": {
        "goal": "Identify visual quality issues (color cast, blur, exposure problems, compression artifacts, etc.) and explain how they harm alignment/training.",
        "checklist": [
            "Check for global color cast, unnatural colors, abnormal saturation/brightness, odd contrast, blur, or noise",
            "Compare views for consistent anomalies (more reliable if both views show the issue)",
            "Explain how the distortion makes recognition/state estimation harder and reduces usability",
        ],
    },
    "gripper": {
        "goal": "Analyze action-vision mismatch, focusing on whether gripper open/close timing matches grasp/place events, and cite evidence.",
        "checklist": [
            "From frames, check whether the gripper appears to grasp/release and whether that conflicts with the gripper signal in the action summary",
            "Explain why this mismatch breaks vision-action alignment and should be considered a quality issue",
        ],
    },
    "instruction": {
        "goal": "Point out mismatches between the instruction and the frames/actions (object/action/target location conflicts) and explain their impact.",
        "checklist": [
            "Check whether the referenced object/container/location in the instruction can be grounded in the frames",
            "If the frames show a different task, state the semantic conflict clearly (object/action/target location)",
            "Explain why instruction-video mismatch misleads multimodal alignment training and should be dropped",
        ],
    },
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate COT for LIBERO processed_data via Qwen API")
    parser.add_argument("--input-dir", type=str, default="~/autodl-tmp/data/libero/processed_data")
    parser.add_argument("--model", type=str, default="qwen3.5-omni-plus")
    parser.add_argument("--base-url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--api-key", type=str, default="sk-ba20e2f0c9a843e8b0a84fd3c5fe5324")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=0, help="本次运行最多处理多少个样本（0 表示不限制）")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--views", type=str, default="main,wrist")
    parser.add_argument(
        "--media-mode",
        type=str,
        default="video",
        choices=["video", "frames"],
        help="输入给API的视觉形式：video=直接传入原始视频（推荐，避免抽帧偏差）；frames=抽帧后以图片列表传入",
    )
    parser.add_argument(
        "--max-video-mb",
        type=int,
        default=256,
        help="media-mode=video 时，本地视频Base64编码的最大文件大小(MB)，超出则报错避免占用过大",
    )
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--max-size", type=int, default=384)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-actions", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=512, help="生成COT的最大token数（越小越精简）")
    parser.add_argument(
        "--thinking",
        type=str,
        default="off",
        choices=["off", "on", "auto"],
        help="是否开启思考模式：off=关闭(更省token/成本), on=开启, auto=不显式指定",
    )
    return parser.parse_args()


def encode_jpeg_to_data_url(img_bgr: Any) -> str:
    import cv2

    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def encode_video_to_data_url(video_path: Path, max_video_mb: int) -> str:
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")
    size_bytes = video_path.stat().st_size
    if max_video_mb and size_bytes > int(max_video_mb) * 1024 * 1024:
        raise ValueError(f"video too large: {video_path} ({size_bytes / (1024*1024):.1f}MB > {max_video_mb}MB)")
    with open(video_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:;base64,{b64}"


def sample_video_frames(
    video_path: Path,
    num_frames: int,
    max_size: int,
    stride: int,
) -> List[str]:
    import cv2

    if num_frames <= 0:
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        total = 0

    wanted = max(1, num_frames)
    if total > 0:
        base_indices = [int(round(i * (total - 1) / max(1, wanted - 1))) for i in range(wanted)]
    else:
        base_indices = list(range(wanted))

    picked_indices = [idx * max(1, stride) for idx in base_indices]

    urls: List[str] = []
    try:
        for idx in picked_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            if max_size and max(frame.shape[0], frame.shape[1]) > max_size:
                h, w = frame.shape[:2]
                if w >= h:
                    new_w = max_size
                    new_h = max(1, int(round(h * (max_size / float(w)))))
                else:
                    new_h = max_size
                    new_w = max(1, int(round(w * (max_size / float(h)))))
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            urls.append(encode_jpeg_to_data_url(frame))
    finally:
        cap.release()

    return urls


def build_prompt(
    sample_id: SampleId,
    metadata: Dict[str, Any],
    views: List[str],
    media_inputs: Dict[str, Any],
    max_actions: int,
    media_mode: str,
) -> List[Dict[str, Any]]:
    instruction = str(metadata.get("instruction", ""))
    actions = metadata.get("actions_sequence", [])
    actions_summary = summarize_actions(actions if isinstance(actions, list) else [], max_actions=max_actions)

    label = 1 if sample_id.sample_type == "positive" else 0
    analysis_guide = ANALYSIS_GUIDE_BY_TYPE.get(sample_id.sample_type, {"goal": "", "checklist": []})

    text = {
        "episode_index": sample_id.episode_index,
        "sample_type": sample_id.sample_type,
        "label": label,
        "instruction": instruction,
        "analysis_guide": analysis_guide,
        "actions_summary": actions_summary,
        "views": views,
        "requirements": {
            "output_format": "JSON_ONLY",
            "fields": {
                "decision": "keep_or_drop",
                "cot": "English. A concise rationale with a clear reasoning chain (you may mention uncertainty). Keep it short.",
                "evidence": "A list of observable evidence items (from frames, instruction, and action statistics).",
                "quality_score": "0-100 overall quality score (alignment + usability).",
            },
            "style": (
                "Do not mention anything about how the data was constructed/perturbed/synthesized, just judge the quality of the data. "
                "Base your judgment only on observable evidence, the actions_summary and the instruction. Write the rationale according to sample_type: "
                "positive explains why it is good; visual/gripper/instruction explains what is wrong and why. "
                "Your final decision must be consistent with label (label=1 => keep, label=0 => drop). "
                "Be specific and evidence-driven. Output a single JSON object only (no extra text, no markdown)."
            ),
        },
    }

    system_msg = {
        "role": "system",
        "content": (
            "You are a data quality inspector for robot manipulation trajectories. "
            "Given multi-view videos (or frames), the task instruction, and an action-sequence summary, "
            "produce a quality-inspection chain-of-thought in a single JSON object. "
            "You must follow sample_type: positive explains why it is good; visual/gripper/instruction identifies the issues and cites evidence. "
            "Do not mention any data construction/perturbation/synthesis. Output JSON only."
        ),
    }

    if media_mode == "video":
        messages: List[Dict[str, Any]] = [system_msg]
        for view in views:
            video_url = media_inputs.get(view)
            if not video_url:
                continue
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "video_url", "video_url": {"url": str(video_url)}},
                        {"type": "text", "text": f"View: {view}"},
                    ],
                }
            )
        messages.append({"role": "user", "content": [{"type": "text", "text": json.dumps(text, ensure_ascii=False)}]})
        return messages

    content: List[Dict[str, Any]] = []
    for view in views:
        for url in media_inputs.get(view, []):
            content.append({"type": "image_url", "image_url": {"url": url}})
    content.append({"type": "text", "text": json.dumps(text, ensure_ascii=False)})
    return [system_msg, {"role": "user", "content": content}]


def generate_cot_via_api(
    model: str,
    base_url: str,
    api_key: str,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    thinking: str,
) -> str:
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError(f"missing dependency: openai ({e})")

    client = OpenAI(api_key=api_key, base_url=base_url)
    extra_body: Dict[str, Any] = {}
    model_lower = str(model).lower()
    if "omni-flash" in model_lower:
        if thinking == "on":
            extra_body["enable_thinking"] = True
        elif thinking == "off":
            extra_body["enable_thinking"] = False

    use_stream = "omni" in model_lower

    def _call(with_extra: bool) -> str:
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if with_extra and extra_body:
            kwargs["extra_body"] = extra_body
        if use_stream:
            kwargs["stream"] = True
            kwargs["stream_options"] = {"include_usage": True}
            kwargs["modalities"] = ["text"]
            acc: List[str] = []
            for chunk in client.chat.completions.create(**kwargs):
                if getattr(chunk, "choices", None):
                    delta = chunk.choices[0].delta
                    part = getattr(delta, "content", None)
                    if part:
                        acc.append(part)
            return "".join(acc).strip()
        completion = client.chat.completions.create(**kwargs)
        return completion.choices[0].message.content or ""

    try:
        return _call(with_extra=True)
    except Exception:
        if extra_body:
            return _call(with_extra=False)
        raise


def get_effective_num_samples(args: argparse.Namespace) -> int:
    if args.num_samples and args.num_samples > 0:
        return int(args.num_samples)
    if args.limit and args.limit > 0:
        return int(args.limit)
    return 0


def validate_args(args: argparse.Namespace, input_dir: Path, views: List[str]) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")
    if not views:
        raise ValueError("views is empty")
    if not args.dry_run and not str(args.api_key).strip():
        raise ValueError("api_key is empty (set --api-key or env DASHSCOPE_API_KEY / OPENAI_API_KEY)")
    if args.media_mode == "video" and not str(args.model).lower().find("omni") >= 0:
        raise ValueError("media-mode=video requires an omni model (e.g., qwen3.5-omni-plus or qwen3-omni-flash)")


def process_one_sample(
    *,
    sample_id: SampleId,
    sample_dir: Path,
    args: argparse.Namespace,
    views: List[str],
    api_key: str,
) -> Tuple[str, Optional[str]]:
    metadata_path = sample_dir / "metadata.json"
    videos_dir = sample_dir / "videos"
    if not metadata_path.exists():
        return "failed", f"{sample_dir.name}: missing metadata.json"
    if not videos_dir.exists():
        return "failed", f"{sample_dir.name}: missing videos/"

    metadata = load_json(metadata_path)
    if (not args.overwrite) and ("cot" in metadata):
        return "skipped", None

    media_inputs: Dict[str, Any] = {}
    if args.media_mode == "video":
        for view in views:
            video_path = videos_dir / f"{view}.mp4"
            media_inputs[view] = encode_video_to_data_url(video_path, max_video_mb=int(args.max_video_mb))
    else:
        for view in views:
            video_path = videos_dir / f"{view}.mp4"
            media_inputs[view] = sample_video_frames(
                video_path=video_path,
                num_frames=args.num_frames,
                max_size=args.max_size,
                stride=args.stride,
            )

    messages = build_prompt(
        sample_id=sample_id,
        metadata=metadata,
        views=views,
        media_inputs=media_inputs,
        max_actions=args.max_actions,
        media_mode=str(args.media_mode),
    )

    if args.dry_run:
        print(json.dumps(messages, ensure_ascii=False)[:4000])
        return "ok", None

    raw = generate_cot_via_api(
        model=args.model,
        base_url=args.base_url,
        api_key=api_key,
        messages=messages,
        max_tokens=args.max_tokens,
        thinking=args.thinking,
    )
    parsed = extract_json_object(raw)
    cot_value = parsed.get("cot") if isinstance(parsed, dict) else None
    if not cot_value:
        cot_value = raw.strip()
    metadata["cot"] = cot_value
    if isinstance(parsed, dict):
        metadata["cot_structured"] = parsed
        if "evidence" in parsed:
            metadata["cot_evidence"] = parsed.get("evidence")
        if "quality_score" in parsed:
            metadata["cot_quality_score"] = parsed.get("quality_score")
        if "decision" in parsed:
            metadata["cot_decision"] = parsed.get("decision")
    metadata["cot_model"] = args.model
    metadata["cot_generated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(metadata_path, metadata)
    return "ok", None


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser()
    views = parse_views(args.views)
    validate_args(args, input_dir, views)

    samples = iter_samples(input_dir)
    num_samples = get_effective_num_samples(args)
    if num_samples and num_samples > 0:
        samples = samples[: num_samples]

    api_key = args.api_key.strip()

    ok = 0
    skipped = 0
    failed = 0
    failures: List[str] = []

    from tqdm import tqdm

    for sample_id, sample_dir in tqdm(samples, desc="Processing samples"):
        try:
            status, failure_msg = process_one_sample(
                sample_id=sample_id,
                sample_dir=sample_dir,
                args=args,
                views=views,
                api_key=api_key,
            )
            if status == "ok":
                ok += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1
                if failure_msg:
                    failures.append(failure_msg)
        except Exception as e:
            failed += 1
            failures.append(f"{sample_dir.name}: {type(e).__name__}: {e}")

    print("=" * 60)
    print("COT生成完成")
    print(f"输入: {input_dir}")
    print(f"总样本: {len(samples)}")
    print(f"成功: {ok}")
    print(f"跳过: {skipped}")
    print(f"失败: {failed}")
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
