#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_CACHE_ROOT = Path.home() / "autodl-fs" / "cache" / "libero_qwenvl"
DEFAULT_MODELSCOPE_CACHE = str(DEFAULT_CACHE_ROOT / "modelscope")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-sample VLM inference sanity check (ms-swift TransformersEngine)")
    p.add_argument("--adapter_path", type=str, default=None)
    p.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--gpu_ids", type=str, default="0")

    p.add_argument("--prompt", type=str, default="")
    p.add_argument("--system", type=str, default="")
    p.add_argument("--video", type=str, action="append", default=[])
    p.add_argument("--image", type=str, action="append", default=[])

    p.add_argument("--jsonl", type=str, default="")
    p.add_argument("--index", type=int, default=0)

    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--quant_bits", type=int, default=0)

    p.add_argument("--video_max_pixels", type=int, default=160400)
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--fps_max_frames", type=int, default=8)
    return p.parse_args()


def resolve_adapter_dir(adapter_path: str) -> str:
    p = Path(adapter_path)
    if not p.exists():
        return adapter_path
    if p.is_file():
        return adapter_path
    if (p / "adapter_model.safetensors").exists():
        return str(p)
    checkpoint_dirs = []
    for child in p.iterdir():
        if child.is_dir() and child.name.startswith("checkpoint-"):
            checkpoint_dirs.append(child)
    if not checkpoint_dirs:
        return str(p)

    def _ckpt_key(d: Path) -> int:
        suffix = d.name.split("checkpoint-", 1)[-1]
        try:
            return int(suffix)
        except Exception:
            return -1

    best = max(checkpoint_dirs, key=_ckpt_key)
    return str(best)


def load_record_from_jsonl(jsonl_path: str, index: int) -> Dict[str, Any]:
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"jsonl not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i != index:
                continue
            line = line.strip()
            if not line:
                break
            return json.loads(line)
    raise IndexError(f"index out of range: index={index}, jsonl={path}")


def build_messages_from_record(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    messages = record.get("messages") or []
    user_messages: List[Dict[str, Any]] = []
    for m in messages:
        if m.get("role") == "assistant":
            continue
        user_messages.append(m)
    return user_messages


def build_single_turn_messages(system_text: str, user_text: str) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": user_text})
    return messages


def ensure_swift_on_path() -> None:
    swift_root = Path(__file__).resolve().parent / "ms-swift"
    if swift_root.exists() and str(swift_root) not in sys.path:
        sys.path.insert(0, str(swift_root))


def main() -> None:
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    os.environ["VIDEO_MAX_PIXELS"] = str(args.video_max_pixels)
    os.environ["FPS"] = str(args.fps)
    os.environ["FPS_MAX_FRAMES"] = str(args.fps_max_frames)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("MODELSCOPE_CACHE", DEFAULT_MODELSCOPE_CACHE)

    ensure_swift_on_path()

    from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine

    engine_kwargs: Dict[str, Any] = {"max_batch_size": 1}
    if args.quant_bits and args.quant_bits > 0:
        try:
            import torch
            from transformers import BitsAndBytesConfig

            if args.quant_bits == 4:
                engine_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            elif args.quant_bits == 8:
                engine_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        except Exception as e:
            print(f"[WARN] quantization disabled: {e}")

    engine: Any
    if args.adapter_path:
        if Path(args.adapter_path).exists():
            resolved_adapter = resolve_adapter_dir(args.adapter_path)
            ckpt_has_config = (Path(resolved_adapter) / "config.json").exists()
            if ckpt_has_config:
                print(f"[INFO] Loading merged checkpoint: {resolved_adapter}")
                engine = TransformersEngine(resolved_adapter, **engine_kwargs)
            else:
                print(f"[INFO] LoRA checkpoint detected: {resolved_adapter}")
                print(f"[INFO] Base model: {args.base_model}")
                engine = TransformersEngine(args.base_model, adapters=[resolved_adapter], **engine_kwargs)
        else:
            print(f"[WARN] adapter_path not found: {args.adapter_path}, using base model.")
            print(f"[INFO] Loading base model: {args.base_model}")
            engine = TransformersEngine(args.base_model, **engine_kwargs)
    else:
        print(f"[INFO] Loading base model: {args.base_model}")
        engine = TransformersEngine(args.base_model, **engine_kwargs)

    req_config = RequestConfig(max_tokens=int(args.max_new_tokens), temperature=float(args.temperature))

    videos: List[str] = []
    images: List[str] = []
    messages: List[Dict[str, Any]]

    if args.jsonl:
        record = load_record_from_jsonl(args.jsonl, int(args.index))
        videos = [str(v) for v in (record.get("videos") or [])]
        images = [str(v) for v in (record.get("images") or [])]
        messages = build_messages_from_record(record)
        if args.system:
            messages = [{"role": "system", "content": args.system}, *messages]
        if args.prompt:
            messages = list(messages)
            messages.append({"role": "user", "content": args.prompt})
    else:
        videos = [str(Path(v).expanduser()) for v in (args.video or [])]
        images = [str(Path(v).expanduser()) for v in (args.image or [])]
        prompt = args.prompt.strip()
        if not prompt:
            raise ValueError("prompt is empty (use --prompt or provide --jsonl)")
        messages = build_single_turn_messages(args.system.strip(), prompt)

    infer_req = InferRequest(messages=messages, videos=videos, images=images)
    resp_list = engine.infer([infer_req], request_config=req_config)
    text = resp_list[0].choices[0].message.content
    print(text)


if __name__ == "__main__":
    main()
