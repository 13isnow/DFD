#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tb_logger import TBLogger
from tqdm import tqdm


DEFAULT_DATA_ROOT = os.environ.get("DATA_ROOT", "/root/autodl-tmp/data")
DEFAULT_OUTPUT_DIR = str(Path.home() / "autodl-fs" / "output" / "libero" / "SFT_eval")
DEFAULT_CACHE_ROOT = Path.home() / "autodl-fs" / "cache" / "libero_qwenvl"
DEFAULT_MODELSCOPE_CACHE = str(DEFAULT_CACHE_ROOT / "modelscope")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate LIBERO SFT model on val.jsonl via ms-swift inference engine")
    p.add_argument("--adapter_path", type=str, default=None)
    p.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--val_jsonl", type=str, default=str(Path(DEFAULT_DATA_ROOT) / "libero" / "val.jsonl"))
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--gpu_ids", type=str, default="0")

    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument(
        "--force_json_output",
        action="store_true",
        help="在推理时追加格式约束提示，要求模型只输出可解析JSON（推荐开启以稳定评估）",
    )
    p.add_argument(
        "--no_force_json_output",
        action="store_true",
        help="关闭格式约束提示（默认开启格式约束提示）",
    )
    p.add_argument(
        "--max_retries",
        type=int,
        default=2,
        help="当输出不可解析/无法提取label时的重试次数（每次会更强约束输出格式）",
    )

    p.add_argument("--video_max_pixels", type=int, default=80200)
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--fps_max_frames", type=int, default=8)

    p.add_argument("--quant_bits", type=int, default=4)
    p.add_argument("--tb_logdir", type=str, default=None)
    return p.parse_args()


def load_records(jsonl_path: str) -> List[Dict[str, Any]]:
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


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


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    s = (text or "").strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        maybe = s[start : end + 1]
        try:
            obj = json.loads(maybe)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None


def build_eval_suffix(mode: str) -> str:
    if mode == "minimal_json":
        return (
            "Output a single JSON object only (no extra text, no markdown). "
            "Required format:\n"
            '{"decision":"keep|drop","label":0|1,"issue_focus":"ok|image_quality|action_alignment|instruction_alignment"}\n'
            "Keep rationale short and include 1-3 evidence items if you add them."
        )
    if mode == "label_only":
        return (
            "Output a single JSON object only (no extra text, no markdown). "
            'Required format: {"label":0} or {"label":1}.'
        )
    return ""


def infer_text_with_retries(
    *,
    engine: Any,
    infer_request_cls: Any,
    base_messages: List[Dict[str, Any]],
    videos: List[Any],
    images: List[Any],
    req_config: Any,
    force_json_output: bool,
    max_retries: int,
) -> Tuple[str, int]:
    modes: List[str] = []
    if force_json_output:
        modes.append("minimal_json")
    modes.append("none")
    if force_json_output:
        modes.append("label_only")

    attempts = max(int(max_retries), 0) + 1
    used_attempts = 0
    last_text = ""

    for attempt in range(attempts):
        used_attempts = attempt + 1
        mode = modes[min(attempt, len(modes) - 1)]
        messages = list(base_messages)
        suffix = build_eval_suffix(mode)
        if suffix:
            messages.append({"role": "user", "content": suffix})

        infer_req = infer_request_cls(messages=messages, videos=videos, images=images)
        try:
            resp_list = engine.infer([infer_req], request_config=req_config)
            last_text = resp_list[0].choices[0].message.content or ""
        except Exception:
            last_text = ""

        if parse_label_from_text(last_text) is not None:
            return last_text, used_attempts
        if extract_json_object(last_text) is not None:
            return last_text, used_attempts

    return last_text, used_attempts


def parse_label_from_text(text: str) -> Optional[int]:
    obj = extract_json_object(text)
    if isinstance(obj, dict):
        if "label" in obj:
            try:
                return int(obj["label"])
            except Exception:
                pass
        if "decision" in obj:
            d = str(obj["decision"]).strip().lower()
            if d in {"keep", "good", "accept", "pass"}:
                return 1
            if d in {"drop", "bad", "reject", "fail"}:
                return 0

    s = (text or "").lower()
    if "keep" in s:
        return 1
    if "drop" in s:
        return 0
    m = re.search(r"\blabel\b\s*[:=]\s*([01])", s)
    if m:
        return int(m.group(1))
    zh = (text or "")
    if re.search(r"(高质量|优质|可用|保留|推荐保留|通过)", zh):
        if not re.search(r"(不高质量|非高质量|不可用|不适合|不建议保留|不通过)", zh):
            return 1
    if re.search(r"(低质量|不可用|不适合|剔除|丢弃|弃用|不通过|错误|不一致|错配)", zh):
        return 0
    if re.search(r"(high[- ]quality|good quality|high quality|suitable for .*training|can be considered .*high[- ]quality)", s):
        if not re.search(r"(not high|low quality|not suitable|unsuitable|should be dropped|discard)", s):
            return 1
    if re.search(r"(low[- ]quality|poor quality|not suitable|unsuitable|should be dropped|discard|mismatch|misaligned)", s):
        return 0
    return None


def parse_issue_focus_from_text(text: str) -> str:
    obj = extract_json_object(text)
    if isinstance(obj, dict) and "issue_focus" in obj:
        return str(obj["issue_focus"])
    return "unknown"


def compute_binary_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    n = len(y_true)
    correct = sum(int(a == b) for a, b in zip(y_true, y_pred))
    acc = correct / n if n else 0.0

    cm = [[0, 0], [0, 0]]
    for t, p in zip(y_true, y_pred):
        cm[int(t)][int(p)] += 1

    def _f1_for_label(label_id: int) -> float:
        tp = cm[label_id][label_id]
        fp = cm[1 - label_id][label_id]
        fn = cm[label_id][1 - label_id]
        denom_p = tp + fp
        denom_r = tp + fn
        precision = tp / denom_p if denom_p else 0.0
        recall = tp / denom_r if denom_r else 0.0
        denom_f1 = precision + recall
        return (2 * precision * recall / denom_f1) if denom_f1 else 0.0

    f1_0 = _f1_for_label(0)
    f1_1 = _f1_for_label(1)
    f1_macro = (f1_0 + f1_1) / 2

    return {
        "total_samples": n,
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_macro, 4),
        "confusion_matrix": cm,
        "f1_label0": round(f1_0, 4),
        "f1_label1": round(f1_1, 4),
    }


def compute_accuracy(y_true: List[int], y_pred: List[int]) -> float:
    if not y_true:
        return 0.0
    correct = sum(int(a == b) for a, b in zip(y_true, y_pred))
    return correct / len(y_true)


def harmonic_mean(a: float, b: float) -> float:
    denom = a + b
    if denom <= 0:
        return 0.0
    return 2.0 * a * b / denom


def parse_sample_type_from_id(record_id: str) -> str:
    rid = str(record_id or "")
    parts = rid.split("_", 2)
    if len(parts) >= 3:
        return parts[2]
    return "unknown"


def log_metrics(tb: TBLogger, metrics: Dict[str, Any], *, prefix: str) -> None:
    cm = metrics.get("confusion_matrix") or [[0, 0], [0, 0]]
    tb.print("\n" + "=" * 60)
    tb.print(f"{prefix} Evaluation Results")
    tb.print("=" * 60)
    tb.print(f"Total samples   : {metrics.get('total_samples', 0)}")
    tb.print(f"Accuracy        : {metrics.get('accuracy', 0.0):.4f}")
    if "pos_acc" in metrics and "neg_acc" in metrics:
        tb.print(f"Pos acc (label=1) : {metrics.get('pos_acc', 0.0):.4f}")
        tb.print(f"Neg acc (label=0) : {metrics.get('neg_acc', 0.0):.4f}")
        tb.print(f"Balanced acc (arith) : {metrics.get('balanced_acc_arith', 0.0):.4f}")
        tb.print(f"Balanced acc (harm)  : {metrics.get('balanced_acc_harm', 0.0):.4f}")
    tb.print(f"F1 (macro)      : {metrics.get('f1_macro', 0.0):.4f}")
    tb.print(f"F1 (label=0)    : {metrics.get('f1_label0', 0.0):.4f}")
    tb.print(f"F1 (label=1)    : {metrics.get('f1_label1', 0.0):.4f}")
    neg_type_acc = metrics.get("neg_type_accuracy") or {}
    if neg_type_acc:
        tb.print("Negative type accuracy:")
        for k in ["visual", "gripper", "instruction"]:
            if k in neg_type_acc:
                tb.print(f"  {k:<12}: {neg_type_acc[k]:.4f}")
    tb.print("Confusion Matrix (rows=true, cols=pred):")
    tb.print("          0      1")
    tb.print(f"true 0  {cm[0][0]:6d} {cm[0][1]:6d}")
    tb.print(f"true 1  {cm[1][0]:6d} {cm[1][1]:6d}")

    type_rows = metrics.get("confusion_by_type_4way") or []
    if type_rows:
        tb.print("-" * 60)
        tb.print("Confusion by sample_type (rows fixed order):")
        tb.print("type           total     tn     fp     fn     tp    acc")
        for r in type_rows:
            t = str(r.get("type", ""))
            total = int(r.get("total", 0))
            tn = int(r.get("tn", 0))
            fp = int(r.get("fp", 0))
            fn = int(r.get("fn", 0))
            tp = int(r.get("tp", 0))
            acc = float(r.get("accuracy", 0.0))
            tb.print(f"{t:<12} {total:6d} {tn:6d} {fp:6d} {fn:6d} {tp:6d} {acc:6.4f}")
    tb.print("=" * 60)

    tb.add_scalar(f"{prefix}/accuracy", metrics.get("accuracy", 0.0), step=0)
    if "pos_acc" in metrics:
        tb.add_scalar(f"{prefix}/pos_acc", metrics.get("pos_acc", 0.0), step=0)
    if "neg_acc" in metrics:
        tb.add_scalar(f"{prefix}/neg_acc", metrics.get("neg_acc", 0.0), step=0)
    if "balanced_acc_arith" in metrics:
        tb.add_scalar(f"{prefix}/balanced_acc_arith", metrics.get("balanced_acc_arith", 0.0), step=0)
    if "balanced_acc_harm" in metrics:
        tb.add_scalar(f"{prefix}/balanced_acc_harm", metrics.get("balanced_acc_harm", 0.0), step=0)
    tb.add_scalar(f"{prefix}/f1_macro", metrics.get("f1_macro", 0.0), step=0)
    tb.add_scalar(f"{prefix}/f1_label0", metrics.get("f1_label0", 0.0), step=0)
    tb.add_scalar(f"{prefix}/f1_label1", metrics.get("f1_label1", 0.0), step=0)
    tb.add_scalar(f"{prefix}/total_samples", float(metrics.get("total_samples", 0)), step=0)

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tb = TBLogger(args.tb_logdir or (output_dir / "tb_eval"))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    os.environ["VIDEO_MAX_PIXELS"] = str(args.video_max_pixels)
    os.environ["FPS"] = str(args.fps)
    os.environ["FPS_MAX_FRAMES"] = str(args.fps_max_frames)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("MODELSCOPE_CACHE", DEFAULT_MODELSCOPE_CACHE)

    val_path = Path(args.val_jsonl)
    if not val_path.exists():
        tb.print(f"[ERROR] val.jsonl not found: {val_path}")
        tb.close()
        sys.exit(1)

    records = load_records(str(val_path))
    tb.print(f"[INFO] Loaded {len(records)} val samples from {val_path}")

    # swift_root = Path(__file__).parent.parent / "ms-swift"
    # if swift_root.exists() and str(swift_root) not in sys.path:
    #     sys.path.insert(0, str(swift_root))

    from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine

    engine_kwargs: Dict[str, Any] = {"max_batch_size": 1}
    if args.quant_bits and args.quant_bits > 0:
        tb.print(f"[INFO] Using {args.quant_bits}-bit quantization")
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
            tb.print(f"[WARN] Quantization disabled (failed to init BitsAndBytesConfig): {e}")

    if args.adapter_path and Path(args.adapter_path).exists():
        resolved_adapter = resolve_adapter_dir(args.adapter_path)
        ckpt_has_config = (Path(resolved_adapter) / "config.json").exists()
        if ckpt_has_config:
            tb.print(f"[INFO] Loading merged checkpoint: {resolved_adapter}")
            engine = TransformersEngine(resolved_adapter, **engine_kwargs)
        else:
            tb.print(f"[INFO] LoRA checkpoint detected: {resolved_adapter}")
            tb.print(f"[INFO] Base model: {args.base_model}")
            engine = TransformersEngine(args.base_model, adapters=[resolved_adapter], **engine_kwargs)
    else:
        if args.adapter_path:
            tb.print(f"[WARN] Checkpoint not found: {args.adapter_path}, using base model.")
        tb.print(f"[INFO] Loading base model: {args.base_model}")
        engine = TransformersEngine(args.base_model, **engine_kwargs)

    req_config = RequestConfig(
        max_tokens=int(args.max_new_tokens),
        temperature=0.0,
    )

    infer_out_path = output_dir / "val_infer.jsonl"
    y_true: List[int] = []
    y_pred: List[int] = []
    group_true: Dict[str, List[int]] = defaultdict(list)
    group_pred: Dict[str, List[int]] = defaultdict(list)
    unknown_pred = 0
    type_true: Dict[str, List[int]] = defaultdict(list)
    type_pred: Dict[str, List[int]] = defaultdict(list)
    pos_true: List[int] = []
    pos_pred: List[int] = []
    neg_true: List[int] = []
    neg_pred: List[int] = []

    id_counts = Counter()
    for r in records:
        rid = str(r.get("id", ""))
        if rid:
            suffix = rid.split("_", 2)[-1]
            id_counts[suffix] += 1

    tb.print(f"[INFO] Sample type distribution: {dict(id_counts)}")

    with open(infer_out_path, "w", encoding="utf-8") as fout:
        for i, record in tqdm(enumerate(records), total=len(records)):
            messages = record.get("messages") or []
            gt_text = ""
            user_messages: List[Dict[str, Any]] = []
            for m in messages:
                if m.get("role") == "assistant":
                    gt_text = str(m.get("content", ""))
                else:
                    user_messages.append(m)

            gt_label = parse_label_from_text(gt_text)  
            gt_focus = parse_issue_focus_from_text(gt_text)
            if gt_label is None:
                raise ValueError(f"[ERROR] sample {i} gt_label is None: {gt_text}")

            force_json_output = bool(args.force_json_output) or (not bool(args.no_force_json_output))
            pred_text, used_attempts = infer_text_with_retries(
                engine=engine,
                infer_request_cls=InferRequest,
                base_messages=user_messages,
                videos=record.get("videos", []),
                images=record.get("images", []),
                req_config=req_config,
                force_json_output=force_json_output,
                max_retries=int(args.max_retries),
            )

            # print("pred_text:", pred_text, end="\t")
            pred_label = parse_label_from_text(pred_text)
            if pred_label is None:
                unknown_pred += 1
                pred_label = 0
            
            # # temp
            # if int(int(gt_label) == 1):
            #     pred_label = 1 - int(pred_label)
                
            y_true.append(int(gt_label))
            y_pred.append(int(pred_label))
            group_true[gt_focus].append(int(gt_label))
            group_pred[gt_focus].append(int(pred_label))

            rid = str(record.get("id", ""))
            sample_type = parse_sample_type_from_id(rid)
            type_true[sample_type].append(int(gt_label))
            type_pred[sample_type].append(int(pred_label))
            if int(gt_label) == 1:
                pos_true.append(int(gt_label))
                pos_pred.append(int(pred_label))
            else:
                neg_true.append(int(gt_label))
                neg_pred.append(int(pred_label))

            out = {
                "id": record.get("id", ""),
                "videos": record.get("videos", []),
                "ground_truth": gt_text,
                "pred_text": pred_text,
                "gt_label": int(gt_label),
                "pred_label": int(pred_label),
                "issue_focus": gt_focus,
                "used_attempts": int(used_attempts),
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

    metrics = compute_binary_metrics(y_true, y_pred)
    metrics["unknown_pred"] = unknown_pred
    metrics["by_issue_focus"] = {}
    for k in sorted(group_true.keys()):
        metrics["by_issue_focus"][k] = compute_binary_metrics(group_true[k], group_pred[k])

    pos_acc = compute_accuracy(pos_true, pos_pred)
    neg_acc = compute_accuracy(neg_true, neg_pred)
    metrics["pos_acc"] = round(pos_acc, 4)
    metrics["neg_acc"] = round(neg_acc, 4)
    metrics["balanced_acc_arith"] = round((pos_acc + neg_acc) / 2.0, 4)
    metrics["balanced_acc_harm"] = round(harmonic_mean(pos_acc, neg_acc), 4)

    neg_type_acc: Dict[str, float] = {}
    for k in ["visual", "gripper", "instruction"]:
        neg_type_acc[k] = round(compute_accuracy(type_true.get(k, []), type_pred.get(k, [])), 4)
    metrics["neg_type_accuracy"] = neg_type_acc
    metrics["by_sample_type"] = {}
    for k in sorted(type_true.keys()):
        metrics["by_sample_type"][k] = compute_binary_metrics(type_true[k], type_pred[k])

    ordered_types = [
        ("positive", "Positive"),
        ("visual", "Visual"),
        ("gripper", "Gripper"),
        ("instruction", "Instruction"),
    ]
    type_rows = []
    for key, label in ordered_types:
        m = metrics["by_sample_type"].get(key) or compute_binary_metrics(type_true.get(key, []), type_pred.get(key, []))
        cm_t = m.get("confusion_matrix") or [[0, 0], [0, 0]]
        total = int(m.get("total_samples", 0) or 0)
        tn, fp = int(cm_t[0][0]), int(cm_t[0][1])
        fn, tp = int(cm_t[1][0]), int(cm_t[1][1])
        acc = (tn + tp) / total if total else 0.0
        type_rows.append(
            {
                "type": label,
                "key": key,
                "total": total,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "accuracy": round(float(acc), 4),
            }
        )
    metrics["confusion_by_type_4way"] = type_rows

    log_metrics(tb, metrics, prefix="eval")
    tb.print(f"[INFO] Unparseable preds fallback: {unknown_pred}")
    tb.print(f"[INFO] Inference results saved to: {infer_out_path}")

    metrics_path = output_dir / "val_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    tb.print(f"[INFO] Metrics saved to: {metrics_path}")
    tb.close()


if __name__ == "__main__":
    main()
