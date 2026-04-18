#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from tb_logger import TBLogger
from SFT_eval import compute_accuracy, compute_binary_metrics, harmonic_mean, load_records, log_metrics, parse_issue_focus_from_text, parse_label_from_text, parse_sample_type_from_id

DEFAULT_DATA_ROOT = os.environ.get("DATA_ROOT", "/root/autodl-tmp/data")
DEFAULT_OUTPUT_DIR = str(Path.home() / "autodl-fs" / "output" / "libero" / "CLIP_eval")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate CLIP baselines on LIBERO jsonl")
    p.add_argument("--mode", choices=["zero_shot", "linear_probe", "fine_tune"], default="zero_shot")
    p.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32")
    p.add_argument("--train_jsonl", type=str, default=str(Path(DEFAULT_DATA_ROOT) / "libero" / "train.jsonl"))
    p.add_argument("--val_jsonl", type=str, default=str(Path(DEFAULT_DATA_ROOT) / "libero" / "val.jsonl"))
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--gpu_ids", type=str, default="0")
    p.add_argument("--num_frames", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tb_logdir", type=str, default=None)
    p.add_argument("--positive_prompt", type=str, default="a high-quality robot manipulation training sample")
    p.add_argument("--negative_prompt", type=str, default="a low-quality robot manipulation training sample")
    p.add_argument("--decision_threshold", type=float, default=0.5, help="Predict label=1 when prob_pos >= threshold for trained heads")
    return p.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_media(jsonl_path, media_path):
    p = Path(media_path)
    if p.exists():
        return str(p)
    p2 = Path(jsonl_path).parent / media_path
    return str(p2 if p2.exists() else p)


def sample_video_frames(video_path, num_frames):
    try:
        import cv2
    except Exception as e:
        raise RuntimeError(f"opencv-python is required: {e}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    idxs = np.linspace(0, total - 1, num=min(num_frames, total), dtype=int)
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames sampled: {video_path}")
    return frames


def load_visuals(record, jsonl_path, num_frames):
    if record.get("videos"):
        return sample_video_frames(resolve_media(jsonl_path, str(record["videos"][0])), num_frames)
    images = []
    for p in (record.get("images") or [])[:num_frames]:
        images.append(Image.open(resolve_media(jsonl_path, str(p))).convert("RGB"))
    if not images:
        raise RuntimeError(f"No visuals for sample: {record.get('id', '')}")
    return images


def get_gt(record):
    gt_text = ""
    for m in record.get("messages") or []:
        if m.get("role") == "assistant":
            gt_text = str(m.get("content", ""))
    gt_label = parse_label_from_text(gt_text)
    if gt_label is None:
        raise ValueError(f"gt_label is None: {gt_text}")
    return int(gt_label), parse_issue_focus_from_text(gt_text), gt_text


def _extract_image_features(model, pixel_values):
    feats = model.get_image_features(pixel_values=pixel_values)
    if not isinstance(feats, torch.Tensor):
        if hasattr(feats, "image_embeds") and feats.image_embeds is not None:
            feats = feats.image_embeds
        elif hasattr(feats, "pooler_output") and feats.pooler_output is not None:
            feats = feats.pooler_output
            if feats.shape[-1] == model.visual_projection.weight.shape[1]:
                feats = model.visual_projection(feats)
        elif hasattr(feats, "last_hidden_state") and feats.last_hidden_state is not None:
            feats = feats.last_hidden_state[:, 0, :]
            if feats.shape[-1] == model.visual_projection.weight.shape[1]:
                feats = model.visual_projection(feats)
        else:
            raise TypeError(f"Unsupported image feature output type: {type(feats)!r}")
    feats = feats / feats.norm(dim=-1, keepdim=True)
    feats = feats.mean(dim=0)
    return feats / feats.norm(dim=-1, keepdim=True)


def encode_record(model, processor, record, jsonl_path, num_frames, device, *, requires_grad=False, return_cpu=True):
    images = load_visuals(record, jsonl_path, num_frames)
    pixel_values = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
    if requires_grad:
        feats = _extract_image_features(model, pixel_values)
    else:
        with torch.no_grad():
            feats = _extract_image_features(model, pixel_values)
    if return_cpu:
        feats = feats.detach().cpu()
    return feats


class Head(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 2)
    def forward(self, x):
        return self.fc(x)


@torch.no_grad()
def build_text_feats(model, processor, args, device):
    x = processor(text=[args.negative_prompt, args.positive_prompt], return_tensors="pt", padding=True).to(device)
    feats = model.get_text_features(**x)
    if not isinstance(feats, torch.Tensor):
        if hasattr(feats, "text_embeds") and feats.text_embeds is not None:
            feats = feats.text_embeds
        elif hasattr(feats, "pooler_output") and feats.pooler_output is not None:
            feats = feats.pooler_output
            if feats.shape[-1] == model.text_projection.weight.shape[1]:
                feats = model.text_projection(feats)
        elif hasattr(feats, "last_hidden_state") and feats.last_hidden_state is not None:
            feats = feats.last_hidden_state[:, 0, :]
            if feats.shape[-1] == model.text_projection.weight.shape[1]:
                feats = model.text_projection(feats)
        else:
            raise TypeError(f"Unsupported text feature output type: {type(feats)!r}")
    return feats / feats.norm(dim=-1, keepdim=True)


@torch.no_grad()
def build_feature_bank(model, processor, records, jsonl_path, args, device):
    feats, labels = [], []
    for r in tqdm(records, desc="extract_features"):
        feats.append(encode_record(model, processor, r, jsonl_path, args.num_frames, device, requires_grad=False, return_cpu=True))
        labels.append(get_gt(r)[0])
    return torch.stack(feats), torch.tensor(labels, dtype=torch.long)


def build_class_weights(labels):
    counts = torch.bincount(labels.long(), minlength=2).float()
    total = counts.sum().clamp_min(1.0)
    weights = total / (counts * len(counts)).clamp_min(1.0)
    return weights


def train_linear_probe(train_feats, train_labels, args, device, tb):
    ds = TensorDataset(train_feats.float(), train_labels.long())
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    head = Head(train_feats.shape[-1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    class_weights = build_class_weights(train_labels).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    tb.print(f"[INFO] linear_probe class_weights: neg={class_weights[0].item():.4f}, pos={class_weights[1].item():.4f}")
    for epoch in range(args.epochs):
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(head(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * y.size(0)
        tb.add_scalar("linear_probe/train_loss", total / max(len(ds), 1), step=epoch)
    return head.eval()


def train_fine_tune(model, processor, records, args, device, tb):
    for p in model.text_model.parameters():
        p.requires_grad = False
    head = Head(model.config.projection_dim).to(device)
    opt = torch.optim.AdamW(list(model.vision_model.parameters()) + list(model.visual_projection.parameters()) + list(head.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_labels = torch.tensor([get_gt(r)[0] for r in records], dtype=torch.long)
    class_weights = build_class_weights(train_labels).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    tb.print(f"[INFO] fine_tune class_weights: neg={class_weights[0].item():.4f}, pos={class_weights[1].item():.4f}")
    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for i in tqdm(range(0, len(records), args.batch_size), desc=f"fine_tune_epoch_{epoch}"):
            batch = records[i:i + args.batch_size]
            xs, ys = [], []
            for r in batch:
                xs.append(encode_record(model, processor, r, args.train_jsonl, args.num_frames, device, requires_grad=True, return_cpu=False))
                ys.append(get_gt(r)[0])
            x = torch.stack(xs)
            y = torch.tensor(ys, dtype=torch.long, device=device)
            loss = loss_fn(head(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * y.size(0)
        tb.add_scalar("fine_tune/train_loss", total / max(len(records), 1), step=epoch)
    model.eval()
    return head.eval()


@torch.no_grad()
def predict(model, processor, record, args, device, text_feats=None, head=None):
    feat = encode_record(model, processor, record, args.val_jsonl, args.num_frames, device, requires_grad=False, return_cpu=False)
    if args.mode == "zero_shot":
        logits = (feat @ text_feats.T).float()
        pred = int(torch.argmax(logits).item())
        return pred, {"score_neg": float(logits[0].detach().cpu()), "score_pos": float(logits[1].detach().cpu())}
    logits = head(feat.unsqueeze(0)).squeeze(0).float()
    probs = torch.softmax(logits, dim=-1)
    prob_pos = float(probs[1].detach().cpu())
    pred = int(prob_pos >= args.decision_threshold)
    return pred, {"prob_neg": float(probs[0].detach().cpu()), "prob_pos": prob_pos, "decision_threshold": args.decision_threshold}


def evaluate(records, args, tb, model, processor, device, text_feats=None, head=None):
    out_path = Path(args.output_dir) / f"val_infer_{args.mode}.jsonl"
    y_true, y_pred = [], []
    group_true, group_pred = defaultdict(list), defaultdict(list)
    type_true, type_pred = defaultdict(list), defaultdict(list)
    pos_true, pos_pred, neg_true, neg_pred = [], [], [], []
    with open(out_path, "w", encoding="utf-8") as fout:
        for record in tqdm(records, desc=f"eval_{args.mode}"):
            gt_label, gt_focus, gt_text = get_gt(record)
            pred_label, detail = predict(model, processor, record, args, device, text_feats, head)
            y_true.append(gt_label); y_pred.append(pred_label)
            group_true[gt_focus].append(gt_label); group_pred[gt_focus].append(pred_label)
            sample_type = parse_sample_type_from_id(str(record.get("id", "")))
            type_true[sample_type].append(gt_label); type_pred[sample_type].append(pred_label)
            if gt_label == 1:
                pos_true.append(gt_label); pos_pred.append(pred_label)
            else:
                neg_true.append(gt_label); neg_pred.append(pred_label)
            fout.write(json.dumps({"id": record.get("id", ""), "videos": record.get("videos", []), "ground_truth": gt_text, "pred_label": pred_label, "gt_label": gt_label, "issue_focus": gt_focus, "mode": args.mode, "detail": detail}, ensure_ascii=False) + "\n")
    metrics = compute_binary_metrics(y_true, y_pred)
    metrics["unknown_pred"] = 0
    metrics["by_issue_focus"] = {k: compute_binary_metrics(group_true[k], group_pred[k]) for k in sorted(group_true.keys())}
    pos_acc, neg_acc = compute_accuracy(pos_true, pos_pred), compute_accuracy(neg_true, neg_pred)
    metrics["pos_acc"] = round(pos_acc, 4); metrics["neg_acc"] = round(neg_acc, 4)
    metrics["balanced_acc_arith"] = round((pos_acc + neg_acc) / 2.0, 4)
    metrics["balanced_acc_harm"] = round(harmonic_mean(pos_acc, neg_acc), 4)
    metrics["neg_type_accuracy"] = {k: round(compute_accuracy(type_true.get(k, []), type_pred.get(k, [])), 4) for k in ["visual", "gripper", "instruction"]}
    metrics["by_sample_type"] = {k: compute_binary_metrics(type_true[k], type_pred[k]) for k in sorted(type_true.keys())}
    rows = []
    for key, label in [("positive", "Positive"), ("visual", "Visual"), ("gripper", "Gripper"), ("instruction", "Instruction")]:
        m = metrics["by_sample_type"].get(key) or compute_binary_metrics(type_true.get(key, []), type_pred.get(key, []))
        cm, total = m.get("confusion_matrix") or [[0, 0], [0, 0]], int(m.get("total_samples", 0) or 0)
        rows.append({"type": label, "key": key, "total": total, "tn": int(cm[0][0]), "fp": int(cm[0][1]), "fn": int(cm[1][0]), "tp": int(cm[1][1]), "accuracy": round(((int(cm[0][0]) + int(cm[1][1])) / total) if total else 0.0, 4)})
    metrics["confusion_by_type_4way"] = rows
    log_metrics(tb, metrics, prefix=args.mode)
    metrics_path = Path(args.output_dir) / f"val_metrics_{args.mode}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    tb.print(f"[INFO] Inference results saved to: {out_path}")
    tb.print(f"[INFO] Metrics saved to: {metrics_path}")


def main():
    args = parse_args()
    set_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tb = TBLogger(args.tb_logdir or (out_dir / f"tb_{args.mode}"))
    need = [args.val_jsonl] + ([args.train_jsonl] if args.mode != "zero_shot" else [])
    for p in need:
        if not Path(p).exists():
            tb.print(f"[ERROR] jsonl not found: {p}")
            tb.close(); sys.exit(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(args.clip_model).to(device)
    processor = CLIPProcessor.from_pretrained(args.clip_model)
    val_records = load_records(args.val_jsonl)
    tb.print(f"[INFO] Loaded {len(val_records)} val samples from {args.val_jsonl}")
    tb.print(f"[INFO] Sample type distribution: {dict(Counter(str(r.get('id', '')).split('_', 2)[-1] for r in val_records if r.get('id')))}")
    text_feats = head = None
    if args.mode == "zero_shot":
        model.eval(); text_feats = build_text_feats(model, processor, args, device)
    else:
        train_records = load_records(args.train_jsonl)
        tb.print(f"[INFO] Loaded {len(train_records)} train samples from {args.train_jsonl}")
        if args.mode == "linear_probe":
            model.eval()
            train_feats, train_labels = build_feature_bank(model, processor, train_records, args.train_jsonl, args, device)
            head = train_linear_probe(train_feats, train_labels, args, device, tb)
        else:
            head = train_fine_tune(model, processor, train_records, args, device, tb)
    evaluate(val_records, args, tb, model, processor, device, text_feats, head)
    tb.close()


if __name__ == "__main__":
    main()
