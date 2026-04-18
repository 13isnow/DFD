#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import copy
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypeVar

EPISODE_DIR_RE = re.compile(r"^episode_(\d{6})_([a-zA-Z0-9_-]+)$")

T = TypeVar("T")

ACTION_SUMMARY_SEMANTICS_TEMPLATE: Dict[str, Any] = {
    "dimensions": {
        "dx": "End-effector translation along +X per step",
        "dy": "End-effector translation along +Y per step",
        "dz": "End-effector translation along +Z per step",
        "droll": "End-effector rotation (roll) per step, radians",
        "dpitch": "End-effector rotation (pitch) per step, radians",
        "dyaw": "End-effector rotation (yaw) per step, radians",
        "gripper": "Gripper opening scalar (interpreted as 0=closed, 1=open); delta>0 means opening",
    },
    "coordinate_frame": "Assumed robot-base right-handed frame: +X forward, +Y left, +Z up; rotations follow right-hand rule (roll-pitch-yaw).",
    "windowing": "Sequence partitioned into up to {k} contiguous windows of equal length; each window reports mean values.",
    "norms_definition": "pos_norm=sqrt(dx^2+dy^2+dz^2), rot_norm=sqrt(droll^2+dpitch^2+dyaw^2).",
    "low_activity_rule": "translation/rotation reported as 'low' if the window norm ≤ global median (p50) of that norm.",
    "magnitude_buckets": "Per-dimension magnitude buckets: small ≤ p50(abs), medium ≤ p90(abs), large > p90(abs) computed across the full sequence.",
    "gripper_events": "Gripper event indices mark step-to-step changes with |Δ| ≥ max(0.02, 0.2*range, 0.5*p95(|Δ|)).",
}

def _build_action_semantics(k: int) -> Dict[str, Any]:
    sem = copy.deepcopy(ACTION_SUMMARY_SEMANTICS_TEMPLATE)
    sem["windowing"] = sem.get("windowing", "").format(k=int(max(k, 0)))
    return sem


@dataclass(frozen=True)
class SampleId:
    episode_index: int
    sample_type: str


def parse_sample_dir_name(name: str) -> Optional[SampleId]:
    match = EPISODE_DIR_RE.match(name)
    if not match:
        return None
    return SampleId(episode_index=int(match.group(1)), sample_type=match.group(2))


def iter_sample_dirs(input_dir: Path) -> List[Tuple[SampleId, Path]]:
    samples: List[Tuple[SampleId, Path]] = []
    for p in input_dir.iterdir():
        if not p.is_dir():
            continue
        sid = parse_sample_dir_name(p.name)
        if sid is None:
            continue
        samples.append((sid, p))
    samples.sort(key=lambda x: (x[0].episode_index, x[0].sample_type, x[1].name))
    return samples


def iter_samples(input_dir: Path) -> List[Tuple[SampleId, Path]]:
    return iter_sample_dirs(input_dir)


def slice_by_limit(items: List[T], limit: int) -> List[T]:
    if limit and limit > 0:
        return items[:limit]
    return items


def parse_views(views_raw: str) -> List[str]:
    views = [v.strip() for v in (views_raw or "").split(",")]
    return [v for v in views if v]


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


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


def summarize_actions(actions: Any, max_actions: int) -> Dict[str, Any]:
    max_actions = int(max_actions) if max_actions is not None else 0
    max_actions = max(max_actions, 0)

    dims = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]

    def _to_7d(a: Any) -> Optional[List[float]]:
        if not isinstance(a, list):
            if isinstance(a, tuple):
                a = list(a)
            elif hasattr(a, "tolist"):
                try:
                    a = a.tolist()
                except Exception:
                    return None
        if not isinstance(a, list) or len(a) < 7:
            return None
        try:
            v = [float(x) for x in a[:7]]
        except Exception:
            return None
        if any(math.isnan(x) or math.isinf(x) for x in v):
            return None
        return v

    seq: List[List[float]] = []
    if not isinstance(actions, list):
        if isinstance(actions, tuple):
            actions = list(actions)
        elif hasattr(actions, "tolist"):
            try:
                actions = actions.tolist()
            except Exception:
                actions = []
    if isinstance(actions, list):
        for a in actions:
            v = _to_7d(a)
            if v is not None:
                seq.append(v)

    if not seq:
        return {
            "schema_version": "action_summary_v1",
            "num_steps": 0,
            "dims": dims,
            "stats": {},
            "windows": [],
            "text_en": "",
            "semantics": _build_action_semantics(0),
        }

    n = len(seq)

    def _quantile(sorted_vals: List[float], q: float) -> float:
        if not sorted_vals:
            return 0.0
        if q <= 0:
            return float(sorted_vals[0])
        if q >= 1:
            return float(sorted_vals[-1])
        pos = (len(sorted_vals) - 1) * q
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return float(sorted_vals[lo])
        w = pos - lo
        return float(sorted_vals[lo] * (1.0 - w) + sorted_vals[hi] * w)

    def _stats(vals: List[float]) -> Dict[str, float]:
        if not vals:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0}
        m = sum(vals) / float(len(vals))
        var = sum((x - m) * (x - m) for x in vals) / float(len(vals))
        s = math.sqrt(var)
        sv = sorted(vals)
        return {
            "mean": float(m),
            "std": float(s),
            "min": float(sv[0]),
            "max": float(sv[-1]),
            "p50": _quantile(sv, 0.50),
            "p90": _quantile(sv, 0.90),
            "p95": _quantile(sv, 0.95),
        }

    by_dim: List[List[float]] = [[] for _ in range(7)]
    pos_norm: List[float] = []
    rot_norm: List[float] = []
    for v in seq:
        for i in range(7):
            by_dim[i].append(v[i])
        dx, dy, dz, droll, dpitch, dyaw, _g = v
        pos_norm.append(math.sqrt(dx * dx + dy * dy + dz * dz))
        rot_norm.append(math.sqrt(droll * droll + dpitch * dpitch + dyaw * dyaw))

    abs_quantiles: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(dims):
        av = sorted(abs(x) for x in by_dim[i])
        abs_quantiles[name] = {
            "p50": _quantile(av, 0.50),
            "p90": _quantile(av, 0.90),
            "p95": _quantile(av, 0.95),
        }

    def _bucket(name: str, value: float) -> str:
        q = abs_quantiles.get(name) or {}
        t1 = float(q.get("p50", 0.0))
        t2 = float(q.get("p90", t1))
        v = abs(value)
        if v <= max(t1, 0.0):
            return "small"
        if v <= max(t2, t1):
            return "medium"
        return "large"

    def _direction(value: float) -> str:
        if value > 0:
            return "+"
        if value < 0:
            return "-"
        return "0"

    g = by_dim[6]
    g_min = min(g)
    g_max = max(g)
    g_range = float(g_max - g_min)
    g_delta = float(g[-1] - g[0]) if len(g) >= 2 else 0.0
    g_step_diffs = [g[i + 1] - g[i] for i in range(len(g) - 1)]
    g_step_abs = sorted(abs(x) for x in g_step_diffs)
    g_step_p95 = _quantile(g_step_abs, 0.95) if g_step_abs else 0.0
    g_event_thr = max(0.02, 0.2 * g_range, 0.5 * g_step_p95)
    g_events = [i + 1 for i, dg in enumerate(g_step_diffs) if abs(dg) >= g_event_thr]

    stats = {dims[i]: _stats(by_dim[i]) for i in range(7)}
    norms = {"pos_norm": _stats(pos_norm), "rot_norm": _stats(rot_norm)}
    pos_low_thr = float(norms["pos_norm"]["p50"])
    rot_low_thr = float(norms["rot_norm"]["p50"])
    eps = 1e-12

    if max_actions <= 0:
        return {
            "schema_version": "action_summary_v1",
            "num_steps": n,
            "dims": dims,
            "stats": {k: {kk: round(vv, 6) for kk, vv in sv.items()} for k, sv in stats.items()},
            "norms": {k: {kk: round(vv, 6) for kk, vv in sv.items()} for k, sv in norms.items()},
            "gripper": {
                "min": round(float(g_min), 6),
                "max": round(float(g_max), 6),
                "mean": round(float(stats["gripper"]["mean"]), 6),
                "delta": round(float(g_delta), 6),
                "event_count": int(len(g_events)),
                "event_indices": g_events[:10],
            },
            "windows": [],
            "text_en": "",
            "semantics": _build_action_semantics(0),
        }

    k = min(max_actions, n)
    win_len = int(math.ceil(n / float(k)))

    windows: List[Dict[str, Any]] = []
    text_lines_en: List[str] = []

    for start in range(0, n, win_len):
        end = min(n, start + win_len)
        seg = seq[start:end]
        if not seg:
            continue

        seg_means: Dict[str, float] = {}
        seg_abs_means: Dict[str, float] = {}
        for i, name in enumerate(dims):
            vs = [v[i] for v in seg]
            m = sum(vs) / float(len(vs))
            seg_means[name] = float(m)
            seg_abs_means[name] = float(sum(abs(x) for x in vs) / float(len(vs)))

        dxm, dym, dzm = seg_means["dx"], seg_means["dy"], seg_means["dz"]
        drm, dpm, dym2 = seg_means["droll"], seg_means["dpitch"], seg_means["dyaw"]
        pm = math.sqrt(dxm * dxm + dym * dym + dzm * dzm)
        rm = math.sqrt(drm * drm + dpm * dpm + dym2 * dym2)
        g_start = seg[0][6]
        g_end = seg[-1][6]
        g_d = float(g_end - g_start)

        trans_candidates = [("dx", seg_abs_means["dx"]), ("dy", seg_abs_means["dy"]), ("dz", seg_abs_means["dz"])]
        rot_candidates = [("droll", seg_abs_means["droll"]), ("dpitch", seg_abs_means["dpitch"]), ("dyaw", seg_abs_means["dyaw"])]
        trans_candidates.sort(key=lambda x: x[1], reverse=True)
        rot_candidates.sort(key=lambda x: x[1], reverse=True)

        trans_top = trans_candidates[:2]
        rot_top = rot_candidates[:2]

        if pm <= pos_low_thr:
            trans_desc_en = "translation: low"
        else:
            parts = [f"{name}{_direction(seg_means[name])}({_bucket(name, seg_means[name])})" for name, _v in trans_top if abs(seg_means[name]) > eps]
            trans_desc_en = "translation: " + (", ".join(parts) if parts else "mixed")

        if rm <= rot_low_thr:
            rot_desc_en = "rotation: low"
        else:
            parts = [f"{name}{_direction(seg_means[name])}({_bucket(name, seg_means[name])})" for name, _v in rot_top if abs(seg_means[name]) > eps]
            rot_desc_en = "rotation: " + (", ".join(parts) if parts else "mixed")

        if abs(g_d) >= g_event_thr:
            gripper_desc_en = f"gripper: {_direction(g_d)}({round(g_d, 4)})"
        else:
            gripper_desc_en = "gripper: stable"

        line_en = f"[{start}-{end - 1}] {trans_desc_en}; {rot_desc_en}; {gripper_desc_en}"

        windows.append(
            {
                "start": int(start),
                "end": int(end - 1),
                "mean": {k: round(float(v), 6) for k, v in seg_means.items()},
                "pos_norm_mean": round(float(pm), 6),
                "rot_norm_mean": round(float(rm), 6),
                "gripper_delta": round(float(g_d), 6),
                "text_en": line_en,
            }
        )
        text_lines_en.append(line_en)

        if len(windows) >= k:
            break

    semantics = _build_action_semantics(k)

    return {
        "schema_version": "action_summary_v1",
        "num_steps": n,
        "dims": dims,
        "stats": {k: {kk: round(vv, 6) for kk, vv in sv.items()} for k, sv in stats.items()},
        "norms": {k: {kk: round(vv, 6) for kk, vv in sv.items()} for k, sv in norms.items()},
        "gripper": {
            "min": round(float(g_min), 6),
            "max": round(float(g_max), 6),
            "mean": round(float(stats["gripper"]["mean"]), 6),
            "delta": round(float(g_delta), 6),
            "event_threshold": round(float(g_event_thr), 6),
            "event_count": int(len(g_events)),
            "event_indices": g_events[:10],
        },
        "windows": windows,
        "text_en": "\n".join(text_lines_en),
        "semantics": semantics,
    }
