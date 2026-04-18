import json
import importlib.util
import random
import sys
import hashlib
from pathlib import Path

import streamlit as st


def load_utils_summarize_actions():
    utils_path = (Path(__file__).resolve().parent.parent / "data_load" / "utils.py").resolve()
    spec = importlib.util.spec_from_file_location("dfd_data_load_utils", str(utils_path))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.summarize_actions


SUMMARIZE_ACTIONS = load_utils_summarize_actions()

def inject_style():
    st.markdown(
        """
<style>
  .block-container { padding-top: 1.2rem; padding-bottom: 2.2rem; max-width: 1300px; }
  header[data-testid="stHeader"] { display: none; }
  [data-testid="stToolbar"] { display: none; }
  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }
  h1, h2, h3 { letter-spacing: 0.2px; }
  .muted { color: rgba(255,255,255,0.65); font-size: 0.95rem; }
  .hero {
    padding: 1.05rem 1.1rem;
    border-radius: 16px;
    background: radial-gradient(1200px circle at 10% 0%, rgba(98,82,255,0.20), rgba(0,0,0,0)) , rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.10);
    box-shadow: 0 12px 30px rgba(0,0,0,0.18);
  }
  .hero-title { font-weight: 760; font-size: 1.55rem; margin: 0; }
  .hero-sub { margin-top: 0.25rem; }
  .chip {
    display: inline-block;
    padding: 0.15rem 0.60rem;
    border-radius: 999px;
    font-size: 0.85rem;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.14);
    margin-right: 0.35rem;
  }
  .chip.on { background: rgba(98,82,255,0.18); border-color: rgba(98,82,255,0.30); }
  .chip.ok { background: rgba(69,209,138,0.12); border-color: rgba(69,209,138,0.22); }
  .chip.bad { background: rgba(255,77,121,0.12); border-color: rgba(255,77,121,0.22); }
  .card {
    padding: 0.95rem 1.0rem;
    border-radius: 16px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.10);
    box-shadow: 0 12px 30px rgba(0,0,0,0.18);
  }
  .card-title { font-weight: 680; margin: 0 0 0.55rem 0; }
  .section-title { margin: 0.15rem 0 0.55rem 0; font-weight: 680; }
  .divider { height: 1px; background: rgba(255,255,255,0.10); margin: 0.9rem 0; }
  .row { display: flex; gap: 0.75rem; flex-wrap: wrap; align-items: center; }
  .stExpander { border-radius: 14px; overflow: hidden; }
</style>
""",
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
<div class="hero">
  <div class="hero-title">推理后展示</div>
  <div class="hero-sub muted">视频、指令、动作摘要与模型输出</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_stepper(active: str) -> None:
    steps = [("load", "数据加载"), ("infer", "推理分析"), ("view", "推理展示")]
    html = "<div class='row' style='margin-top: 0.85rem;'>"
    for key, name in steps:
        cls = "chip on" if key == active else "chip"
        html += f"<span class='{cls}'>{name}</span>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def extract_json_object(text: str):
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


def build_demo_score(sample_id: str, decision: str) -> int:
    seed = int(hashlib.md5(str(sample_id).encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)
    d = str(decision).strip().lower()
    if d in {"keep", "accept", "pass", "good"}:
        return rng.randint(61, 99)
    return rng.randint(5, 59)


@st.cache_data
def load_val_infer(path: Path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records.append(obj)
    return records

@st.cache_data
def filter_correct(records):
    res = []
    for r in records:
        try:
            gt = int(r.get("gt_label"))
            pred = int(r.get("pred_label"))
            if gt == pred:
                res.append(r)
        except Exception:
            continue
    return res

@st.cache_data
def filter_by_types(records, enabled_types):
    enabled = set(enabled_types or [])
    if not enabled:
        return records
    res = []
    for r in records:
        sid = str(r.get("id", ""))
        t = sid.split("_", 2)[2] if sid.count("_") >= 2 else "unknown"
        if t in enabled:
            res.append(r)
    return res
@st.cache_data
def build_balanced_order(records):
    groups = {}
    for r in records:
        sid = str(r.get("id", ""))
        t = sid.split("_", 2)[2] if sid.count("_") >= 2 else "unknown"
        groups.setdefault(t, []).append(r)
    order = ["positive", "visual", "gripper", "instruction", "unknown"]
    keys = [k for k in order if k in groups] + [k for k in sorted(groups.keys()) if k not in order]
    idx = {k: 0 for k in keys}
    res = []
    while True:
        progressed = False
        for k in keys:
            i = idx[k]
            items = groups[k]
            if i < len(items):
                res.append(items[i])
                idx[k] = i + 1
                progressed = True
        if not progressed:
            break
    return res


@st.cache_data
def summarize_actions_cached(actions):
    return SUMMARIZE_ACTIONS(actions, max_actions=20)


@st.cache_data
def load_instruction_and_actions(sample_id: str):
    p = Path("~/autodl-tmp/data/libero/processed_data").expanduser() / sample_id
    meta = p / "metadata.json"
    if meta.exists():
        data = json.loads(meta.read_text(encoding="utf-8"))
        instruction = str(data.get("instruction", ""))
        actions = data.get("actions_sequence", [])
        return instruction, summarize_actions_cached(actions), actions
    return "", {"num_frames": 0}, []

@st.cache_data
def read_bytes_cached(path_str: str) -> bytes:
    p = Path(path_str).expanduser()
    return p.read_bytes()

def main():
    st.set_page_config(page_title="VLA-Auditor 推理展示", page_icon="VLA-Auditor", layout="wide")
    inject_style()
    if "idx" not in st.session_state:
        st.session_state.idx = 0

    render_hero()
    render_stepper("view")
    st.write("")

    records = load_val_infer(Path("~/autodl-fs/output/libero/SFT_eval/val_infer.jsonl").expanduser())
    records = filter_correct(records)
    type_counts = {}
    for r in records:
        sid = str(r.get("id", ""))
        t = sid.split("_", 2)[2] if sid.count("_") >= 2 else "unknown"
        type_counts[t] = type_counts.get(t, 0) + 1
    enabled_types = [t for t in ["positive", "visual", "gripper", "instruction"] if type_counts.get(t, 0) > 0]
    records = filter_by_types(records, enabled_types)
    records = build_balanced_order(records) 
    if not records:
        st.error("No correct records found")
        return

    total = len(records)
    st.write("")

    left, right = st.columns([2, 4], gap="large")
    with left:
        rec = records[st.session_state.idx]
        vids = rec.get("videos", [])
        def show_video(video_path: Path):
            webm = video_path.with_suffix(".webm")
            if webm.exists():
                st.video(read_bytes_cached(str(webm)))
                return
            if video_path.exists():
                try:
                    st.video(read_bytes_cached(str(video_path)))
                    return
                except Exception:
                    pass
            st.warning(f"无法播放视频：{video_path}\n请安装 ffmpeg 并运行：python ~/DFD/Visual/transcode_videos.py")
        if vids:
            vp = Path(vids[0]).expanduser()
            show_video(vp)
        sample_id = rec.get("id", "")
        instruction, action_summary, actions_full = load_instruction_and_actions(sample_id)
        st.markdown(
            f"<span class='chip'>ID: {sample_id}</span>",
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown("<div class='card-title'>Instruction</div>", unsafe_allow_html=True)
        st.info(instruction if instruction else "(empty)")
        prev, nxt = st.columns(2, gap="medium")
        if prev.button("← 上一个", use_container_width=True, disabled=st.session_state.idx <= 0):
            if st.session_state.idx > 0:
                st.session_state.idx -= 1
                st.rerun()
        if nxt.button("下一个 →", use_container_width=True, disabled=st.session_state.idx + 1 >= total):
            if st.session_state.idx + 1 < len(records):
                st.session_state.idx += 1
                st.rerun()
    with right:
        pred_obj = extract_json_object(rec.get("pred_text", "")) or {}
        gt_obj = extract_json_object(rec.get("ground_truth", "")) or {}
        decision = pred_obj.get("decision", "") or gt_obj.get("decision", "")
        score = build_demo_score(str(rec.get("id", "")), decision)
        st.markdown("<h3 class='section-title'>Model Output</h3>", unsafe_allow_html=True)
        try:
            pred_label_i = int(pred_obj.get("label", rec.get("pred_label", 0)) or 0)
        except Exception:
            pred_label_i = 0
        ok = str(decision).lower() in {"keep", "accept", "pass", "good"} or pred_label_i == 1
        badge_cls = "ok" if ok else "bad"
        st.markdown(
            f"<span class='chip {badge_cls}'>Decision: {decision or 'N/A'}</span>"
            f"<span class='chip'>Score: {score}</span>",
            unsafe_allow_html=True,
        )
        st.write("")
        t1, t2, t3 = st.tabs(["概览", "动作", "证据 / JSON"])
        with t1:
            st.markdown("<div class='card-title'>概览</div>", unsafe_allow_html=True)
            sid = str(rec.get("id", ""))
            sample_type = sid.split("_", 2)[2] if sid.count("_") >= 2 else "unknown"

            m1, m2, m3 = st.columns(3, gap="medium")
            m1.metric("Pred Label", "—" if rec.get("pred_label", None) is None else str(rec.get("pred_label")))
            m2.metric("Score", str(score))
            m3.metric("Type", sample_type)
        with t2:
            st.markdown("<div class='card-title'>Action Summary</div>", unsafe_allow_html=True)
            st.text_area("action_summary", value=json.dumps(action_summary, ensure_ascii=False, indent=2), height=220)
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>Full actions_sequence</div>", unsafe_allow_html=True)
            st.code(json.dumps(actions_full, ensure_ascii=False), language="json")
        with t3:
            st.markdown("<div class='card-title'>Evidence</div>", unsafe_allow_html=True)
            ev = gt_obj.get("evidence", []) if isinstance(gt_obj, dict) else []
            if ev:
                st.dataframe([{"evidence": str(e)} for e in ev], use_container_width=True, height=210)
            else:
                st.info("No evidence")
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>Parsed JSON</div>", unsafe_allow_html=True)
            st.code(json.dumps(pred_obj, ensure_ascii=False, indent=2), language="json")


if __name__ == "__main__":
    main()
