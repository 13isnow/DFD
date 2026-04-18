import json
import importlib.util
import sys
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
  .block-container { padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1400px; }
  header[data-testid="stHeader"] { display: none; }
  [data-testid="stToolbar"] { display: none; }
  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }
  h1, h2, h3 { letter-spacing: 0.2px; }
  .muted { color: rgba(255,255,255,0.65); font-size: 0.95rem; }
  .chip {
    display: inline-block;
    padding: 0.15rem 0.55rem;
    border-radius: 999px;
    font-size: 0.85rem;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.12);
    margin-right: 0.35rem;
  }
  .chip.ok { background: rgba(69,209,138,0.12); border-color: rgba(69,209,138,0.22); }
  .chip.bad { background: rgba(255,77,121,0.12); border-color: rgba(255,77,121,0.22); }
  .card {
    padding: 0.85rem 0.95rem;
    border-radius: 14px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.10);
    box-shadow: 0 10px 30px rgba(0,0,0,0.20);
  }
  .section-title { margin: 0.15rem 0 0.55rem 0; font-weight: 650; }
  .stExpander { border-radius: 14px; overflow: hidden; }
</style>
""",
        unsafe_allow_html=True,
    )


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
    st.set_page_config(page_title="LIBERO VLM Evaluation Viewer", layout="wide")
    inject_style()
    if "idx" not in st.session_state:
        st.session_state.idx = 0
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
        st.markdown("#### Instruction")
        st.info(instruction if instruction else "(empty)")
        prev, nxt = st.columns(2)
        if prev.button("← 上一个"):
            if st.session_state.idx > 0:
                st.session_state.idx -= 1
        if nxt.button("下一个 →"):
            if st.session_state.idx + 1 < len(records):
                st.session_state.idx += 1
    with right:
        pred = extract_json_object(rec.get("pred_text", "")) or extract_json_object(rec.get("ground_truth", "")) or {}
        decision = pred.get("decision", "")
        issue = pred.get("issue_focus", "")
        score = pred.get("quality_score", None)
        st.markdown("<h3 class='section-title'>Model Output</h3>", unsafe_allow_html=True)
        ok = str(decision).lower() in {"keep", "accept", "pass", "good"} or int(pred.get("label", 0) or 0) == 1
        badge_cls = "ok" if ok else "bad"
        st.markdown(
            f"<span class='chip {badge_cls}'>Decision: {decision or 'N/A'}</span>"
            f"<span class='chip'>Issue: {issue or 'N/A'}</span>"
            f"<span class='chip'>Score: {'' if score is None else score}</span>",
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown("#### Action Summary")
        with st.expander("Action Summary（展开查看）", expanded=False):
            st.text_area("action_summary", value=json.dumps(action_summary, ensure_ascii=False, indent=2), height=260)
        with st.expander("Full actions_sequence（展开查看完整动作序列）", expanded=False):
            st.code(json.dumps(actions_full, ensure_ascii=False), language="json")
        st.write("")
        st.markdown("#### Evidence")
        for e in pred.get("evidence", []):
            st.write(f"- {e}")
        if pred:
            with st.expander("Parsed JSON（展开查看）", expanded=False):
                st.code(json.dumps(pred, ensure_ascii=False, indent=2), language="json")


if __name__ == "__main__":
    main()
