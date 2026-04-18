import time
from pathlib import Path

import streamlit as st


DEFAULT_DATASET_PATH = "/root/autodl-tmp/data/libero"
DEFAULT_PROGRESS_SLEEP_SEC = 0.03


def inject_style() -> None:
    st.markdown(
        """
<style>
  .block-container { padding-top: 1.2rem; padding-bottom: 2.2rem; max-width: 1100px; }
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
  .card-title { font-weight: 680; margin: 0 0 0.45rem 0; }
  .row { display: flex; gap: 0.75rem; flex-wrap: wrap; align-items: center; }
  .divider { height: 1px; background: rgba(255,255,255,0.10); margin: 0.9rem 0; }
</style>
""",
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
<div class="hero">
  <div class="hero-title">VLA-Auditor 项目展示</div>
  <div class="hero-sub muted">数据加载与可视化流程引导</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_stepper(active: str) -> None:
    steps = [("mode_select", "选择模式"), ("offline_path", "选择路径"), ("loading", "加载完成")]
    html = "<div class='row' style='margin-top: 0.85rem;'>"
    for key, name in steps:
        cls = "chip on" if key == active else "chip"
        html += f"<span class='{cls}'>{name}</span>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### 操作引导")
        st.write("1. 选择加载方式")
        st.write("2. 设置数据集路径")
        st.write("3. 查看加载进度与统计信息")
        st.markdown("---")
        st.markdown("### 展示设置")
        st.slider(
            "加载速度",
            min_value=0.01,
            max_value=0.08,
            value=float(st.session_state.get("progress_sleep_sec", DEFAULT_PROGRESS_SLEEP_SEC)),
            step=0.01,
            key="progress_sleep_sec",
            help="数值越大，进度条越慢，便于演示过程细节。",
        )


def init_state() -> None:
    st.session_state.setdefault("step", "mode_select")
    st.session_state.setdefault("load_mode", "")
    st.session_state.setdefault("dataset_path", DEFAULT_DATASET_PATH)
    st.session_state.setdefault("last_error", "")
    st.session_state.setdefault("loaded", False)
    st.session_state.setdefault("loaded_summary", {})
    st.session_state.setdefault("loading_done", False)
    st.session_state.setdefault("progress_sleep_sec", DEFAULT_PROGRESS_SLEEP_SEC)


def goto(step: str) -> None:
    st.session_state.step = step
    st.session_state.last_error = ""
    st.rerun()


def page_mode_select() -> None:
    render_hero()
    render_stepper("mode_select")
    st.write("")
    st.markdown("## 数据加载")
    st.caption("选择加载方式开始。")
    st.write("")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("<div class='card'><div class='card-title'>Offline</div><div class='muted'>从本地路径加载数据集</div></div>", unsafe_allow_html=True)
        if st.button("选择 Offline", use_container_width=True):
            st.session_state.load_mode = "offline"
            goto("offline_path")
    with c2:
        st.markdown("<div class='card'><div class='card-title'>Online</div><div class='muted'>在线拉取/连接数据源</div></div>", unsafe_allow_html=True)
        if st.button("选择 Online", use_container_width=True):
            st.session_state.load_mode = "online"
            goto("online_stub")


def page_online_stub() -> None:
    render_hero()
    render_stepper("mode_select")
    st.markdown("## Online 加载")
    st.info("当前版本暂不对接真实在线数据源。")
    st.write("")
    if st.button("返回", use_container_width=True):
        goto("mode_select")


def page_offline_path() -> None:
    render_hero()
    render_stepper("offline_path")
    st.markdown("## Offline 加载")
    st.caption("填写数据集根目录，然后点击“确定并加载”。")
    st.write("")
    p = Path(st.session_state.dataset_path).expanduser()
    exists = p.exists()
    badge = "chip ok" if exists else "chip bad"
    badge_text = "路径可用" if exists else "路径不可用"
    st.markdown(f"<div class='row'><span class='{badge}'>{badge_text}</span><span class='chip'>默认：{DEFAULT_DATASET_PATH}</span></div>", unsafe_allow_html=True)
    st.write("")
    st.text_input("数据集路径", key="dataset_path", help="例如：/root/autodl-tmp/data/libero")
    with st.expander("更多选项", expanded=False):
        st.toggle("显示加载细节", key="show_load_details", value=True)
    if st.session_state.last_error:
        st.error(st.session_state.last_error)

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        if st.button("确定并加载", use_container_width=True):
            p = Path(st.session_state.dataset_path).expanduser()
            if not str(st.session_state.dataset_path).strip():
                st.session_state.last_error = "路径不能为空"
                st.rerun()
            if not p.exists():
                st.session_state.last_error = f"路径不存在：{p}"
                st.rerun()
            st.session_state.loaded = False
            st.session_state.loaded_summary = {}
            st.session_state.loading_done = False
            goto("loading")
    with col2:
        if st.button("返回", use_container_width=True):
            goto("mode_select")


def simulate_loading(p: Path) -> dict:
    processed = p / "processed_data"
    examples = 0
    if processed.exists() and processed.is_dir():
        try:
            examples = sum(1 for _ in processed.iterdir() if _.is_dir())
        except Exception:
            examples = 0
    return {
        "dataset_root": str(p),
        "processed_data": str(processed),
        "examples": int(examples),
        "processed_exists": bool(processed.exists() and processed.is_dir()),
    }


def list_sample_ids(p: Path, limit: int = 12) -> list[str]:
    processed = p / "processed_data"
    if not processed.exists() or not processed.is_dir():
        return [f"sample_{i:04d}" for i in range(1, limit + 1)]
    items = []
    try:
        for child in processed.iterdir():
            if child.is_dir():
                items.append(child.name)
            if len(items) >= limit:
                break
    except Exception:
        return [f"sample_{i:04d}" for i in range(1, limit + 1)]
    if not items:
        return [f"sample_{i:04d}" for i in range(1, limit + 1)]
    return items


def normalize_sample_id(sample_id: str) -> str:
    s = (sample_id or "").strip()
    parts = s.split("_")
    if len(parts) >= 3 and parts[0] == "episode" and parts[1].isdigit():
        return f"{parts[0]}_{parts[1]}"
    return s


def page_loading() -> None:
    render_hero()
    render_stepper("loading")
    st.markdown("## 正在加载数据")
    p = Path(st.session_state.dataset_path).expanduser()
    st.markdown("<div class='card'><div class='card-title'>数据集路径</div></div>", unsafe_allow_html=True)
    st.code(str(p), language=None)
    st.write("")

    if not st.session_state.loading_done:
        progress = st.progress(0, text="初始化…")
        status = st.empty()
        details = st.empty()

        steps = [
            ("扫描目录结构", 18),
            ("加载索引", 42),
            ("构建缓存", 68),
            ("准备可视化资源", 88),
            ("完成", 100),
        ]
        last = 0
        for label, target in steps:
            status.write(label)
            if st.session_state.get("show_load_details", True):
                details.info(f"当前阶段：{label}")
            for v in range(last, target + 1):
                progress.progress(v, text=f"{label} {v}%")
                time.sleep(float(st.session_state.get("progress_sleep_sec", DEFAULT_PROGRESS_SLEEP_SEC)))
            last = target

        st.session_state.loaded_summary = simulate_loading(p)
        st.session_state.loaded = True
        st.session_state.loading_done = True

    st.write("")
    st.success("加载完成")
    summary = st.session_state.loaded_summary or {}
    examples = int(summary.get("examples") or 0)
    processed_exists = bool(summary.get("processed_exists"))
    processed_path = str(summary.get("processed_data") or "")
    t1, t2 = st.tabs(["概览", "目录与示例"])
    with t1:
        m1, m2, m3 = st.columns(3, gap="large")
        with m1:
            st.metric("样本数量", f"{examples:,}" if examples > 0 else "未知")
        with m2:
            st.metric("processed_data", "存在" if processed_exists else "未找到")
        with m3:
            st.metric("加载状态", "就绪")
        st.write("")
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown("#### 关键路径")
        st.code(str(p), language=None)
        if processed_path:
            st.code(processed_path, language=None)

    with t2:
        ids = [normalize_sample_id(x) for x in list_sample_ids(p, limit=12)]
        c1, c2 = st.columns([2, 3], gap="large")
        with c1:
            st.markdown("#### 示例条目")
            st.dataframe([{"sample_id": s} for s in ids], use_container_width=True, height=360)
        with c2:
            st.markdown("#### 目录速览")
            rows = []
            for name in ["processed_data", "raw_data", "cache", "videos"]:
                pp = p / name
                rows.append({"name": name, "exists": "是" if pp.exists() else "否", "path": str(pp)})
            st.dataframe(rows, use_container_width=True, height=360)

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        if st.button("重新选择路径", use_container_width=True):
            goto("offline_path")
    with col2:
        if st.button("返回首页", use_container_width=True):
            goto("mode_select")


def main() -> None:
    st.set_page_config(page_title="VLA-Auditor Demo Frontend", page_icon="VLA-Auditor", layout="centered")
    inject_style()
    init_state()
    render_sidebar()

    step = st.session_state.step
    if step == "mode_select":
        page_mode_select()
    elif step == "online_stub":
        page_online_stub()
    elif step == "offline_path":
        page_offline_path()
    elif step == "loading":
        page_loading()
    else:
        st.session_state.step = "mode_select"
        st.rerun()


if __name__ == "__main__":
    main()
