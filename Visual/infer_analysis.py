import random
import time

import altair as alt
import streamlit as st


DEFAULT_TOTAL_SAMPLES = 40
DEFAULT_SECONDS_PER_SAMPLE = 0.5


def inject_style() -> None:
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
  .card {
    padding: 0.95rem 1.0rem;
    border-radius: 16px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.10);
    box-shadow: 0 12px 30px rgba(0,0,0,0.18);
  }
  .card-title { font-weight: 680; margin: 0 0 0.55rem 0; }
  .divider { height: 1px; background: rgba(255,255,255,0.10); margin: 0.9rem 0; }
  .big-num { font-size: 2.2rem; font-weight: 760; line-height: 1.05; margin: 0.35rem 0 0.15rem 0; }
  .tiny { font-size: 0.9rem; color: rgba(255,255,255,0.65); }
</style>
""",
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
<div class="hero">
  <div class="hero-title">推理分析</div>
  <div class="hero-sub muted">进度追踪与得分趋势</div>
</div>
""",
        unsafe_allow_html=True,
    )


def init_state() -> None:
    st.session_state.setdefault("scores", [])
    st.session_state.setdefault("cur_idx", 0)
    st.session_state.setdefault("seed", 7)
    st.session_state.setdefault("total_samples", DEFAULT_TOTAL_SAMPLES)
    st.session_state.setdefault("seconds_per_sample", DEFAULT_SECONDS_PER_SAMPLE)
    st.session_state.setdefault("infer_done", False)


def build_score_chart(scores: list[float]) -> alt.Chart:
    data = [{"i": i + 1, "score": float(s)} for i, s in enumerate(scores)]
    base = (
        alt.Chart(alt.Data(values=data))
        .encode(
            x=alt.X("i:Q", title="Sample", axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("score:Q", title="Score", scale=alt.Scale(domain=[0, 100])),
            tooltip=[alt.Tooltip("i:Q", title="Sample"), alt.Tooltip("score:Q", title="Score", format=".1f")],
        )
        .properties(height=360)
    )
    line = base.mark_line(color="#8B7BFF", strokeWidth=2.2)
    pts = base.mark_circle(color="#D6D1FF", size=45, opacity=0.9)
    ref = (
        alt.Chart(alt.Data(values=[{"y": 60.0}]))
        .mark_rule(color="rgba(255,255,255,0.55)", strokeDash=[6, 6], strokeWidth=1.5)
        .encode(y="y:Q")
    )
    return (line + pts + ref).interactive()


def generate_scores(total: int, rng: random.Random) -> list[float]:
    total = int(total)
    if total <= 0:
        return []
    high = int(round(total * 0.4))
    above_60 = int(round(total * 0.8))
    mid = max(0, above_60 - high)
    low = max(0, total - high - mid)
    if high + mid + low != total:
        low = max(0, total - high - mid)

    out: list[float] = []
    for _ in range(high):
        out.append(float(rng.uniform(90.0, 100.0)))
    for _ in range(mid):
        out.append(float(rng.uniform(60.0, 89.8)))
    for _ in range(low):
        out.append(float(rng.uniform(25.0, 59.8)))

    rng.shuffle(out)
    return out


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### 参数")
        st.number_input("样本数量", min_value=5, max_value=300, value=int(st.session_state.total_samples), step=5, key="total_samples")
        st.slider(
            "每个样本耗时（秒）",
            min_value=0.1,
            max_value=1.2,
            value=float(st.session_state.seconds_per_sample),
            step=0.1,
            key="seconds_per_sample",
        )
        st.number_input("随机种子", min_value=0, max_value=9999, value=int(st.session_state.seed), step=1, key="seed")
        st.markdown("---")
        st.markdown("### 操作")
        st.caption("开始后会按固定节奏更新进度与折线图。")


def page_main() -> None:
    render_hero()
    st.write("")
    st.markdown("<span class='chip on'>推理中</span><span class='chip'>得分趋势</span>", unsafe_allow_html=True)
    st.write("")

    left, right = st.columns([2, 3], gap="large")
    with left:
        st.markdown("<div class='card'><div class='card-title'>进度</div></div>", unsafe_allow_html=True)
        pwrap = st.container()
        st.write("")
        c_l, c_m, c_r = st.columns([1, 4, 1])
        with c_m:
            progress_ph = st.empty()
            state_ph = st.empty()
            big_ph = st.empty()
            btn_row = st.columns(2, gap="medium")
            start = btn_row[0].button("开始", use_container_width=True)
            reset = btn_row[1].button("重置", use_container_width=True)

    with right:
        st.markdown("<div class='card'><div class='card-title'>得分曲线</div></div>", unsafe_allow_html=True)
        st.write("")
        chart_ph = st.empty()
        metric_row = st.columns(3, gap="large")
        m_cur = metric_row[0].empty()
        m_avg = metric_row[1].empty()
        m_hit = metric_row[2].empty()

    if reset:
        st.session_state.scores = []
        st.session_state.cur_idx = 0
        st.session_state.infer_done = False
        st.rerun()

    scores: list[float] = list(st.session_state.scores or [])
    total = int(st.session_state.total_samples)
    cur = int(st.session_state.cur_idx)

    def _render() -> None:
        ratio = 0.0 if total <= 0 else min(max(cur / total, 0.0), 1.0)
        progress_ph.progress(int(ratio * 100), text=f"{cur}/{total}")
        big_ph.markdown(f"<div class='big-num'>{cur}</div><div class='tiny'>已处理样本</div>", unsafe_allow_html=True)
        if scores:
            chart_ph.altair_chart(build_score_chart(scores), use_container_width=True)
            cur_score = float(scores[-1])
            avg = float(sum(scores) / len(scores))
            hit = int(sum(1 for s in scores if float(s) >= 60.0))
            m_cur.metric("最新得分", f"{cur_score:.1f}")
            m_avg.metric("平均得分", f"{avg:.1f}")
            m_hit.metric("≥ 60", f"{hit}/{len(scores)}")
        else:
            chart_ph.altair_chart(build_score_chart([]), use_container_width=True)
            m_cur.metric("最新得分", "—")
            m_avg.metric("平均得分", "—")
            m_hit.metric("≥ 60", "—")

    _render()

    if start:
        rng = random.Random(int(st.session_state.seed))
        planned = generate_scores(total, rng)
        scores: list[float] = []
        st.session_state.scores = []
        st.session_state.cur_idx = 0
        for i in range(total):
            time.sleep(float(st.session_state.seconds_per_sample))
            scores.append(float(planned[i]))
            st.session_state.scores = list(scores)
            st.session_state.cur_idx = i + 1
            cur = i + 1
            _render()
        st.session_state.infer_done = True
        st.success("推理完成")


def main() -> None:
    st.set_page_config(page_title="VLA-Auditor Inference Analysis", page_icon="VLA-Auditor", layout="wide")
    inject_style()
    init_state()
    render_sidebar()
    page_main()


if __name__ == "__main__":
    main()
