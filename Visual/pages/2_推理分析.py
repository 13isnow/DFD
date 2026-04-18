import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if not bool(st.session_state.get("loading_done")):
    st.warning("请先完成数据加载步骤。")
    if st.button("前往：数据加载", type="primary", use_container_width=True):
        st.switch_page("pages/1_数据加载.py")
    st.stop()

from infer_analysis import main as infer_main

infer_main()

st.write("")
st.markdown("---")
c1, c2 = st.columns(2, gap="medium")
if c1.button("上一步：数据加载", use_container_width=True):
    st.switch_page("pages/1_数据加载.py")
if c2.button(
    "下一步：推理展示",
    type="primary",
    use_container_width=True,
    disabled=not bool(st.session_state.get("infer_done")),
):
    st.switch_page("pages/3_推理展示.py")
