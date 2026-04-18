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

if not bool(st.session_state.get("infer_done")):
    st.warning("请先完成推理分析步骤。")
    if st.button("前往：推理分析", type="primary", use_container_width=True):
        st.switch_page("pages/2_推理分析.py")
    st.stop()

from surface_streamlit import main as view_main

view_main()

st.write("")
st.markdown("---")
if st.button("上一步：推理分析", use_container_width=True):
    st.switch_page("pages/2_推理分析.py")
