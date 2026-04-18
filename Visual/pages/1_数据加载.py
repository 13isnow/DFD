import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo_frontend import main as load_main

load_main()

st.write("")
st.markdown("---")
c1, c2 = st.columns(2, gap="medium")
if c1.button("返回入口", use_container_width=True):
    st.switch_page("/app_hub.py")
if c2.button(
    "下一步：推理分析",
    type="primary",
    use_container_width=True,
    disabled=not bool(st.session_state.get("loading_done")),
):
    st.switch_page("pages/2_推理分析.py")
