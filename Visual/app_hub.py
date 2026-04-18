import streamlit as st


def main() -> None:
    st.set_page_config(page_title="VLA-Auditor 展示系统", page_icon="VLA-Auditor", layout="centered")
    st.markdown("## VLA-Auditor 项目展示流程")
    st.caption("请按顺序完成：数据加载 → 推理分析 → 推理展示")
    st.write("")
    st.info("点击下方按钮开始第一步。后续页面会提供“下一步”跳转。")
    if st.button("开始：数据加载", type="primary", use_container_width=True):
        st.switch_page("pages/1_数据加载.py")


if __name__ == "__main__":
    main()
