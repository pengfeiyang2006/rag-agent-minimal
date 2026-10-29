"""Streamlit 演示界面。

启动（在项目根目录）：streamlit run ui/app.py
侧边栏可切换 Mock 模式（无 Key 演示）与真实模式。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from app import config
from app.llm import LLMClient, MockLLM
from app.react_loop import run_agent

st.set_page_config(page_title="RAG Agent 演示", layout="wide")

st.title("RAG + ReAct Agent 演示")

with st.sidebar:
    st.header("配置")
    use_mock = st.checkbox("Mock 模式（无 Key 演示）", value=config.MOCK_MODE)
    provider = st.selectbox("模型供应商", ["deepseek", "qwen"],
                            index=0 if config.LLM_PROVIDER == "deepseek" else 1)
    model = st.text_input("模型名", config.CHAT_MODEL)
    max_steps = st.slider("最大步数（防死循环）", 1, 10, config.MAX_STEPS)
    if st.button("清空对话"):
        st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

question = st.chat_input("输入问题，例如：什么是MCP？")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    if use_mock:
        llm = MockLLM()
    else:
        base_url = (
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
            if provider == "qwen"
            else config.BASE_URL
        )
        llm = LLMClient(base_url=base_url, model=model)

    with st.chat_message("assistant"):
        with st.spinner("Agent 思考中..."):
            answer, trace = run_agent(question, llm, max_steps=max_steps)
            with st.expander(f"查看思考轨迹（{len(trace)} 步）"):
                for t in trace:
                    st.json(t["message"])
            st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
