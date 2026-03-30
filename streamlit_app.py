# streamlit_app.py
import os
import sys
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# 确保能导入 agent 模块
sys.path.append(os.path.dirname(__file__))
from agent.agent_core import create_agent_instance

load_dotenv()

# 创建 Agent 实例（根据环境变量自动选择检索方式）
agent, conn = create_agent_instance()

# ================= Streamlit UI =================
st.set_page_config(page_title="电商客服助手", page_icon="🤖")
st.title("🤖 智能电商客服助手")

if "session_id" not in st.session_state:
    st.session_state.session_id = "user_001"
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("请输入您的问题"):
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    config = {"configurable": {"thread_id": st.session_state.session_id}}
    with st.spinner("思考中..."):
        response = agent.invoke(
            {"messages": [HumanMessage(content=prompt)]},
            config=config
        )
        reply = response["messages"][-1].content

    st.chat_message("assistant").write(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})