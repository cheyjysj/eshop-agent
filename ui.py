# ui.py
import streamlit as st
import requests

# 后端 API 地址（本地测试用，部署时改为 Render 公网地址）
API_URL = "http://localhost:8000/chat"

st.set_page_config(page_title="电商客服助手", page_icon="🤖")
st.title("🤖 智能电商客服助手")

# 会话管理
if "session_id" not in st.session_state:
    st.session_state.session_id = "user_001"

if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 用户输入
if prompt := st.chat_input("请输入您的问题"):
    # 显示用户消息
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 调用后端 API
    try:
        with st.spinner("思考中..."):
            resp = requests.post(API_URL, json={
                "message": prompt,
                "session_id": st.session_state.session_id
            })
            if resp.status_code == 200:
                reply = resp.json()["reply"]
            else:
                reply = f"错误 {resp.status_code}: {resp.text}"
    except Exception as e:
        reply = f"连接后端失败: {e}"

    # 显示助手回复
    st.chat_message("assistant").write(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})