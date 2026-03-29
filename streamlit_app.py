# streamlit_app.py
import os
import streamlit as st
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.tools import tool

# ================= 定义一个简单的推荐工具（规则匹配） =================
@tool
def recommend(query: str) -> str:
    """根据用户需求推荐商品"""
    if "耳机" in query:
        return "根据您的偏好，推荐：\n1. 索尼 WH-1000XM5 降噪耳机\n2. 苹果 AirPods Pro 2\n3. 小米 Buds 4 Pro"
    elif "儿童" in query or "字典" in query:
        return "根据您的偏好，推荐：\n1. 猜猜我有多爱你 绘本\n2. 不一样的卡梅拉 系列\n3. 神奇校车 科普绘本"
    else:
        return "根据当前热门商品，推荐：\n1. 华为 MateBook X Pro\n2. 戴森吸尘器 V15\n3. 乐高 创意百变系列"

tools = [recommend]

# ================= 配置 LLM =================
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    st.error("请在 Streamlit Cloud 的 Secrets 中设置 DEEPSEEK_API_KEY")
    st.stop()

model = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
    api_key=api_key,
    temperature=0.7,
    max_tokens=1024,
)

# ================= 系统提示 =================
system_prompt = """你是一个智能电商客服助手，根据用户需求调用推荐工具。只回答推荐相关的内容。"""

# ================= 创建 Agent（使用内存检查点）=================
checkpointer = InMemorySaver()
agent = create_react_agent(
    model=model,
    tools=tools,
    prompt=system_prompt,
    checkpointer=checkpointer,
)

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