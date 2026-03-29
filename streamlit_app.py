# streamlit_app.py
import os
import sys
import sqlite3
import streamlit as st
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# 确保能导入模块
sys.path.append(os.path.dirname(__file__))
from knowledge_base.hybrid_retriever import HybridRetriever
from agent.tools import KnowledgeBaseTool, RecommendTool

load_dotenv()

# ================= 初始化 Agent（使用缓存，避免重复加载） =================
@st.cache_resource
def get_agent():
    # 1. 检索器
    CHROMA_PERSIST_DIR = "./chroma_db_rag"
    COLLECTION_NAME = "Eshopping_rag_collection"
    # 注意：部署时向量库可能不存在，需要处理（见下文）
    if os.path.exists(CHROMA_PERSIST_DIR):
        retriever = HybridRetriever(CHROMA_PERSIST_DIR, COLLECTION_NAME, use_rerank=True)
        tools = [KnowledgeBaseTool(retriever), RecommendTool()]
    else:
        # 如果没有向量库，只使用推荐工具
        tools = [RecommendTool()]

    # 2. LLM
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

    # 3. 系统提示
    system_prompt = """你是一个智能电商客服助手..."""

    # 4. 检查点
    conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    # 5. 创建 Agent
    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=system_prompt,
        checkpointer=checkpointer,
    )
    return agent, conn

agent, conn = get_agent()

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