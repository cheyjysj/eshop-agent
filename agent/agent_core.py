# agent/agent_core.py
import os
import sys
import sqlite3
from typing import List, Tuple
from dotenv import load_dotenv

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI


# 确保能导入模块
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from knowledge_base.hybrid_retriever import HybridRetriever
from agent.tools import KnowledgeBaseTool, RecommendTool

load_dotenv()

def create_agent_instance() -> Tuple:
    """
    初始化混合检索器、工具、LLM、系统提示、SQLite 检查点，并创建 LangGraph Agent。
    返回 (agent, conn) 供外部使用，conn 用于最后关闭。
    """
    # ================= 1. 初始化检索器 =================
    CHROMA_PERSIST_DIR = "./chroma_db_rag"
    COLLECTION_NAME = "Eshopping_rag_collection"

    if not os.path.exists(CHROMA_PERSIST_DIR):
        base_dir = os.path.dirname(os.path.dirname(__file__))
        CHROMA_PERSIST_DIR = os.path.join(base_dir, "chroma_db_rag")

    retriever = HybridRetriever(CHROMA_PERSIST_DIR, COLLECTION_NAME, use_rerank=True)

    # ================= 2. 创建工具列表 =================
    tools: List = [
        KnowledgeBaseTool(retriever),
        RecommendTool()
    ]

    # ================= 3. 配置 LLM =================
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    if not DEEPSEEK_API_KEY:
        raise ValueError("请设置环境变量 DEEPSEEK_API_KEY")

    model = ChatOpenAI(
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
        api_key=DEEPSEEK_API_KEY,
        temperature=0.7,
        max_tokens=1024,
    )

    # ================= 4. 系统提示 =================
    system_prompt = """你是一个智能电商客服助手，你的任务是根据用户的问题，调用合适的工具来回答。
    你可以使用以下工具：
    {tools}
    
    工具名称: {tool_names}
    
    请严格按照以下格式回答：
    Question: 用户输入的问题
    Thought: 思考是否需要调用工具
    Action: 工具名称（如果需要）
    Action Input: 工具输入参数
    Observation: 工具返回结果
    ...（可以重复上述步骤）
    Thought: 我现在可以回答用户了
    Final Answer: 最终的友好回答
    只回答原始数据有的物品，如果没有则回答抱歉
    开始！"""

    # ================= 5. 创建 SQLite 持久化检查点 =================
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints.db")
    db_dir = os.path.dirname(db_path)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)

    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    # ================= 6. 创建 Agent =================
    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=system_prompt,
        checkpointer=checkpointer
    )

    return agent, conn