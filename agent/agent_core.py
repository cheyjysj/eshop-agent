# agent/agent_core.py
import os
import sys
import sqlite3
from typing import List, Tuple
from dotenv import load_dotenv

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# 确保能导入模块
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# 根据环境变量决定使用本地检索还是 Pinecone 检索
USE_PINECONE = os.getenv("USE_PINECONE", "false").lower() == "true"

if not USE_PINECONE:
    # 本地模式：导入 ChromaDB 检索器
    from knowledge_base.hybrid_retriever import HybridRetriever
    from agent.tools import KnowledgeBaseTool, RecommendTool
else:
    # Pinecone 模式：导入 pinecone 和 requests
    import pinecone
    import requests
    from agent.tools import RecommendTool  # 保留规则推荐

load_dotenv()

def create_agent_instance() -> Tuple:
    """
    初始化 Agent，根据 USE_PINECONE 环境变量选择检索方式。
    返回 (agent, conn) 供外部使用，conn 用于最后关闭。
    """
    # ================= 1. 初始化检索器 =================
    if not USE_PINECONE:
        # 本地 ChromaDB 检索
        CHROMA_PERSIST_DIR = "./chroma_db_rag"
        COLLECTION_NAME = "Eshopping_rag_collection"

        if not os.path.exists(CHROMA_PERSIST_DIR):
            base_dir = os.path.dirname(os.path.dirname(__file__))
            CHROMA_PERSIST_DIR = os.path.join(base_dir, "chroma_db_rag")

        retriever = HybridRetriever(CHROMA_PERSIST_DIR, COLLECTION_NAME, use_rerank=True)

        # 创建工具列表
        tools: List = [
            KnowledgeBaseTool(retriever),
            RecommendTool()
        ]
    else:
        # Pinecone 云端检索
        # 初始化 Pinecone 客户端
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
        if not PINECONE_API_KEY or not PINECONE_ENVIRONMENT:
            raise ValueError("Pinecone 模式下需要设置 PINECONE_API_KEY 和 PINECONE_ENVIRONMENT")

        pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index("eshop-index")   # 索引名称，根据实际情况调整

        # 获取 Hugging Face Token（可选，但推荐）
        HF_API_TOKEN = os.getenv("HF_API_TOKEN")
        MODEL_ID = "BAAI/bge-small-zh-v1.5"   # 与你导出的向量模型一致

        def get_embedding(text: str) -> list:
            """调用 Hugging Face Inference API 生成查询向量"""
            api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{MODEL_ID}"
            headers = {"Authorization": f"Bearer {HF_API_TOKEN}"} if HF_API_TOKEN else {}
            payload = {"inputs": text}
            try:
                response = requests.post(api_url, headers=headers, json=payload, timeout=10)
                if response.status_code == 200:
                    return response.json()
                else:
                    raise Exception(f"HF API 错误: {response.status_code} - {response.text}")
            except Exception as e:
                raise Exception(f"向量生成失败: {e}")

        # 定义 Pinecone 检索工具
        @tool
        def pinecone_search(query: str) -> str:
            """从云端商品知识库中检索相关信息"""
            try:
                # 1. 生成查询向量
                vector = get_embedding(query)
                # 2. 在 Pinecone 中检索
                results = index.query(vector=vector, top_k=3, include_metadata=True)
                if not results['matches']:
                    return "抱歉，没有找到相关的商品信息。"

                # 3. 格式化结果
                formatted = []
                for match in results['matches']:
                    text = match['metadata'].get('text', '')
                    score = match['score']
                    formatted.append(f"[相似度 {score:.2f}] {text[:150]}...")
                return "\n\n".join(formatted)
            except Exception as e:
                return f"检索失败：{str(e)}"

        tools = [pinecone_search, RecommendTool()]

    # ================= 2. 配置 LLM =================
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

    # ================= 3. 系统提示 =================
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

    # ================= 4. 创建 SQLite 持久化检查点 =================
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints.db")
    db_dir = os.path.dirname(db_path)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)

    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    # ================= 5. 创建 Agent =================
    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=system_prompt,
        checkpointer=checkpointer,
    )

    return agent, conn