# streamlit_app.py
import os
import streamlit as st
import requests
import pinecone
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

# ================= 环境变量加载 =================
load_dotenv()

# 1. DeepSeek API
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    st.error("请在环境变量或 Streamlit Secrets 中设置 DEEPSEEK_API_KEY")
    st.stop()

# 2. Pinecone 配置
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
if not PINECONE_API_KEY or not PINECONE_ENVIRONMENT:
    st.error("请在环境变量或 Streamlit Secrets 中设置 PINECONE_API_KEY 和 PINECONE_ENVIRONMENT")
    st.stop()

# 3. Hugging Face API Token（可选，用于提高请求频率）
HF_API_TOKEN = os.getenv("HF_API_TOKEN", None)

# ================= Pinecone 客户端初始化 =================
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "eshop-index"  # 与上传时一致
try:
    index = pc.Index(INDEX_NAME)
except Exception as e:
    st.error(f"无法连接到 Pinecone 索引 '{INDEX_NAME}': {e}")
    st.stop()

# ================= 向量生成函数（使用 Hugging Face Inference API）=================
MODEL_ID = "BAAI/bge-small-zh-v1.5"  # 必须与导出向量时使用的模型一致

def get_embedding(text: str) -> list:
    """调用 Hugging Face Inference API 生成查询向量"""
    api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{MODEL_ID}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"} if HF_API_TOKEN else {}
    payload = {"inputs": text}

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=15)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Hugging Face API 错误: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"向量生成失败: {e}")
        return None

# ================= 知识库检索工具 =================
@tool
def search_products(query: str) -> str:
    """从商品知识库中检索相关商品信息（如评分、描述、类别等）"""
    # 1. 生成查询向量
    vector = get_embedding(query)
    if vector is None:
        return "抱歉，无法处理您的查询，请稍后再试。"

    # 2. 在 Pinecone 中检索
    try:
        results = index.query(
            vector=vector,
            top_k=3,                 # 返回最相关的3个结果
            include_metadata=True
        )
        if not results['matches']:
            return "没有找到相关商品。"

        # 3. 格式化输出
        output = []
        for match in results['matches']:
            metadata = match['metadata']
            text = metadata.get('text', '无描述')
            score = match['score']
            output.append(f"[相关度 {score:.2f}] {text[:150]}...")
        return "\n\n".join(output)
    except Exception as e:
        return f"检索失败: {e}"

# ================= 工具列表（仅使用检索工具）=================
tools = [search_products]

# ================= 配置 LLM =================
model = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
    api_key=DEEPSEEK_API_KEY,
    temperature=0.7,
    max_tokens=1024,
)

# ================= 系统提示 =================
system_prompt = """你是一个智能电商客服助手，你的任务是根据用户的问题，调用 search_products 工具来回答商品相关问题。
如果用户询问商品详情、评分、推荐等，请使用该工具从知识库中检索信息。
如果用户询问订单等非商品问题，请礼貌告知暂无法处理。
回答时请结合检索结果，以友好、专业的语气回复。"""

# ================= 创建 Agent（使用内存检查点）=================
checkpointer = InMemorySaver()
agent = create_react_agent(
    model=model,
    tools=tools,
    prompt=system_prompt,
    checkpointer=checkpointer,
)

# ================= Streamlit UI =================
st.set_page_config(page_title="智能电商客服助手", page_icon="🤖")
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

    # 调用 Agent
    config = {"configurable": {"thread_id": st.session_state.session_id}}
    with st.spinner("思考中..."):
        response = agent.invoke(
            {"messages": [HumanMessage(content=prompt)]},
            config=config
        )
        reply = response["messages"][-1].content

    # 显示助手回复
    st.chat_message("assistant").write(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})