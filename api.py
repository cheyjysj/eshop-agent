# api.py
import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

# 导入你的 agent 核心模块
sys.path.append(os.path.dirname(__file__))
from agent.agent_core import create_agent_instance

# 创建全局 Agent 实例（应用启动时调用一次）
agent, conn = create_agent_instance()

app = FastAPI(title="电商客服助手 API")

# 允许跨域（方便前端 Streamlit 或其他客户端调用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # 生产环境可改为特定域名
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    reply: str

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        config = {"configurable": {"thread_id": req.session_id}}
        response = agent.invoke(
            {"messages": [HumanMessage(content=req.message)]},
            config=config
        )
        reply = response["messages"][-1].content
        return ChatResponse(reply=reply)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.on_event("shutdown")
def shutdown():
    conn.close()