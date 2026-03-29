# run_agent.py
import os
import sys
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from agent.agent_core import create_agent_instance

load_dotenv()

# 创建 Agent 实例
agent, conn = create_agent_instance()

# ================= 测试对话 =================
session_id = "test_user_001"
config = {"configurable": {"thread_id": session_id}}

questions = [
    "我想买一本适合儿童看的字典，有什么推荐吗？",
    "订单号 123456 现在怎么样了？",
    "帮我推荐一款降噪耳机"
]

for q in questions:
    print("\n" + "=" * 60)
    print(f"用户: {q}")
    response = agent.invoke(
        {"messages": [HumanMessage(content=q)]},
        config=config
    )
    assistant_reply = response["messages"][-1].content
    print(f"助手: {assistant_reply}")

print("\n对话结束，所有历史已自动保存到 checkpoints.db")
conn.close()