# agent/tools.py
import sys
import os
from typing import Type, Optional

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

# 确保能导入 knowledge_base 模块
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from knowledge_base.hybrid_retriever import HybridRetriever


# ================= 输入模型定义 =================
class KnowledgeBaseToolInput(BaseModel):
    """知识库检索工具输入参数"""
    query: str = Field(description="用户想要了解的商品问题，例如：'推荐一款降噪耳机' 或 '这款商品的好评率是多少'")

class RecommendToolInput(BaseModel):
    """推荐工具输入参数"""
    user_preference: str = Field(description="用户偏好描述，例如：'运动耳机' 或 '儿童绘本'")


# ================= 知识库检索工具 =================
class KnowledgeBaseTool(BaseTool):
    name: str = "knowledge_base"  #  添加类型注解
    description: str = "从商品知识库中检索相关信息，包括商品描述、评分、评论摘要等。当用户询问商品特性、推荐商品、查询商品口碑时使用。"  # ✅ 添加类型注解
    args_schema: Type[BaseModel] = KnowledgeBaseToolInput

    def __init__(self, retriever: HybridRetriever):
        super().__init__()
        self._retriever = retriever

    def _run(self, query: str) -> str:
        """同步执行检索并返回格式化的文本"""
        try:
            # 使用混合检索 + Rerank，返回 Top-3
            results = self._retriever.hybrid_search(query, top_k=3, use_rerank=True)
            if not results:
                return "抱歉，没有找到相关的商品信息。"

            formatted = []
            for i, (doc_id, text, meta, score) in enumerate(results, 1):
                # text 格式示例：商品：XXX，类别：图书，平均评分：4.5，评论数：120，好评率：0.83
                formatted.append(f"{i}. {text}")
            return "\n\n".join(formatted)
        except Exception as e:
            return f"检索失败：{str(e)}"

    async def _arun(self, query: str) -> str:
        """异步实现（可选）"""
        return self._run(query)


# ================= 推荐工具（简单规则） =================
class RecommendTool(BaseTool):
    name: str = "recommend"  # 添加类型注解
    description: str = "基于用户偏好或热门商品推荐商品。"  # 添加类型注解
    args_schema: Type[BaseModel] = RecommendToolInput

    def _run(self, user_preference: str) -> str:
        # 简单规则：根据关键词匹配预设商品
        # 实际应用中可结合用户画像、协同过滤等
        if "耳机" in user_preference:
            return "根据您的偏好，推荐：\n1. 索尼 WH-1000XM5 降噪耳机\n2. 苹果 AirPods Pro 2\n3. 小米 Buds 4 Pro"
        elif "儿童" in user_preference or "绘本" in user_preference:
            return "根据您的偏好，推荐：\n1. 猜猜我有多爱你 绘本\n2. 不一样的卡梅拉 系列\n3. 神奇校车 科普绘本"
        else:
            return "根据当前热门商品，推荐：\n1. 华为 MateBook X Pro\n2. 戴森吸尘器 V15\n3. 乐高 创意百变系列"

    async def _arun(self, user_preference: str) -> str:
        return self._run(user_preference)