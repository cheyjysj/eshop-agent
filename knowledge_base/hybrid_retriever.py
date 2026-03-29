import os
import sys
import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
import jieba
from typing import List, Tuple, Dict, Any
from sentence_transformers import SentenceTransformer, CrossEncoder
from chromadb.api.types import EmbeddingFunction, Documents

# 确保能导入同级目录的其他模块
sys.path.append(os.path.dirname(__file__))

# ==========================================
# 1. 定义与构建时一致的 Embedding 函数
# ==========================================
MODEL_NAME = "BAAI/bge-small-zh-v1.5"
# Rerank 模型配置
RERANK_MODEL_NAME = "BAAI/bge-reranker-large"

class Text2vec_Large_Chinese_EmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name=MODEL_NAME):
        print(f"正在加载 Embedding 模型：{model_name} ...")
        try:
            self.model = SentenceTransformer(model_name)
            print("Embedding 模型加载完成！")
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise e

    def __call__(self, input: Documents) -> list[list[float]]:
        embeddings = self.model.encode(
            input,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=64
        )
        return embeddings.tolist()


#初始化混合检索器
class HybridRetriever:
    def __init__(self, chroma_persist_dir: str, collection_name: str,
                 batch_size: int = 500, use_rerank: bool = True):
        print(f"正在初始化混合检索器...")
        print(f"- 连接 ChromaDB: {chroma_persist_dir}")
        # 1. 初始化 ChromaDB 和 Embedding
        # 实例化与构建时相同的 Embedding 函数
        embedding_fn = Text2vec_Large_Chinese_EmbeddingFunction(model_name=MODEL_NAME)

        # 1. 连接 ChromaDB 并指定 embedding_function
        self.client = chromadb.PersistentClient(path=chroma_persist_dir)

        # 在 get_collection 时传入 embedding_function
        self.collection = self.client.get_collection(
            name=collection_name,
            embedding_function=embedding_fn
        )

        # 获取总数
        total_count = self.collection.count()
        print(f"   - 向量库中文档总数：{total_count}")

        # 2. 分批加载所有文档到内存
        print(f"   - 正在分批加载所有文档 (每批 {batch_size} 条) 以构建 BM25 索引...")
        self.doc_ids, self.doc_texts, self.doc_metadatas = self._load_all_documents_batched(batch_size)

        if not self.doc_texts:
            raise ValueError("向量库中没有任何文档！请先运行 build_vectorstore.py")

        print(f"   - 成功加载 {len(self.doc_texts)} 篇文档到内存。")

        # 2.2 中文分词并构建 BM25 索引
        print("   - 正在进行中文分词并构建 BM25 模型 (这可能需要 10-30 秒)...")
        self.tokenized_corpus = [list(jieba.cut(text)) for text in self.doc_texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # 2.3 建立 ID 到索引的映射
        self.id_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.doc_ids)}

        # 3. 初始化 Rerank 模型
        self.reranker = None
        if use_rerank:
            print(f"   - 正在加载 Rerank 模型：{RERANK_MODEL_NAME} (首次运行需下载约 1.3GB)...")
            try:
                # max_length 根据模型上下文限制设定，bge-reranker-large 通常为 512
                self.reranker = CrossEncoder(RERANK_MODEL_NAME, max_length=512)
                print("   - Rerank 模型加载完成！")
            except Exception as e:
                print(f"   - Rerank 模型加载失败：{e}")
                print("   - 将禁用 Rerank 功能，请检查网络连接或显存。")
                self.reranker = None

        print("混合检索器初始化完成！")

    def _load_all_documents_batched(self, batch_size: int):
        """分批获取所有文档"""
        total_count = self.collection.count()
        all_ids = []
        all_texts = []
        all_metadatas = []

        # 阶段 1: 获取 ID
        print(" 阶段 1: 读取所有文档 ID...")
        for i in range(0, total_count, batch_size):
            try:
                batch_result = self.collection.get(limit=batch_size, offset=i, include=[])
            except Exception:
                batch_result = self.collection.get(limit=batch_size, offset=i)

            all_ids.extend(batch_result['ids'])
            current = min(i + batch_size, total_count)
            if (i // batch_size) % 5 == 0:
                print(f"         ... 已读取 ID {current}/{total_count}", end='\r')
        print(f"         ... 已读取所有 {len(all_ids)} 个 ID。")

        # 阶段 2: 获取内容
        print("      -> 阶段 2: 读取文档内容和元数据...")
        for i in range(0, len(all_ids), batch_size):
            batch_ids = all_ids[i: i + batch_size]
            batch_result = self.collection.get(
                ids=batch_ids,
                include=["documents", "metadatas"]
            )
            all_texts.extend(batch_result['documents'])
            all_metadatas.extend(batch_result['metadatas'])
            current = min(i + batch_size, len(all_ids))
            print(f"         ... 已加载内容 {current}/{len(all_ids)}", end='\r')
        print()
        return all_ids, all_texts, all_metadatas

    def vector_search(self, query: str, top_k: int = 20) -> List[Tuple[str, str, Dict, float]]:
        """执行向量检索"""
        # 现在 collection 已经绑定了正确的 embedding_function，可以直接传 query_texts
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        ids = results['ids'][0]
        docs = results['documents'][0]
        metas = results['metadatas'][0]
        distances = results['distances'][0]

        scores = [max(0.0, 1.0 - d) for d in distances]
        return list(zip(ids, docs, metas, scores))

    def bm25_search(self, query: str, top_k: int = 20) -> List[Tuple[str, str, Dict, float]]:
        """执行 BM25 关键词检索"""
        tokenized_query = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                doc_id = self.doc_ids[idx]
                doc_text = self.doc_texts[idx]
                meta = self.doc_metadatas[idx]
                score = float(scores[idx])
                results.append((doc_id, doc_text, meta, score))
        return results

    def rrf_fusion(self, vec_results: List, bm25_results: List, k: int = 60, top_k: int = 20) -> List[
        Tuple[str, str, Dict, float]]:
        """RRF 融合"""
        score_dict: Dict[str, Tuple[float, str, Dict]] = {}

        for rank, (doc_id, text, meta, _) in enumerate(vec_results, start=1):
            rrf_score = 1.0 / (k + rank)
            if doc_id in score_dict:
                old_score, _, _ = score_dict[doc_id]
                score_dict[doc_id] = (old_score + rrf_score, text, meta)
            else:
                score_dict[doc_id] = (rrf_score, text, meta)

        for rank, (doc_id, text, meta, _) in enumerate(bm25_results, start=1):
            rrf_score = 1.0 / (k + rank)
            if doc_id in score_dict:
                old_score, _, _ = score_dict[doc_id]
                score_dict[doc_id] = (old_score + rrf_score, text, meta)
            else:
                score_dict[doc_id] = (rrf_score, text, meta)

        sorted_items = sorted(score_dict.items(), key=lambda x: x[1][0], reverse=True)
        final_results = []
        for doc_id, (fused_score, text, meta) in sorted_items[:top_k]:
            final_results.append((doc_id, text, meta, fused_score))
        return final_results

    # ==========================================
    # 核心 Rerank 方法
    # ==========================================
    def _rerank(self, query: str, candidates: List[Tuple[str, str, Dict, float]], top_k: int = 3) -> List[
        Tuple[str, str, Dict, float]]:
        """
        使用 CrossEncoder 对候选文档进行重排序
        :param query: 用户查询
        :param candidates: RRF 融合后的候选列表 [(id, text, meta, score), ...]
        :param top_k: 最终返回的数量
        :return: 重排序后的列表
        """
        if not self.reranker:
            print("Rerank 模型未加载，跳过重排序步骤。")
            return candidates[:top_k]

        if not candidates:
            return []

        print(f"   - 🔍 正在对 {len(candidates)} 个候选文档进行 Rerank 精排...")

        # 1. 构造 (query, document) 对
        # CrossEncoder 需要成对的输入
        pairs = [[query, doc_text] for _, doc_text, _, _ in candidates]

        # 2. 预测分数
        # predict 返回的是一个列表，包含每个 pair 的相关性分数 (logits 或 sigmoid 后的值)
        rerank_scores = self.reranker.predict(pairs)

        # 3. 绑定分数并排序
        # 将原始数据与新分数结合
        ranked_candidates = []
        for i, (doc_id, doc_text, meta, original_score) in enumerate(candidates):
            ranked_candidates.append({
                'id': doc_id,
                'text': doc_text,
                'meta': meta,
                'original_score': original_score,
                'rerank_score': float(rerank_scores[i])
            })

        # 按 rerank_score 降序排序
        ranked_candidates.sort(key=lambda x: x['rerank_score'], reverse=True)

        # 4. 格式化返回结果 (保留 top_k)
        final_results = []
        for item in ranked_candidates[:top_k]:
            # 这里我们返回 (id, text, meta, rerank_score)，也可以把 original_score 放在 meta 里
            final_results.append((
                item['id'],
                item['text'],
                item['meta'],
                item['rerank_score']
            ))

        return final_results

    def hybrid_search(self, query: str, top_k: int = 3,
                      vec_top_k: int = 20, bm25_top_k: int = 20,
                      use_rerank: bool = True) -> List[Tuple[str, str, Dict, float]]:
        """
        混合检索主函数
        :param use_rerank: 是否启用 Rerank 精排
        """
        # 1. 双路召回
        vec_results = self.vector_search(query, top_k=vec_top_k)
        bm25_results = self.bm25_search(query, top_k=bm25_top_k)

        # 2. RRF 融合
        # 注意：如果要用 Rerank，这里融合的候选数量 (rrf_top_k) 应该大于最终需要的 top_k
        # 比如最终要 3 个，这里可以先融合出 15-20 个给 Rerank 挑选
        rrf_top_k = 20 if use_rerank else top_k
        fused_results = self.rrf_fusion(vec_results, bm25_results, k=60, top_k=rrf_top_k)

        # 3. 【新增】Rerank 精排
        if use_rerank and self.reranker:
            final_results = self._rerank(query, fused_results, top_k=top_k)
        else:
            final_results = fused_results[:top_k]

        return final_results


if __name__ == "__main__":
    CHROMA_PERSIST_DIR = "./chroma_db_rag"
    COLLECTION_NAME = "Eshopping_rag_collection"


    if not os.path.exists(CHROMA_PERSIST_DIR):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(base_dir)
        CHROMA_PERSIST_DIR = os.path.join(project_root, "chroma_db_rag")

    try:
        # 初始化时加载 Rerank 模型
        retriever = HybridRetriever(CHROMA_PERSIST_DIR, COLLECTION_NAME, use_rerank=True)

        test_queries = [
            "霍比特人",
            "适合儿童看的字典",
            "生物学 细胞控制",
            "性价比高的童话书"
        ]

        for query in test_queries:
            print("\n" + "=" * 80)
            print(f"🔍 用户查询：{query}")
            print("=" * 80)

            # --- 实验 A: 仅混合检索 (无 Rerank) ---
            print("\n[A] 仅混合检索 (RRF Only):")
            results_no_rerank = retriever.hybrid_search(query, top_k=3, use_rerank=False)
            for i, (doc_id, text, meta, score) in enumerate(results_no_rerank, 1):
                display_text = text[:60] + "..." if len(text) > 60 else text
                print(f"  {i}. [Score: {score:.4f}] ID:{doc_id} | {display_text}")

            # --- 实验 B: 混合检索 + Rerank ---
            print("\n[B] 混合检索 + Rerank (Final):")
            results_with_rerank = retriever.hybrid_search(query, top_k=3, use_rerank=True)
            for i, (doc_id, text, meta, score) in enumerate(results_with_rerank, 1):
                display_text = text[:60] + "..." if len(text) > 60 else text
                # 这里的 score 现在是 Rerank 分数
                print(f"  {i}. [Rerank: {score:.4f}] ID:{doc_id} | {display_text}")

    except Exception as e:
        print(f"测试失败：{e}")
        import traceback

        traceback.print_exc()