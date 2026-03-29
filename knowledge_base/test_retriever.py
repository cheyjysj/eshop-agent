import os
import sys
import time
import json
from typing import List, Tuple, Dict, Any

# 导入检索器
from hybrid_retriever import HybridRetriever

# ==========================================
# 1. 构建测试查询集 (无需标注 ID)
# ==========================================
TEST_QUERIES = [
    {"query": "霍比特人", "category": "精确商品名"},
    {"query": "适合儿童看的字典", "category": "类别 + 功能"},
    {"query": "生物学 细胞控制", "category": "特殊关键词"},
    {"query": "性价比高的童话书", "category": "模糊需求"},
    {"query": "哈利波特全集", "category": "品牌/系列"},
    {"query": "安徒生童话", "category": "品牌/系列"},
    {"query": "古代文明 玛雅", "category": "特定主题"},
    {"query": "折纸教程", "category": "技能/教程"},
    {"query": "细胞生物学教材", "category": "教材/学术"},
    {"query": "幼儿诵读 经典", "category": "特定受众"},
    {"query": "英汉对照 历史", "category": "语言特征"},
    {"query": "套装书 童话", "category": "规格特征"}
]


# ==========================================
# 2. 辅助函数
# ==========================================

def format_result(item: Tuple[str, str, Dict, float], show_score: bool = True) -> str:
    """格式化单条结果用于打印"""
    doc_id, text, meta, score = item
    # 截取文本前 50 字
    short_text = text[:50] + "..." if len(text) > 50 else text
    score_str = f"[{score:.4f}]" if show_score else ""
    return f"   {doc_id} {score_str} | {short_text}"


def calculate_diversity(results: List[Tuple[str, str, Dict, float]]) -> float:
    """
    简单计算结果的多样性 (基于 ID 的唯一性，这里主要是防重)
    在 Top-K 中，如果 ID 都不同，多样性为 1.0
    """
    if not results:
        return 0.0
    unique_ids = set([item[0] for item in results])
    return len(unique_ids) / len(results)


# ==========================================
# 3. 主评估流程
# ==========================================

def run_evaluation():
    print("=" * 80)
    print("开始 RAG 检索效果评估 (无监督模式)")
    print("=" * 80)

    # 配置
    CHROMA_PERSIST_DIR = "./chroma_db_rag"
    COLLECTION_NAME = "Eshopping_rag_collection"

    # 路径修正
    if not os.path.exists(CHROMA_PERSIST_DIR):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(base_dir)
        CHROMA_PERSIST_DIR = os.path.join(project_root, "chroma_db_rag")

    # 初始化
    retriever = HybridRetriever(CHROMA_PERSIST_DIR, COLLECTION_NAME, use_rerank=True)

    # 统计数据
    stats = {
        "Vector": {"latency_sum": 0.0, "count": 0},
        "Hybrid": {"latency_sum": 0.0, "count": 0},
        "Rerank": {"latency_sum": 0.0, "count": 0, "score_sum": 0.0}
    }

    report_lines = []
    report_lines.append("### 检索策略横向对比报告\n")
    report_lines.append("| 查询 (Category) | 策略 A: 纯向量 | 策略 B: 混合 (RRF) | 策略 C: 混合 + Rerank |")
    report_lines.append("| :--- | :--- | :--- | :--- |")

    total_queries = len(TEST_QUERIES)

    for idx, test_case in enumerate(TEST_QUERIES):
        query = test_case['query']
        category = test_case['category']

        print(f"\n[{idx + 1}/{total_queries}] 🔍 查询: '{query}' ({category})")
        print("-" * 60)

        results_store = {}

        # --- 策略 1: 纯向量 ---
        start = time.time()
        res_vec = retriever.vector_search(query, top_k=3)
        lat_vec = time.time() - start
        stats["Vector"]["latency_sum"] += lat_vec
        stats["Vector"]["count"] += 1
        results_store['vec'] = res_vec
        print(f"向量检索耗时: {lat_vec:.3f}s")
        for i, item in enumerate(res_vec):
            print(f"  [Vec-{i + 1}] {format_result(item)}")

        # --- 策略 2: 混合 (无 Rerank) ---
        start = time.time()
        res_hyb = retriever.hybrid_search(query, top_k=3, use_rerank=False)
        lat_hyb = time.time() - start
        stats["Hybrid"]["latency_sum"] += lat_hyb
        stats["Hybrid"]["count"] += 1
        results_store['hyb'] = res_hyb
        print(f"⏱混合检索耗时: {lat_hyb:.3f}s")
        for i, item in enumerate(res_hyb):
            print(f"  [Hyb-{i + 1}] {format_result(item)}")

        # --- 策略 3: 混合 + Rerank ---
        start = time.time()
        res_rerank = retriever.hybrid_search(query, top_k=3, use_rerank=True)
        lat_rerank = time.time() - start
        stats["Rerank"]["latency_sum"] += lat_rerank
        stats["Rerank"]["count"] += 1

        # 计算 Rerank 分数的平均值 (作为置信度指标)
        if res_rerank:
            avg_score = sum([item[3] for item in res_rerank]) / len(res_rerank)
            stats["Rerank"]["score_sum"] += avg_score

        results_store['rnk'] = res_rerank
        print(f"Rerank 总耗时: {lat_rerank:.3f}s (含精排)")
        for i, item in enumerate(res_rerank):
            print(f"  [Rnk-{i + 1}] {format_result(item)}")

        # --- 简单分析 ---
        # 检查 Top-1 是否一致
        top1_vec = res_vec[0][0] if res_vec else None
        top1_hyb = res_hyb[0][0] if res_hyb else None
        top1_rnk = res_rerank[0][0] if res_rerank else None

        change_status = "不变"
        if top1_vec != top1_rnk:
            change_status = "Rerank 改变了 Top-1"
        elif top1_hyb != top1_rnk:
            change_status = "Rerank 改变了 Top-1 (相对于混合)"

        print(f"分析: {change_status}")

        # --- 生成 Markdown 表格行 ---
        # 为了表格不爆炸，只放 Top-1 的 ID 和简短文本
        def get_md_cell(items):
            if not items: return "无结果"
            top = items[0]
            return f"**{top[0]}**: {top[1][:20]}..."

        row = f"| {query}<br>*({category})* | {get_md_cell(res_vec)} | {get_md_cell(res_hyb)} | {get_md_cell(res_rerank)} |"
        report_lines.append(row)

    # ==========================================
    # 4. 汇总统计
    # ==========================================
    print("\n" + "=" * 80)
    print("性能与质量统计")
    print("=" * 80)

    avg_lat_vec = stats["Vector"]["latency_sum"] / stats["Vector"]["count"]
    avg_lat_hyb = stats["Hybrid"]["latency_sum"] / stats["Hybrid"]["count"]
    avg_lat_rnk = stats["Rerank"]["latency_sum"] / stats["Rerank"]["count"]
    avg_conf_rnk = stats["Rerank"]["score_sum"] / stats["Rerank"]["count"]

    print(f"平均响应时间:")
    print(f"  - 纯向量检索 : {avg_lat_vec:.3f} 秒")
    print(f"  - 混合检索   : {avg_lat_hyb:.3f} 秒 ({((avg_lat_hyb - avg_lat_vec) / avg_lat_vec) * 100:.1f}% 开销)")
    print(f"  - 混合+Rerank: {avg_lat_rnk:.3f} 秒 ({((avg_lat_rnk - avg_lat_vec) / avg_lat_vec) * 100:.1f}% 开销)")
    print(f"\nRerank 平均置信度得分: {avg_conf_rnk:.4f} (越接近 1 越相关)")

    # ==========================================
    # 5. 输出 Markdown 报告
    # ==========================================
    print("\n" + "=" * 80)
    print("复制以下内容为 Markdown 报告")
    print("=" * 80)

    md_report = "\n".join(report_lines)
    print(md_report)

    print("\n**性能对比：**")
    print(f"- **纯向量检索**: 平均耗时 `{avg_lat_vec:.3f}s`")
    print(f"- **混合检索**: 平均耗时 `{avg_lat_hyb:.3f}s`")
    print(f"- **混合 + Rerank**: 平均耗时 `{avg_lat_rnk:.3f}s`, 平均置信度 `{avg_conf_rnk:.3f}`")


    # 1. 观察 **Rerank** 列是否比 **向量** 列更贴合查询意图。
    # 2. 特别注意‘细胞控制’、‘儿童字典’等复杂查询，看 Rerank 是否修正了关键词匹配错误。
    # 3. 如果 Rerank 改变了 Top-1，请判断这种改变是‘更准确了’还是‘误杀了’。


if __name__ == "__main__":
    try:
        run_evaluation()
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        import traceback

        traceback.print_exc()