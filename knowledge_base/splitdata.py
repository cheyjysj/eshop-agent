import pandas as pd
from langchain_core.documents import Document # 导入 Document 类
from langchain_text_splitters import RecursiveCharacterTextSplitter as RCTS

def get_processed_documents(file_path="D:/rag-QA/data/processed/kb_docs/docs.parquet", threshold=500):

    docs_df = pd.read_parquet(file_path)

    # 用 .head() 查看前几行，看看 doc_text 的格式（例如是纯文本还是包含统计信息）。
    print(docs_df.head())
    print(docs_df.shape)

    # 用 len(df) 获取文档总数。
    print(f"文档总数为：{len(docs_df)}")


    # 2. df['doc_text'].str.len().mean() 计算平均长度。计算平均长度
    # 确保列存在且为字符串类型，防止报错
    if 'doc_text' in docs_df.columns:
        avg_length = docs_df['doc_text'].astype(str).str.len().mean()
        print(f"文档平均长度为：{avg_length:.2f} 字符")
    else:
        raise ValueError("DataFrame 中未找到 'doc_text' 列")

    # 3.将df对象转换为documents对象

    # 主键列名为 'productId'，如果不存在请改为其他列名或去掉
    id_col = 'productId' if 'productId' in docs_df.columns else 'id'

    documents = [
        Document(
            page_content=str(row['doc_text']),
            metadata={id_col: row[id_col]} if id_col in docs_df.columns else {}
        )
        for _, row in docs_df.iterrows()
    ]
    print(f"   已转换 {len(documents)} 个原始文档对象。")

    # 4. 根据阈值决定是否需要分块
    THRESHOLD = 500
    final_documents = []  # 初始化变量，防止作用域问题
    # 4.1 需要分块
    print("\n--- 分块决策 ---")
    if avg_length > THRESHOLD:
        print(f"⚠️平均长度 ({avg_length:.2f}) 超过阈值 ({THRESHOLD})，需要执行分块操作。")
        print("下面进行分块...")

        # 初始化切分器
        splitter = RCTS(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )

        # 执行切分动作
        split_docs = splitter.split_documents(documents)

        # 计算统计数据
        print(f"   分块完成！原始文档数: {len(documents)} -> 分块后片段数: {len(split_docs)}")

        num_chunks = len(split_docs)

        if num_chunks > 0:
            # 计算所有分块的总字符数
            total_chars = sum(len(doc.page_content) for doc in split_docs)
            avg_chunk_len = total_chars / num_chunks

            print(f"   - 分块后平均长度: {avg_chunk_len:.2f} 字符")

            # 检查是否有句子被切断（简单检查：看最后一块是否以标点结尾）
            last_chunk = split_docs[-1].page_content.strip()

            # 定义常见的结束标点
            end_punctuations = (".", "!", "?", "。", "！", "？", "\n")
            is_cut_off = not last_chunk.endswith(end_punctuations)

            # 预览最后一个片段的前 100 字
            preview = last_chunk[:50].replace("\n", " ")
            print(f'   - 末尾片段预览: "{preview}..."')

            if is_cut_off:
                print(f"   ⚠️ 注意：末尾片段似乎未完整结束（可能被切断）")
            else:
                print(f"   ✅ 语义完整性较好")

            final_documents = split_docs

        else:
            print("   ❌ 分块结果为空，请检查数据或分割参数。")
            final_documents = []

    # 4.2 不需要分块
    else:
        print(f"平均长度 ({avg_length:.2f}) 小于阈值 ({THRESHOLD})。")
        print("   -> 策略：不进行切分，直接将每行作为一个完整的 Document 存入向量库。")
        print(f"   已生成 {len(documents)} 个文档对象。")
        final_documents = documents


    return final_documents

# 允许直接运行此脚本进行测试
if __name__ == "__main__":
    docs = get_processed_documents()
    if docs:
        print(f"\n预览第一条: {docs[0].page_content[:50]}...")