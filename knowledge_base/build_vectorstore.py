import os
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.api.types import EmbeddingFunction, Documents
from tqdm import tqdm
from splitdata import get_processed_documents

# 配置
DATA_PATH = "D:/rag-QA/data/processed/kb_docs/docs.parquet"
CHROMA_PERSIST_DIR = r"D:\rag-QA\chroma_db_rag"
COLLECTION_NAME = "Eshopping_rag_collection"
MODEL_NAME = "BAAI/bge-small-zh-v1.5"
BATCH_SIZE = 1000

print(f"数据路径: {DATA_PATH}")
print(f"向量库存储路径: {CHROMA_PERSIST_DIR}")
print(f"使用模型: {MODEL_NAME}")

# 1. 获取文档
langchain_docs = get_processed_documents(file_path=DATA_PATH, threshold=500)
if not langchain_docs:
    raise ValueError("未获取到任何文档")

all_texts = [doc.page_content for doc in langchain_docs]
all_ids = [str(doc.metadata.get('productId', f"doc_index_{i}")) for i, doc in enumerate(langchain_docs)]
print(f"数据提取完成。文本数: {len(all_texts)}, ID 数: {len(all_ids)}")

# 2. 定义 Embedding 函数
class Text2vec_EmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name=MODEL_NAME):
        print(f"🔄 加载模型：{model_name}")
        self.model = SentenceTransformer(model_name)
        print("✅ 模型加载完成")

    def __call__(self, input: Documents) -> list[list[float]]:
        return self.model.encode(
            input,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=64
        ).tolist()

# 3. 创建持久化客户端（直接写入磁盘）
client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

# 删除旧集合（如果需要）
try:
    client.delete_collection(COLLECTION_NAME)
    print("已删除旧集合")
except:
    pass

embed_fn = Text2vec_EmbeddingFunction()
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embed_fn,
    metadata={"hnsw:space": "cosine"}
)
print(f"✅ 集合 '{COLLECTION_NAME}' 已就绪")

# 4. 分批写入数据（直接持久化）
print(f"\n开始写入 {len(all_texts)} 条文档，批次大小 {BATCH_SIZE}")
total_docs = len(all_texts)
num_batches = (total_docs + BATCH_SIZE - 1) // BATCH_SIZE

for i in tqdm(range(num_batches), desc="写入批次"):
    start_idx = i * BATCH_SIZE
    end_idx = min((i + 1) * BATCH_SIZE, total_docs)
    batch_texts = all_texts[start_idx:end_idx]
    batch_ids = all_ids[start_idx:end_idx]
    batch_metadatas = [{"productId": pid} for pid in batch_ids]

    collection.add(
        documents=batch_texts,
        ids=batch_ids,
        metadatas=batch_metadatas
    )

# 5. 验证
final_count = collection.count()
print(f"\n✅ 完成！向量库中总文档数: {final_count}")
print(f"💾 向量库已保存至: {os.path.abspath(CHROMA_PERSIST_DIR)}")