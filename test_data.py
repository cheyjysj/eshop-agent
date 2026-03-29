# import chromadb
#
# client = chromadb.PersistentClient(path="./chroma_db_rag")
# collection = client.get_collection("Eshopping_rag_collection")
#
# # 查询文档内容中包含 "AKG" 的记录
# results = collection.get(
#     where_document={"$contains": "AKG"},
#     limit=10,
#     include=["documents", "metadatas"]
# )
#
# print("找到的记录：")
# for doc, meta in zip(results['documents'], results['metadatas']):
#     print(f"文档: {doc[:200]}...")
#     print(f"元数据: {meta}\n")
#
# # 或者查看所有文档的前 100 字符（不设条件）
# all_docs = collection.get(limit=10, include=["documents"])
# for doc in all_docs['documents']:
#     print(doc[:100])

import pandas as pd
df = pd.read_parquet("D:/rag-QA/data/processed/kb_docs/docs.parquet")
print("AKG 商品数量:", df[df['doc_text'].str.contains('AKG', na=False)].shape[0])
print("总商品数:", df.shape[0])