import chromadb

client = chromadb.PersistentClient(path="./chroma_db_rag")
try:
    client.delete_collection("Eshopping_rag_collection")
    print("集合已删除")
except Exception as e:
    print(f"删除失败或集合不存在: {e}")