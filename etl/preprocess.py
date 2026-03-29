import os
import pandas as pd
import numpy as np
from pathlib import Path

# 项目路径
BASE_DIR = Path(__file__).parent.parent
RAW_PATH = BASE_DIR / "data/raw"
PROCESSED_PATH = BASE_DIR / "data/processed"

# 1. 读取产品表
print("读取 products.csv...")
products = pd.read_csv(RAW_PATH / "products.csv")
print(f"产品数: {len(products)}")

# 2. 读取类别表
print("读取 categories.csv...")
categories = pd.read_csv(RAW_PATH / "categories.csv")
print(f"类别数: {len(categories)}")

# 3. 读取评论表（分块读取，避免内存不足）
print("读取 ratings.csv（分块处理）...")
chunk_size = 500000  # 每批50万行
ratings_chunks = []
total_rows = 0

# 只保留需要的列，并过滤短评论
for chunk in pd.read_csv(RAW_PATH / "ratings.csv", chunksize=chunk_size):
    # 保留有用字段，过滤空或过短的评论
    chunk_clean = chunk[['userId', 'productId', 'rating', 'comment']].copy()
    chunk_clean = chunk_clean.dropna(subset=['comment'])
    chunk_clean = chunk_clean[chunk_clean['comment'].str.len() >= 10]
    ratings_chunks.append(chunk_clean)
    total_rows += len(chunk_clean)
    print(f"已处理 {total_rows} 条有效评论")

ratings = pd.concat(ratings_chunks, ignore_index=True)
print(f"有效评论总数: {len(ratings)}")

# 4. 计算商品统计指标
print("计算商品统计指标...")
product_stats = ratings.groupby('productId').agg(
    review_count=('rating', 'count'),
    avg_rating=('rating', 'mean'),
    good_count=('rating', lambda x: (x >= 4).sum())
).reset_index()
product_stats['good_rate'] = product_stats['good_count'] / product_stats['review_count']

# 5. 处理商品类别（取一级类目）
print("处理商品类别...")
# 拆分 catIds，取第一个类目 ID
products['first_cat_id'] = products['catIds'].str.split(',').str[0].astype(int, errors='ignore')
# 关联类别名称
products_with_cat = products.merge(categories, left_on='first_cat_id', right_on='catId', how='left')
products_with_cat = products_with_cat[['productId', 'name', 'category']].rename(columns={'category': 'main_category'})

# 6. 合并统计信息与商品信息
print("合并数据...")
final_df = product_stats.merge(products_with_cat, on='productId', how='inner')
print(f"最终商品数（有评论且有类别）: {len(final_df)}")

# 7. 构建知识库文档文本
print("构建知识库文档...")
final_df['doc_text'] = (
    "商品：" + final_df['name'] + "，" +
    "类别：" + final_df['main_category'].fillna('未知') + "，" +
    "平均评分：" + final_df['avg_rating'].round(2).astype(str) + "，" +
    "评论数：" + final_df['review_count'].astype(str) + "，" +
    "好评率：" + final_df['good_rate'].round(4).astype(str)
)

kb_docs = final_df[['productId', 'doc_text']]

# 8. 保存为 Parquet（或 CSV）
print("保存结果...")
os.makedirs(PROCESSED_PATH / "kb_docs", exist_ok=True)
kb_docs.to_parquet(PROCESSED_PATH / "kb_docs" / "docs.parquet", index=False)
# 如果想保存为 CSV，用下面这行（但 Parquet 更省空间）
# kb_docs.to_csv(PROCESSED_PATH / "kb_docs.csv", index=False, encoding='utf-8')

print("ETL 完成！知识库文档保存至 data/processed/kb_docs/docs.parquet")