# 真实检索对照实验 &与embedding对比 RRF混合检索融合

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# 1. 准备知识库
documents = [
    "Python KeyError occurs when a dictionary key is missing.",
    "Use try except to catch a missing dictionary key in Python.",
    "Python lists support indexing and slicing.",
    "Rust ownership helps manage memory safely.",
    "Python KeyError can be handled using try except.",
    "Python dictionaries store key value pairs."
]

# 2. 用户问题
query = "How to catch Python KeyError?"

# 3. 分词：统一转换成小写，并去掉简单标点
def tokenize(text):
    return text.lower().replace(".", "").replace("?", "").split()


# 4. 对知识库进行分词
tokenized_documents = [
    tokenize(doc)
    for doc in documents
]

# 5. 创建 BM25 检索器
bm25 = BM25Okapi(tokenized_documents)

# 6. 对用户问题分词
tokenized_query = tokenize(query)

# 7. 计算所有文档的 BM25 分数
scores = bm25.get_scores(tokenized_query)

# 8. 按照 BM25 分数排序
ranked_indices = sorted(
    range(len(documents)),
    key=lambda i: scores[i],
    reverse=True
)

# 9. 输出检索结果
for rank, i in enumerate(ranked_indices, start=1):
    print(f"\n排名：{rank}")
    print(f"文档 ID：{i}")
    print(f"BM25 得分：{scores[i]:.4f}")
    print(f"内容：{documents[i]}")

# ==========================
# Step 6.2：Embedding 检索
# ==========================

# 1. 加载 Embedding 模型
embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
)

# 2. 将知识库文档转换为向量
document_embeddings = embedding_model.encode(
    documents,
    normalize_embeddings=True
)

# 3. 将用户问题转换为向量
query_embedding = embedding_model.encode(
    query,
    normalize_embeddings=True
)

# 4. 计算余弦相似度
similarities = document_embeddings @ query_embedding

# 5. 按相似度从高到低排序
vector_ranked_indices = similarities.argsort()[::-1]

# 6. 输出结果
print("\n========== Embedding 检索结果 ==========")

for rank, i in enumerate(vector_ranked_indices, start=1):
    print(f"\n排名：{rank}")
    print(f"文档 ID：{i}")
    print(f"相似度：{similarities[i]:.4f}")
    print(f"内容：{documents[i]}")

# ==========================
# Step 6.3：RRF 混合检索
# ==========================

def rrf_fusion(result_lists, k=60):
    scores = {}

    for results in result_lists:
        for rank, doc_id in enumerate(results, start=1):
            doc_id = int(doc_id)

            if doc_id not in scores:
                scores[doc_id] = 0.0

            scores[doc_id] += 1 / (k + rank)

    return sorted(
        scores.items(),
        key=lambda item: (-item[1], item[0])
    )


# BM25 的文档排名
bm25_ids = ranked_indices

# Embedding 的文档排名
vector_ids = vector_ranked_indices

# 执行 RRF 融合
hybrid_results = rrf_fusion([
    bm25_ids,
    vector_ids
])

print("\n========== Hybrid Search（RRF）==========")

for rank, (doc_id, score) in enumerate(hybrid_results, start=1):
    print(f"\n排名：{rank}")
    print(f"文档 ID：{doc_id}")
    print(f"RRF 得分：{score:.6f}")
    print(f"内容：{documents[doc_id]}")

# ==========================
# Step 6.4：检索效果评估
# ==========================

relevant_ids = {1, 4}


def evaluate(ranked_ids, relevant_ids, k):
    top_k = [int(doc_id) for doc_id in ranked_ids[:k]]

    # 前 K 个结果中有多少相关文档
    hits = len(set(top_k) & relevant_ids)

    precision = hits / k
    recall = hits / len(relevant_ids)

    return precision, recall


# 获取三种检索方法的排序结果
bm25_ids = ranked_indices
vector_ids = vector_ranked_indices
hybrid_ids = [doc_id for doc_id, score in hybrid_results]


print("\n========== 检索效果评估 ==========")

for name, ids in [
    ("BM25", bm25_ids),
    ("Embedding", vector_ids),
    ("Hybrid", hybrid_ids)
]:
    precision, recall = evaluate(ids, relevant_ids, k=1)

    print(f"\n{name}")
    print(f"Precision@1: {precision:.2f}")
    print(f"Recall@1: {recall:.2f}")