#模型使用Cross-Encoder

from sentence_transformers import CrossEncoder

# 1. 加载 Reranker 模型
model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# 2. 用户问题
query = "How can I catch a KeyError in Python?"

# 3. 模拟第一阶段检索到的候选文档
documents = [
    "A KeyError occurs when a dictionary key does not exist.",
    "Python supports object-oriented programming.",
    "Use try-except KeyError to catch missing dictionary keys.",
    "Python dictionaries store key-value pairs."
]

# 4. 构造问题与文档的配对
pairs = [
    (query, document)
    for document in documents
]

# 5. 使用 Cross-Encoder 计算相关性分数
scores = model.predict(pairs)

# 6. 将文档与分数对应起来
results = list(zip(documents, scores))

# 7. 根据分数降序排列
results.sort(
    key=lambda item: float(item[1]),
    reverse=True
)

# 8. 输出最终排名
for rank, (document, score) in enumerate(results, start=1):
    print(f"\n排名：{rank}")
    print(f"相关性得分：{float(score):.4f}")
    print(f"文档：{document}")