
# 检索结果：按向量相似度从高到低排列
retrieved_ids = ["A", "B", "C", "D", "E", "F", "G"]

# 人工标注的全部相关文本块
relevant_ids = {"A", "B", "E", "G"}


def evaluate_retrieval(retrieved_ids, relevant_ids, k):
    # 只取排名前 K 的结果
    top_k_ids = retrieved_ids[:k]

    # 统计前 K 个结果中真正相关的数量
    relevant_count = sum(
        1 for doc_id in top_k_ids
        if doc_id in relevant_ids
    )

    # Precision@K：相关结果数 / 实际返回结果数
    precision = (
        relevant_count / len(top_k_ids)
        if top_k_ids else 0
    )

    # Recall@K：相关结果数 / 全部相关结果数
    recall = (
        relevant_count / len(relevant_ids)
        if relevant_ids else 0
    )

    return precision, recall


for k in [1, 3, 5, 7]:
    precision, recall = evaluate_retrieval(
        retrieved_ids,
        relevant_ids,
        k
    )

    print(
        f"K={k} | "
        f"Precision={precision:.2f} | "
        f"Recall={recall:.2f}"
    )
