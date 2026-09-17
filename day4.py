#day4 RAG、Embedding&向量检索

#Vector DB -> Top-k -> Threshold过滤 -> 得到相关chunk

import os #读取环境变量
import numpy as np #做向量计算,.dot()计算点积，.argsort()返回排序后的下标·············

from dotenv import load_dotenv #加载.env文件
from openai import OpenAI #创捷Deepseek API 客户端
from sentence_transformers import SentenceTransformer #加载Embedding模型

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

#创建Embedding 模型 将文本变成向量
embedding_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

#知识库 Chunk
knowledge_base = [
    "Rust 的所有权系统用于管理内存，每个值都有一个所有者。",
    "当变量离开作用域时，Rust 会自动释放它拥有的数据。",
    "借用允许程序在不转移所有权的情况下访问数据。",
    "Rust 的生命周期用于描述引用保持有效的范围。",
    "Python 的列表是一种可变序列。"
]

# 3. 提前生成所有 Chunk 的向量
chunk_embeddings = embedding_model.encode(
    knowledge_base,
    normalize_embeddings=True #把每个向量归一化，点积直接表示Cosine Similarity
)#chunk_embeddings实际上是很多个向量组成的矩阵

# 4. 检索函数，query：用户问题
def retrieve(query, top_k=3, threshold=0.4):

    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True
    )

    #计算问题和所有chunk的相似度
    similarities = np.dot(
        chunk_embeddings,
        query_embedding
    )

    sorted_indices = np.argsort(
        similarities
    )[::-1]  #按相似度从高到低排列Chunk

    results = []

    for index in sorted_indices[:top_k]:

        score = similarities[index]

        if score >= threshold:
            results.append({
                "text": knowledge_base[index],
                "score": float(score)   # NumPy的数值类型有时是np.float32，转成普通float
            })

    return results

# 5. RAG

def run_rag(user_input):

    results = retrieve(user_input)

    if not results:
        return "知识库中没有找到足够相关的资料。"

    chunks = [
        item["text"]
        for item in results
    ]

    context_text = "\n\n".join(chunks)

    prompt = f"""
请只根据下面提供的资料回答问题。

资料：

{context_text}

用户问题：

{user_input}

如果资料不足以回答，请明确说明资料不足。
"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content

#用户输入
while True:
    question = input("你：")

    if question.strip().lower() in [
        "退出",
        "exit",
        "quit"
    ]:
        break
    answer = run_rag(question)

    print("AI：", answer)