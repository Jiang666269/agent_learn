#day5  Chroma/Collection/Chroma Embedding/Chroma RAG

import os

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# 1. 加载环境变量
load_dotenv()

# 2. DeepSeek 客户端
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# 3. Embedding 模型
embedding_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# 4. 创建持久化 Chroma
chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

# 5. 创建 / 获取 Collection
collection = chroma_client.get_or_create_collection(
    name="rust_notes",
    metadata={"hnsw:space": "cosine"}
)

def load_document(file_path):
    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:
        return file.read()

def split_text(
    text,
    chunk_size=200,
    overlap=50
):
    paragraphs = text.split("\n\n")
    chunks = []

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        if len(paragraph) <= chunk_size:
            chunks.append(paragraph)

        else:
            start = 0
            while start < len(paragraph):

                end = start + chunk_size
                chunk = paragraph[start:end]
                chunks.append(chunk)
                start += chunk_size - overlap
    return chunks

document = load_document(
    "rust_notes.txt"
)

knowledge_base = split_text(
    document,
    chunk_size=200,
    overlap=50
)

print("Chunk 数量：",len(knowledge_base))

# Chunk -> Embedding

chunk_embeddings = embedding_model.encode(
    knowledge_base,
    normalize_embeddings=True
)

#给每个chunk创建唯一ID

source_name = "rust_notes"

ids = [
    f"{source_name}_chunk_{i}"
    for i in range(len(knowledge_base))
]

#给Chunk创建metadata

metadatas = []

for i in range(len(knowledge_base)):
    metadatas.append({
        "source": "rust_notes.txt",
        "chunk_index": i
    })

#写入Chroma
collection.upsert(
    ids=ids,
    documents=knowledge_base,
    embeddings=chunk_embeddings.tolist(),
    metadatas=metadatas
)

print("知识库已成功写入Chroma！")

#查询，distance越小越相似，而similarity越大越相似

def retrieve(
    query,
    top_k=3,
    max_distance=0.6
):
    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True
    )
    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=top_k
    )

    filtered_documents = []

    documents = results["documents"][0]
    distances = results["distances"][0]

    #zip会把两个列表一一配对
    for document, distance in zip(
        documents,
        distances
    ):
        if distance <= max_distance:
            filtered_documents.append(document)

    return filtered_documents  #Chroma支持一次查询多个Query，所以返回结构是二维的 ex. rusults["documents"][0]

#写进context

def run_rag(user_input):

    documents = retrieve(user_input)
    if not documents:
        return "知识库中没有找到足够相关的资料。"
    context_text = "\n\n".join(documents)
    prompt = f"""
请只根据下面提供的资料回答问题。
资料：

{context_text}

用户问题：

{user_input}

如果资料不足，请明确说明资料不足。
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

question = input("你：")

answer = run_rag(question)

print("AI:", answer)
