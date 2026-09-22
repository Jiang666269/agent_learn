# Day 6：Multi-file RAG

import os
import json
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# 加载环境变量
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

# 4. Chroma 持久化客户端

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

# 5. 创建 / 获取 Collection

collection = chroma_client.get_or_create_collection(
    name="multi_notes",
    metadata={
        "hnsw:space": "cosine"
    }
)

# 6. 读取单个文本文件

def load_document(file_path):

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        return file.read()

# 7. Chunk

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

STATE_FILE = "file_state.json"

   #读取文件状态
def load_file_state():
    try:
        with open(
            "file_state.json",
            "r",
            encoding="utf-8"
        ) as file:
            return json.load(file)
    except FileNotFoundError:
        return {}
   #保存文件状态

def save_file_state(state):
    with open(
        "file_state.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            state,
            file,
            ensure_ascii=False,
            indent=4
        )

#step 10 把“知识库同步”和“RAG对话”拆成函数

def sync_knowledge_base():

    # 8. 找到 knowledge 文件夹中的txt
    knowledge_dir = Path("./knowledge")  #path（）代表当前目录下的（）文件


    # 9. 找到所有 txt 文件
    txt_files = list(
        knowledge_dir.glob("*.txt")  #glob(...) 找到目录下所有以.txt结尾的文件
    )


    print("找到的文件：", len(txt_files))


    # -------------
    # step 9 处理“文件被删除”的情况
    # txt文件被删除，而Chunk还在，检索时可能搜到已经不存在的资料
    # 保证knowledge文件夹和Chroma数据库保持同步


    file_state = load_file_state()  #读取以前的文件状态


    # 找当前存在的文件
    current_files = {
        file_path.name
        for file_path in txt_files
    }


    # 找以前有现在无的文件
    deleted_files = (
        set(file_state.keys())
        - current_files
    )


    # 从Chroma和状态文件里一起删除
    for file_name in deleted_files:

        print(
            file_name,
            "已从knowledge中删除，清理数据库中..."
        )

        collection.delete(
            where={
                "source": file_name
            }
        )

        del file_state[file_name]  # 将state从file_state删除

        print(
            file_name,
            "清理完成"
        )


    #***避免每次启动重复Embedding和upsert，增量更新
    #第一次看到文件->入库，文件没变->跳过，文件发生修改
    #->删除旧Chunk，重新Chunk，重新Embedding，写入新Chunk


    #增量处理每个文件
    #扫描
    for file_path in txt_files:

        file_name = file_path.name

        modified_time = file_path.stat().st_mtime  #查看文件最后修改时间


        # 判断
        if (
            file_name in file_state
            and file_state[file_name] == modified_time
        ):

            print(
                file_name,
                "没有变化，跳过"
            )

            continue


        print(
            file_name,
            "发生变化，更新..."
        )


        #删除这个文件以前存进Chroma的所有Chunk
        collection.delete(
            where={
                "source": file_name
            }
        )


        #重新读取文件
        document = load_document(
            file_path
        )


        #重新Chunk
        chunks = split_text(
            document,
            chunk_size=200,
            overlap=50
        )


        source_name = file_path.stem


        #创建 ID
        ids = [
            f"{source_name}_chunk_{i}"
            for i in range(len(chunks))
        ]


        #创建 metadata
        metadatas = []

        for i in range(len(chunks)):

            metadatas.append({
                "source": file_name,
                "chunk_index": i
            })


        #重新 Embedding
        if chunks:

            embeddings = embedding_model.encode(
                chunks,
                normalize_embeddings=True
            )


            #写入 Chroma
            collection.upsert(
                ids=ids,
                documents=chunks,
                embeddings=embeddings.tolist(),
                metadatas=metadatas
            )


        #更新该文件的状态
        file_state[file_name] = modified_time


        print(
            file_name,
            "更新完成，共",
            len(chunks),
            "个 Chunk。"
        )


    #保存所有文件的新状态
    save_file_state(
        file_state
    )


    print("知识库检查完成！")

'''由于后续新增了增量更新，以下代码为step3,4的初始化
# 10. 用于汇总所有文件的数据

all_chunks = []
all_ids = []
all_metadatas = []

# 11. 逐个处理文件

for file_path in txt_files:

    print("正在处理：", file_path.name)

    # 读取文件
    document = load_document(
        file_path
    )
    # 切 Chunk
    chunks = split_text(
        document,
        chunk_size=200,
        overlap=50
    )
    # 去掉文件后缀
    # rust_notes.txt -> rust_notes
    source_name = file_path.stem
    # 12. 处理当前文件中的每一个 Chunk,创建ID和metadata

    for i, chunk in enumerate(chunks):
        #enumerate(chunks)得到i->Chunk下标，chunk->Chunk内容
        # 唯一 ID
        chunk_id = f"{source_name}_chunk_{i}"

        # metadata
        metadata = {
            "source": file_path.name,
            "chunk_index": i
        }

        # 汇总
        all_chunks.append(chunk)
        all_ids.append(chunk_id)
        all_metadatas.append(metadata)

# 13. 查看汇总结果

print("总 Chunk 数量：", len(all_chunks))
print("总 ID 数量：", len(all_ids))
print("总 metadata 数量：", len(all_metadatas))

#14.将所有Chunk一次性Embedding

all_embeddings = embedding_model.encode(
    all_chunks,
    normalize_embeddings=True #归一化，让向量长度变为1
)

#15.四个列表必须严格一一对应
#检查数量是否一致
print("Chunk 数量：", len(all_chunks))
print("ID 数量：", len(all_ids))
print("Embedding 数量：", len(all_embeddings))
print("Metadata 数量：", len(all_metadatas))

#16.统一写入Chroma
#向当前collection写入数据
collection.upsert(
    ids=all_ids,
    documents=all_chunks,
    embeddings=all_embeddings.tolist(),#Numpy数组转普通python列表
    metadatas=all_metadatas
)
print("多文件知识库已成功写入Chroma")
'''

def retrieve(
    query,
    top_k=3,
    max_distance=0.6
):
    # 1. 用户问题 → Query Embedding
    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True
    )
    # 2. 在 Chroma 中检索
    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=top_k,
        #明确要求返回
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )
    # 3. 取出当前 Query 的结果
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    # 4. 保存过滤后的结果
    filtered_results = []
    # 5. 把文本、metadata、distance 一一对应
    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):
        # 6. Distance Threshold
        if distance <= max_distance:

            filtered_results.append({
                "text": document,
                "source": metadata["source"],
                "chunk_index": metadata["chunk_index"],
                "distance": distance
            })

    return filtered_results

#把多文件检索结果拼成Context，并让LLM回答时标注来源

def run_rag(user_input):

    results = retrieve(user_input)
    if not results:
        return "知识库中没有找到足够相关的资料。"

    context_parts = [] #暂时保存每一条检索结果整理后的文本

    for item in results:
        context_parts.append(
            f"""
来源：{item["source"]}
Chunk：{item["chunk_index"]}
内容：
{item["text"]}
"""
        )
    context_text = "\n\n".join(context_parts)
    prompt = f"""
请只根据下面提供的资料回答问题。
资料：
{context_text}
用户问题：
{user_input}
要求：
1. 如果资料足够，请根据资料回答。
2. 如果资料不足，请明确说明资料不足。
3. 回答最后请列出本次参考的来源文件。
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

#主程序
#先同步知识库
sync_knowledge_base()

#进入RAG聊天
while True:

    question = input("你:")

    if question.strip().lower() in[
        "退出",
        "结束",
        "exit",
        "quit"
    ]:
        print("AI:再见")
        break

    answer = run_rag(
        question
    )

    print(
        "AI:",
        answer
    )