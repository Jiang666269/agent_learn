#day 7 Chunk级增量更新#10 + hash + stable ID

# Day 7：Chunk 级增量更新 + RAG

import os
import json
import hashlib
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer


# =========================
# Step 1：基础路径
# =========================

BASE_DIR = Path(__file__).resolve().parent

KNOWLEDGE_DIR = BASE_DIR / "knowledge"
STATE_FILE = BASE_DIR / "file_state.json"
CHROMA_DIR = BASE_DIR / "chroma_db"

# 如果 knowledge 文件夹不存在，就创建它
KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Step 2：DeepSeek API
# =========================

# .env 位于 agent_learn 项目根目录
load_dotenv(BASE_DIR.parent / ".env")

api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise ValueError(
        "没有找到 DEEPSEEK_API_KEY，请检查项目根目录的 .env"
    )

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)


# =========================
# Step 3：Embedding 模型
# =========================

embedding_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


# =========================
# Step 4：初始化 Chroma
# =========================

chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = chroma_client.get_or_create_collection(
    name="day7_knowledge",
    metadata={
        "hnsw:space": "cosine"
    }
)


# =========================
# Step 5：读取文档
# =========================

def load_document(file_path):

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        return file.read()


# =========================
# Step 6：切分 Chunk
# =========================

def split_text(
    text,
    chunk_size=200,
    overlap=50
):

    if chunk_size <= overlap:
        raise ValueError(
            "chunk_size 必须大于 overlap"
        )

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


# =========================
# Step 7：生成 Chunk Hash
# =========================

def get_chunk_hash(chunk):

    return hashlib.sha256(
        chunk.encode("utf-8")
    ).hexdigest()


# =========================
# Step 8：读取 / 保存状态
# =========================

def load_file_state():

    try:

        with open(
            STATE_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

            return data if isinstance(data, dict) else {}

    except FileNotFoundError:

        return {}


def save_file_state(state):

    with open(
        STATE_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            state,
            file,
            ensure_ascii=False,
            indent=4
        )


# =========================
# Step 9：构建 Chunk 映射
# =========================

def build_chunk_map(chunks):

    chunk_map = {}
    hash_counts = {}

    for index, chunk in enumerate(chunks):

        chunk_hash = get_chunk_hash(chunk)

        # 为相同内容的重复 Chunk 添加出现序号
        occurrence = hash_counts.get(chunk_hash, 0)
        hash_counts[chunk_hash] = occurrence + 1

        # 相同文本的第 0、1、2 次出现具有不同 key
        chunk_key = f"{chunk_hash}_{occurrence}"

        chunk_map[chunk_key] = {
            "text": chunk,
            "hash": chunk_hash,
            "index": index
        }

    return chunk_map


# =========================
# Step 10：Chunk 级增量同步
# =========================

def sync_knowledge_base():

    if not KNOWLEDGE_DIR.is_dir():
        raise FileNotFoundError(
            f"知识库目录不存在：{KNOWLEDGE_DIR}"
        )

    txt_files = sorted(
        KNOWLEDGE_DIR.glob("*.txt")
    )

    print("找到 TXT 文件：", len(txt_files))

    file_state = load_file_state()

    # 数据库为空但状态记录有旧 Chunk 时，
    # 说明数据库可能被清空，需要重新建立索引
    if (
        collection.count() == 0
        and any(
            isinstance(state, dict)
            and state.get("chunk_keys")
            for state in file_state.values()
        )
    ):

        print("Chroma 数据库为空，重新建立索引。")
        file_state = {}


    # -------------------------
    # 10.1 检查已经删除的文件
    # -------------------------

    current_files = {
        file_path.name
        for file_path in txt_files
    }

    deleted_files = (
        set(file_state.keys()) - current_files
    )

    for file_name in deleted_files:

        print(file_name, "已经删除，清理旧数据。")

        collection.delete(
            where={
                "source": file_name
            }
        )

        del file_state[file_name]


    # -------------------------
    # 10.2 逐个处理现存文件
    # -------------------------

    for file_path in txt_files:

        file_name = file_path.name
        source_name = file_path.stem

        modified_time = file_path.stat().st_mtime

        previous_state = file_state.get(
            file_name,
            {}
        )

        if not isinstance(previous_state, dict):
            previous_state = {}

        # 检查状态格式是否包含 Chunk key
        has_previous_keys = (
            isinstance(previous_state.get("chunk_keys"), list)
        )

        # 文件未变化，直接跳过
        if (
            has_previous_keys
            and previous_state.get("modified_time") == modified_time
        ):

            print(f"文件未变化，跳过：{file_name}")
            continue

        print(file_name, "发生变化，开始比较 Chunk...")


        # -------------------------
        # 10.3 读取并重新切分文件
        # -------------------------

        document = load_document(file_path)

        chunks = split_text(
            document,
            chunk_size=200,
            overlap=50
        )

        new_chunk_map = build_chunk_map(chunks)


        # -------------------------
        # 10.4 获取新旧 Chunk key
        # -------------------------

        old_keys = set(
            previous_state.get("chunk_keys", [])
            if has_previous_keys else []
        )

        new_keys = set(
            new_chunk_map.keys()
        )


        # 如果是首次建立记录，清理同来源的遗留数据
        if not has_previous_keys:

            collection.delete(
                where={
                    "source": file_name
                }
            )


        # -------------------------
        # 10.5 比较新旧 Chunk
        # -------------------------

        unchanged_keys = old_keys & new_keys
        added_keys = new_keys - old_keys
        deleted_keys = old_keys - new_keys

        print(
            f"{source_name}: "
            f"未变化={len(unchanged_keys)}, "
            f"新增={len(added_keys)}, "
            f"删除={len(deleted_keys)}"
        )


        # -------------------------
        # 10.6 删除消失的 Chunk
        # -------------------------

        deleted_ids = [
            f"{source_name}_{key}"
            for key in deleted_keys
        ]

        if deleted_ids:

            collection.delete(
                ids=deleted_ids
            )


        # -------------------------
        # 10.7 更新未变化 Chunk 的 metadata
        # -------------------------
        # 不重新 Embedding，只更新 Chunk 的位置等信息

        unchanged_list = [
            key
            for key in new_chunk_map
            if key in unchanged_keys
        ]

        unchanged_ids = [
            f"{source_name}_{key}"
            for key in unchanged_list
        ]

        unchanged_metadatas = [
            {
                "source": file_name,
                "chunk_index": new_chunk_map[key]["index"],
                "chunk_hash": new_chunk_map[key]["hash"]
            }
            for key in unchanged_list
        ]

        if unchanged_ids:

            collection.update(
                ids=unchanged_ids,
                metadatas=unchanged_metadatas
            )


        # -------------------------
        # 10.8 只给新增 Chunk 做 Embedding
        # -------------------------

        added_list = [
            key
            for key in new_chunk_map
            if key in added_keys
        ]

        if added_list:

            added_chunks = [
                new_chunk_map[key]["text"]
                for key in added_list
            ]

            added_ids = [
                f"{source_name}_{key}"
                for key in added_list
            ]

            added_metadatas = [
                {
                    "source": file_name,
                    "chunk_index": new_chunk_map[key]["index"],
                    "chunk_hash": new_chunk_map[key]["hash"]
                }
                for key in added_list
            ]

            # 只计算新增 Chunk 的向量
            embeddings = embedding_model.encode(
                added_chunks,
                normalize_embeddings=True
            )

            # 写入新增 Chunk
            collection.upsert(
                ids=added_ids,
                documents=added_chunks,
                embeddings=embeddings.tolist(),
                metadatas=added_metadatas
            )


        # -------------------------
        # 10.9 更新文件状态
        # -------------------------

        file_state[file_name] = {
            "modified_time": modified_time,

            # 保留每个 Chunk 的原始 Hash
            "chunk_hashes": [
                new_chunk_map[key]["hash"]
                for key in new_chunk_map
            ],

            # 保存用于稳定识别 Chunk 的 key
            "chunk_keys": list(new_chunk_map.keys())
        }


    # -------------------------
    # 10.10 保存状态
    # -------------------------

    save_file_state(file_state)

    print("Chunk 级增量同步完成！")


# =========================
# Step 11：向量检索
# =========================

def retrieve(
    query,
    top_k=3,
    max_distance=0.6
):

    total = collection.count()

    if total == 0:
        return []

    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True
    )

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],

        n_results=min(top_k, total),

        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    filtered_results = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):

        if distance <= max_distance:

            filtered_results.append({
                "text": document,
                "source": metadata["source"],
                "chunk_index": metadata["chunk_index"],
                "distance": float(distance)
            })

    return filtered_results


# =========================
# Step 12：RAG 回答
# =========================

def run_rag(user_input):

    results = retrieve(user_input)

    if not results:

        return "知识库中没有找到足够相关的资料。"

    context_parts = []

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
请根据下面提供的资料回答用户的问题。

资料：
{context_text}

用户问题：
{user_input}

要求：
1. 优先依据提供的资料回答。
2. 如果资料不足，请明确说明。
3. 不要编造资料中没有的具体事实。
4. 最后列出参考来源文件。
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

    return (
        response.choices[0].message.content
        or "模型没有返回文本回答。"
    )


# =========================
# Step 13：主程序
# =========================

if __name__ == "__main__":

    # 启动时先同步知识库
    sync_knowledge_base()

    # 进入 RAG 聊天循环
    while True:

        question = input("你：")

        if question.strip().lower() in [
            "退出",
            "结束",
            "exit",
            "quit"
        ]:

            print("AI：再见")
            break

        answer = run_rag(question)

        print("AI：", answer)

