
text = (
    "Python 的列表是一种可变数据结构，可以存储多个元素。"
    "列表支持 append() 方法，可以在末尾添加元素。"
    "insert() 方法可以在指定位置插入元素。"
    "pop() 方法可以删除并返回元素。"
    "与列表不同，元组创建后不能修改。"
    "因此列表适合频繁修改的数据，而元组适合固定的数据。"
)


def split_text(text, chunk_size, overlap):
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")

    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap 必须满足 0 <= overlap < chunk_size")

    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size]
        chunks.append(chunk)

        if start + chunk_size >= len(text):
            break

    return chunks


# 每组参数：(Chunk Size, Overlap)
configs = [
    (30, 0),
    (30, 10),
    (60, 0),
    (60, 10),
]

for chunk_size, overlap in configs:
    chunks = split_text(text, chunk_size, overlap)

    print(f"\nChunk Size={chunk_size}, Overlap={overlap}")
    print(f"分块数量：{len(chunks)}")

    total_chars = sum(len(chunk) for chunk in chunks)
    print(f"存储字符总量：{total_chars}")

    for index, chunk in enumerate(chunks, start=1):
        print(f"Chunk {index}: {chunk}")

