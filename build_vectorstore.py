"""
构建 FAISS 向量数据库
  1. 读取 data/ 下所有 .md 商品知识库文档
  2. 使用 Ollama nomic-embed-text 进行语义切片和向量化
  3. 存入 FAISS 索引 + 元数据
"""
import os
import json
import time
from typing import List, Dict, Tuple

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import faiss

from config import (
    DATA_DIR,
    VECTORSTORE_DIR,
    FAISS_INDEX_PATH,
    EMBEDDING_DIM,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
# FAISS 临时文件目录（必须纯 ASCII 路径，FAISS C++ 后端不支持中文路径）
FAISS_TMP_DIR = r"F:\tmp"

from embedding_utils import get_embeddings_batch, encode_query


def load_documents(data_dir: str) -> List[Document]:
    """加载 data/ 下所有 .md 文件为 LangChain Document 列表"""
    docs = []
    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith(".md"):
            fpath = os.path.join(data_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
            # 提取商品名作为元数据
            product_name = fname.replace(".md", "").replace("product_", "").replace("_", " ")
            doc = Document(
                page_content=content,
                metadata={"source": fname, "product": product_name.strip()},
            )
            docs.append(doc)
    print(f"[OK] 加载了 {len(docs)} 个商品知识库文档")
    return docs


def semantic_chunk_documents(docs: List[Document]) -> List[Document]:
    """用 RecursiveCharacterTextSplitter 按语义粒度切片"""
    # 使用中文友好的分隔符
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n## ", "\n\n### ", "\n\n", "\n", "。", "，", " ", ""],
        keep_separator=True,
    )
    chunks = splitter.split_documents(docs)
    print(f"[OK] 切片完成，共 {len(chunks)} 个文本块")
    return chunks


def build_faiss_index(
    chunks: List[Document],
    index_path: str,
) -> Tuple[faiss.Index, List[Dict]]:
    """
    使用 Ollama 嵌入模型构建 FAISS 索引并保存
    返回 (faiss_index, metadata_list)
    """
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]

    print(f"[Embedding] 正在通过 Ollama 向量化 {len(texts)} 个文本块...")
    embeddings = get_embeddings_batch(texts, batch_size=4, show_progress=True)

    # 构建 FAISS 索引（内积 = 余弦相似度，因为已归一化）
    index = faiss.IndexFlatIP(EMBEDDING_DIM)  # Inner Product
    index.add(np.array(embeddings, dtype=np.float32))
    print(f"[OK] FAISS 索引构建完成，共 {index.ntotal} 条向量")

    # 确保目录存在
    index_dir = os.path.dirname(index_path)
    os.makedirs(index_dir, exist_ok=True)

    # FAISS C++ 后端对中文路径处理不佳，写入纯英文路径再移动
    os.makedirs(FAISS_TMP_DIR, exist_ok=True)
    tmp_index = os.path.join(FAISS_TMP_DIR, "_faiss_write.index")
    faiss.write_index(index, tmp_index)
    index_file = os.path.join(index_dir, "faiss_index.index")
    import shutil
    shutil.move(tmp_index, index_file)

    # 元数据直接写入（Python 原生支持中文路径）
    meta_file = os.path.join(index_dir, "faiss_index.meta")
    print(f"[OK] 索引已保存: {index_file}")

    # 保存文本元数据
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "texts": texts,
                "metadatas": metadatas,
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[OK] 向量库已保存到 vectorstore/ 目录")
    print(f"  索引: {index_file}\n  元数据: {meta_file}")
    return index, metadatas


def load_faiss_index(index_path: str):
    """加载已有 FAISS 索引 + 元数据"""
    index_dir = os.path.dirname(index_path)
    index_file = os.path.join(index_dir, "faiss_index.index")
    meta_file = os.path.join(index_dir, "faiss_index.meta")

    # FAISS C++ 后端对中文路径支持不佳，先复制到纯英文路径再读取
    import shutil
    os.makedirs(FAISS_TMP_DIR, exist_ok=True)
    tmp_index = os.path.join(FAISS_TMP_DIR, "_faiss_read.index")
    shutil.copy2(index_file, tmp_index)
    index = faiss.read_index(tmp_index)
    try:
        os.remove(tmp_index)
    except OSError:
        pass
    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
    print(f"[OK] 已加载现有向量库: {index.ntotal} 条向量, "
          f"来源 {len(set(m['source'] for m in meta['metadatas']))} 个文档")
    return index, meta


def main():
    print("=" * 60)
    print("   带货直播间商品知识库 - FAISS 向量化构建")
    print(f"   嵌入模型: Ollama (nomic-embed-text, {EMBEDDING_DIM}维)")
    print("=" * 60)

    # 1. 加载文档
    docs = load_documents(DATA_DIR)
    if not docs:
        print("[ERROR] data/ 目录下没有 .md 文件，请先创建知识库文档")
        return

    # 2. 切片
    chunks = semantic_chunk_documents(docs)

    # 3. 向量化 + 构建索引（通过 Ollama 嵌入 API）
    t0 = time.time()
    build_faiss_index(chunks, FAISS_INDEX_PATH)
    print(f"[OK] 向量化耗时: {time.time() - t0:.1f}s")

    # 4. 验证：用示例查询检索
    print("\n" + "=" * 60)
    print("   检索验证")
    print("=" * 60)
    test_queries = [
        "面霜里含有什么修护成分？",
        "敏感肌能不能用这个精华？",
        "精华和面霜叠加会搓泥吗？",
    ]
    index, meta = load_faiss_index(FAISS_INDEX_PATH)
    for q in test_queries:
        q_emb = encode_query(q)
        scores, indices = index.search(q_emb, 3)
        print(f"\n[Query] {q}")
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            text_preview = meta["texts"][idx][:100].replace("\n", " ")
            print(f"  #{i+1} score={score:.4f} | {text_preview}...")


if __name__ == "__main__":
    main()
