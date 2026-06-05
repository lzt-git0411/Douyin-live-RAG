"""
Ollama Embedding 工具模块
通过 Ollama API 调用 nomic-embed-text 进行文本向量化
"""
import time
from typing import List

import numpy as np
import requests

from config import OLLAMA_BASE_URL, EMBEDDING_MODEL_NAME


def get_embedding(text: str) -> np.ndarray:
    """获取单条文本的嵌入向量"""
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": EMBEDDING_MODEL_NAME, "prompt": text},
        timeout=30,
    )
    resp.raise_for_status()
    return np.array(resp.json()["embedding"], dtype=np.float32)


def get_embeddings_batch(texts: List[str], batch_size: int = 8, show_progress: bool = True) -> np.ndarray:
    """批量获取嵌入向量，返回 (n, dim) 的 numpy 数组"""
    all_embeddings = []
    total = len(texts)

    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        batch_embeds = []
        for text in batch:
            emb = get_embedding(text)
            batch_embeds.append(emb)
            if show_progress:
                print(f"\r  Embedding {len(all_embeddings) + len(batch_embeds)}/{total}...", end="", flush=True)

        all_embeddings.extend(batch_embeds)
        if i + batch_size < total:
            time.sleep(0.05)  # 避免压垮 Ollama

    if show_progress:
        print()

    result = np.array(all_embeddings, dtype=np.float32)

    # L2 归一化（使得内积 = 余弦相似度）
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    result = result / norms

    return result


def encode_query(query: str) -> np.ndarray:
    """编码查询文本（单条，已归一化）"""
    emb = get_embedding(query)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb.reshape(1, -1)
