"""
RAG 检索 + 生成管道
  1. 加载 FAISS 向量库
  2. 用户问题 -> 检索 Top-K 相关文档片段
  3. 拼接 Prompt 模板 -> Ollama 大模型生成答案
  4. 提供纯模型（无 RAG）对比接口
"""
import json
import time
from typing import List, Dict, Optional

import numpy as np
import requests

from config import (
    FAISS_INDEX_PATH,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TEMPERATURE,
    OLLAMA_MAX_TOKENS,
    TOP_K,
    SIMILARITY_THRESHOLD,
)
from build_vectorstore import load_faiss_index
from embedding_utils import encode_query


# ==================== Prompt 模板 ====================

RAG_SYSTEM_PROMPT = """## 角色
你是一名专业的直播间美妆护肤顾问。你正在协助主播回答观众关于商品的问题。

## 规则（必须严格遵守）
1. **只能依据【参考文档】中的内容回答**，不得使用你预训练时学到的外部知识。
2. 如果【参考文档】中没有提供足够信息回答用户问题，请明确说"根据现有商品资料，暂时无法回答这个问题"。
3. 语言要亲切、专业、简洁，像直播间话术一样自然流畅。
4. 禁止编造成分、功效、价格、用法等任何不在参考文档中的信息。
5. 回答时尽量引用文档中的具体成分名称和数据，让观众感受到专业性。

## 用户问题
{question}

## 参考文档
{context}"""

PURE_MODEL_SYSTEM_PROMPT = """## 角色
你是一名专业的直播间美妆护肤顾问。你正在协助主播回答观众关于商品的问题。

## 规则
1. 请根据你的知识直接回答用户问题。
2. 语言要亲切、专业、简洁，像直播间话术一样自然流畅。
3. 如果你不确定答案，请诚实告知。

## 用户问题
{question}"""


# ==================== Ollama API ====================

def call_ollama(messages: List[Dict], stream: bool = False) -> str:
    """
    调用 Ollama Chat API
    messages 格式: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
    """
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": stream,
        "options": {
            "temperature": OLLAMA_TEMPERATURE,
            "num_predict": OLLAMA_MAX_TOKENS,
        },
    }
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            f"无法连接到 Ollama ({OLLAMA_BASE_URL})。"
            f"请确认 Ollama 已启动且模型 {OLLAMA_MODEL} 已安装。"
        )
    except Exception as e:
        raise RuntimeError(f"Ollama 调用失败: {e}")


# ==================== 检索器 ====================

class RAGPipeline:
    """RAG 完整管道：检索 + 生成"""

    def __init__(self):
        self.index, self.meta = load_faiss_index(FAISS_INDEX_PATH)

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        """
        检索最相关的文档片段（通过 Ollama 嵌入 API）
        返回: [{"text": ..., "score": ..., "source": ..., "product": ...}, ...]
        """
        q_emb = encode_query(query)
        scores, indices = self.index.search(q_emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score < SIMILARITY_THRESHOLD:
                continue  # 过滤低相关性片段
            meta = self.meta["metadatas"][idx]
            results.append({
                "text": self.meta["texts"][idx],
                "score": float(score),
                "source": meta.get("source", "unknown"),
                "product": meta.get("product", "unknown"),
            })
        return results

    def generate_rag(self, question: str) -> Dict:
        """
        RAG 模式：检索 + 生成
        """
        # Step 1: 检索
        retrieved = self.retrieve(question)

        # Step 2: 拼接上下文
        if not retrieved:
            context = "（未检索到相关文档片段）"
            rag_answer = "根据现有商品资料，暂时无法回答这个问题。建议咨询品牌方获取更多信息。"
            return {
                "question": question,
                "mode": "RAG",
                "answer": rag_answer,
                "retrieved_chunks": [],
                "context": context,
                "elapsed": 0,
            }

        context_parts = []
        for i, chunk in enumerate(retrieved):
            context_parts.append(
                f"[来源: {chunk['product']} | 相关度: {chunk['score']:.3f}]\n{chunk['text']}"
            )
        context = "\n\n---\n\n".join(context_parts)

        # Step 3: 拼接 Prompt
        system_prompt = RAG_SYSTEM_PROMPT.format(question=question, context=context)

        # Step 4: 调用 LLM
        t0 = time.time()
        messages = [{"role": "system", "content": system_prompt}]
        answer = call_ollama(messages)
        elapsed = round(time.time() - t0, 2)

        return {
            "question": question,
            "mode": "RAG",
            "answer": answer,
            "retrieved_chunks": [
                {"text": r["text"][:200], "score": r["score"], "source": r["source"]}
                for r in retrieved
            ],
            "context": context,
            "elapsed": elapsed,
        }

    def generate_pure(self, question: str) -> Dict:
        """
        纯模型模式（无 RAG）：直接问 LLM
        """
        system_prompt = PURE_MODEL_SYSTEM_PROMPT.format(question=question)

        t0 = time.time()
        messages = [{"role": "system", "content": system_prompt}]
        answer = call_ollama(messages)
        elapsed = round(time.time() - t0, 2)

        return {
            "question": question,
            "mode": "Pure LLM",
            "answer": answer,
            "retrieved_chunks": [],
            "context": None,
            "elapsed": elapsed,
        }

    def answer(self, question: str, mode: str = "rag") -> Dict:
        """
        统一接口：根据 mode 选择回答方式
        mode: "rag" | "pure" | "both"
        """
        if mode == "rag":
            return self.generate_rag(question)
        elif mode == "pure":
            return self.generate_pure(question)
        elif mode == "both":
            rag_result = self.generate_rag(question)
            pure_result = self.generate_pure(question)
            return {"rag": rag_result, "pure": pure_result}
        else:
            raise ValueError(f"Unknown mode: {mode}")


# ==================== 交互式问答 ====================

def interactive_qa():
    """交互式命令行问答"""
    pipeline = RAGPipeline()

    print("=" * 60)
    print("   带货直播间 RAG 智能问答系统")
    print(f"   模型: {OLLAMA_MODEL} | 知识库: {pipeline.index.ntotal} 条向量")
    print("=" * 60)
    print("输入问题开始 (输入 'quit' 退出, 'toggle' 切换对比模式)\n")

    show_compare = False
    while True:
        try:
            q = input("\n观众> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not q:
            continue
        if q.lower() == "quit":
            print("再见！")
            break
        if q.lower() == "toggle":
            show_compare = not show_compare
            print(f"[切换] 对比模式: {'ON' if show_compare else 'OFF'}")
            continue

        if show_compare:
            result = pipeline.answer(q, mode="both")
            rag = result["rag"]
            pure = result["pure"]
            print(f"\n{'='*40}")
            print(f"[RAG 模式] ({rag['elapsed']}s)")
            print(f"{'='*40}")
            print(rag["answer"])
            print(f"\n检索片段数: {len(rag['retrieved_chunks'])}")
            print(f"\n{'='*40}")
            print(f"[纯模型模式] ({pure['elapsed']}s)")
            print(f"{'='*40}")
            print(pure["answer"])
        else:
            result = pipeline.answer(q, mode="rag")
            print(f"\n主播AI ({result['elapsed']}s):")
            print(result["answer"])
            if result["retrieved_chunks"]:
                print(f"\n(参考 {len(result['retrieved_chunks'])} 条知识库片段)")


if __name__ == "__main__":
    interactive_qa()
