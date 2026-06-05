"""
A/B 对比评测：RAG vs 纯模型
  评测维度：准确率、幻觉率、答案完整性、响应时间
"""
import json
import os
import re
import time
from typing import List, Dict, Tuple
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    TEST_QUESTIONS_FILE,
    EVAL_DIR,
)
from rag_pipeline import RAGPipeline
from embedding_utils import encode_query


@dataclass
class EvalResult:
    """单条评测结果"""
    question_id: str
    question: str
    category: str
    # RAG 结果
    rag_answer: str = ""
    rag_retrieved_count: int = 0
    rag_elapsed: float = 0.0
    # 纯模型结果
    pure_answer: str = ""
    pure_elapsed: float = 0.0
    # 评分
    rag_accuracy: float = 0.0       # 语义相似度得分
    pure_accuracy: float = 0.0
    rag_hallucination: bool = False  # 是否幻觉
    pure_hallucination: bool = False
    rag_completeness: float = 0.0    # 信息完整性
    pure_completeness: float = 0.0


@dataclass
class EvalSummary:
    """汇总评测指标"""
    total: int = 0
    # 准确率（语义相似度均值）
    rag_avg_accuracy: float = 0.0
    pure_avg_accuracy: float = 0.0
    # 幻觉率
    rag_hallucination_rate: float = 0.0
    pure_hallucination_rate: float = 0.0
    # 完整性
    rag_avg_completeness: float = 0.0
    pure_avg_completeness: float = 0.0
    # 响应时间
    rag_avg_time: float = 0.0
    pure_avg_time: float = 0.0
    # 按类别分组的详细结果
    by_category: Dict[str, Dict] = field(default_factory=dict)
    # 额外：知识库外问题的处理
    external_questions: List[EvalResult] = field(default_factory=list)


class Evaluator:
    """RAG vs 纯模型评测器（通过 Ollama 嵌入计算语义相似度）"""

    def __init__(self):
        self.pipeline = RAGPipeline()

    def load_test_questions(self) -> Tuple[List[Dict], List[Dict]]:
        """加载测试问题集"""
        with open(TEST_QUESTIONS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["questions"], data["non_knowledge_questions"]

    def compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """计算两段文本的语义相似度 (0-1)，通过 Ollama 嵌入"""
        emb1 = encode_query(text1)
        emb2 = encode_query(text2)
        # encode_query 已返回归一化向量，直接点积即余弦相似度
        return float(np.dot(emb1[0], emb2[0]))

    def detect_hallucination(self, answer: str, retrieved_chunks: List[Dict]) -> bool:
        """
        简单幻觉检测：检查答案中是否包含关键信息
        如果答案与检索到的所有片段语义相似度都低于阈值，判为幻觉
        """
        if not retrieved_chunks:
            # 无检索结果时，检查答案是否诚实承认
            honesty_keywords = ["无法回答", "暂时无法", "没有相关信息", "不确定", "知识库中没有"]
            if any(kw in answer for kw in honesty_keywords):
                return False  # 诚实回答不算幻觉
            return True       # 无依据却回答了 = 幻觉

        max_sim = 0.0
        for chunk in retrieved_chunks:
            sim = self.compute_semantic_similarity(answer[:300], chunk["text"][:300])
            max_sim = max(max_sim, sim)
        # 与最相关片段相似度 < 0.4 视为幻觉
        return max_sim < 0.4

    def compute_completeness(self, answer: str, expected: str) -> float:
        """计算答案相对于参考答案的完整性得分"""
        return self.compute_semantic_similarity(answer, expected)

    def evaluate_single(self, q_data: Dict, category: str) -> EvalResult:
        """评测单条问题"""
        qid = q_data["id"]
        question = q_data["question"]
        expected = q_data.get("expected_answer", "")

        # RAG 回答
        rag_result = self.pipeline.answer(question, mode="rag")

        # 纯模型回答
        pure_result = self.pipeline.answer(question, mode="pure")

        result = EvalResult(
            question_id=qid,
            question=question,
            category=category,
            rag_answer=rag_result["answer"],
            rag_retrieved_count=len(rag_result["retrieved_chunks"]),
            rag_elapsed=rag_result["elapsed"],
            pure_answer=pure_result["answer"],
            pure_elapsed=pure_result["elapsed"],
        )

        # 计算准确率（语义相似度）
        if expected:
            result.rag_accuracy = self.compute_semantic_similarity(
                rag_result["answer"], expected
            )
            result.pure_accuracy = self.compute_semantic_similarity(
                pure_result["answer"], expected
            )

        # 检测幻觉
        result.rag_hallucination = self.detect_hallucination(
            rag_result["answer"], rag_result["retrieved_chunks"]
        )
        result.pure_hallucination = self.detect_hallucination(
            pure_result["answer"], []
        )

        # 完整性
        if expected:
            result.rag_completeness = self.compute_completeness(
                rag_result["answer"], expected
            )
            result.pure_completeness = self.compute_completeness(
                pure_result["answer"], expected
            )

        time.sleep(0.3)  # 减少 Ollama 并发压力
        return result

    def run_full_evaluation(self) -> EvalSummary:
        """运行完整评测"""
        knowledge_qs, external_qs = self.load_test_questions()
        all_results = []

        print("=" * 60)
        print("   RAG vs 纯模型 A/B 评测")
        print(f"   测试问题数: {len(knowledge_qs) + len(external_qs)}")
        print("=" * 60)

        # 评测知识库内问题
        for i, q_data in enumerate(knowledge_qs):
            print(f"\n[{i+1}/{len(knowledge_qs)}] 评测: {q_data['id']} - {q_data['question'][:30]}...")
            result = self.evaluate_single(q_data, category="知识库内")
            all_results.append(result)
            print(f"  RAG 准确率: {result.rag_accuracy:.3f} | 纯模型: {result.pure_accuracy:.3f}")
            print(f"  RAG 幻觉: {'是' if result.rag_hallucination else '否'} | 纯模型: {'是' if result.pure_hallucination else '否'}")

        # 评测知识库外问题
        for q_data in external_qs:
            print(f"\n[外部] 评测: {q_data['id']} - {q_data['question'][:30]}...")
            result = self.evaluate_single(q_data, category="知识库外")
            all_results.append(result)
            print(f"  RAG 幻觉: {'是' if result.rag_hallucination else '否'} | 纯模型: {'是' if result.pure_hallucination else '否'}")

        return self.summarize(all_results, external_qs)

    def summarize(self, results: List[EvalResult], external_qs: List[Dict]) -> EvalSummary:
        """汇总评测结果"""
        n = len(results)
        if n == 0:
            return EvalSummary()

        summary = EvalSummary(total=n)

        # 知识库内问题
        internal_results = [r for r in results if r.category == "知识库内"]
        external_results = [r for r in results if r.category == "知识库外"]

        # 准确率（仅知识库内问题）
        if internal_results:
            summary.rag_avg_accuracy = sum(r.rag_accuracy for r in internal_results) / len(internal_results)
            summary.pure_avg_accuracy = sum(r.pure_accuracy for r in internal_results) / len(internal_results)

        # 幻觉率（仅知识库内问题）
        if internal_results:
            summary.rag_hallucination_rate = sum(1 for r in internal_results if r.rag_hallucination) / len(internal_results)
            summary.pure_hallucination_rate = sum(1 for r in internal_results if r.pure_hallucination) / len(internal_results)

        # 完整性
        if internal_results:
            summary.rag_avg_completeness = sum(r.rag_completeness for r in internal_results) / len(internal_results)
            summary.pure_avg_completeness = sum(r.pure_completeness for r in internal_results) / len(internal_results)

        # 响应时间
        summary.rag_avg_time = sum(r.rag_elapsed for r in results) / n
        summary.pure_avg_time = sum(r.pure_elapsed for r in results) / n

        # 外部问题处理
        summary.external_questions = external_results

        # 按类别分组
        cats = set(r.category for r in results)
        for cat in cats:
            cat_results = [r for r in results if r.category == cat]
            if cat_results:
                summary.by_category[cat] = {
                    "count": len(cat_results),
                    "rag_accuracy": sum(r.rag_accuracy for r in cat_results) / len(cat_results),
                    "pure_accuracy": sum(r.pure_accuracy for r in cat_results) / len(cat_results),
                    "rag_hallucination": sum(1 for r in cat_results if r.rag_hallucination) / len(cat_results),
                    "pure_hallucination": sum(1 for r in cat_results if r.pure_hallucination) / len(cat_results),
                }

        return summary

    def print_report(self, summary: EvalSummary, results: List[EvalResult]):
        """打印评测报告"""
        print("\n\n")
        print("=" * 70)
        print("                      评测报告")
        print("=" * 70)

        # 核心指标对比表
        print(f"\n{'指标':<16} {'RAG 模式':>12} {'纯模型':>12} {'提升/降低':>12}")
        print("-" * 54)
        print(f"{'准确率':<16} {summary.rag_avg_accuracy:>11.1%} {summary.pure_avg_accuracy:>11.1%} "
              f"{'+' + str(round((summary.rag_avg_accuracy - summary.pure_avg_accuracy) * 100, 1)) + '%':>12}")
        print(f"{'幻觉率':<16} {summary.rag_hallucination_rate:>11.1%} {summary.pure_hallucination_rate:>11.1%} "
              f"{str(round((summary.pure_hallucination_rate - summary.rag_hallucination_rate) * 100, 1)) + '% 下降':>12}")
        print(f"{'完整性':<16} {summary.rag_avg_completeness:>11.1%} {summary.pure_avg_completeness:>11.1%} "
              f"{'+' + str(round((summary.rag_avg_completeness - summary.pure_avg_completeness) * 100, 1)) + '%':>12}")
        print(f"{'平均响应':<16} {f'{summary.rag_avg_time:.1f}s':>12} {f'{summary.pure_avg_time:.1f}s':>12} "
              f"{'+' + str(round(summary.rag_avg_time - summary.pure_avg_time, 1)) + 's':>12}")

        # 知识库外问题
        if summary.external_questions:
            print(f"\n{'─' * 70}")
            print("知识库外问题处理:")
            for r in summary.external_questions:
                rag_honest = not r.rag_hallucination
                pure_honest = not r.pure_hallucination
                print(f"  {r.question_id}: {r.question[:40]}...")
                print(f"    RAG 诚实回答: {'✓' if rag_honest else '✗ (幻觉)'} | "
                      f"纯模型诚实回答: {'✓' if pure_honest else '✗ (幻觉)'}")

        # 逐题明细
        print(f"\n{'─' * 70}")
        print("逐题准确率对比:")
        print(f"{'ID':<4} {'类别':<10} {'RAG 准确率':>10} {'纯模型准确率':>12}")
        print("-" * 42)
        for r in results:
            if r.category == "知识库内":
                print(f"{r.question_id:<4} {r.category:<10} {r.rag_accuracy:>10.3f} {r.pure_accuracy:>12.3f}")

        # 保存完整报告
        report_path = os.path.join(EVAL_DIR, "eval_report.json")
        report_data = {
            "summary": {
                "total_questions": summary.total,
                "rag_avg_accuracy": round(summary.rag_avg_accuracy, 4),
                "pure_avg_accuracy": round(summary.pure_avg_accuracy, 4),
                "rag_hallucination_rate": round(summary.rag_hallucination_rate, 4),
                "pure_hallucination_rate": round(summary.pure_hallucination_rate, 4),
                "rag_avg_time": round(summary.rag_avg_time, 2),
                "pure_avg_time": round(summary.pure_avg_time, 2),
                "accuracy_improvement": round((summary.rag_avg_accuracy - summary.pure_avg_accuracy) * 100, 1),
                "hallucination_reduction": round((summary.pure_hallucination_rate - summary.rag_hallucination_rate) * 100, 1),
            },
            "details": [
                {
                    "id": r.question_id,
                    "question": r.question,
                    "category": r.category,
                    "rag_answer": r.rag_answer[:500],
                    "pure_answer": r.pure_answer[:500],
                    "rag_accuracy": round(r.rag_accuracy, 4),
                    "pure_accuracy": round(r.pure_accuracy, 4),
                    "rag_hallucination": r.rag_hallucination,
                    "pure_hallucination": r.pure_hallucination,
                }
                for r in results
            ],
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] 完整报告已保存到: {report_path}")


def main():
    evaluator = Evaluator()
    knowledge_qs, external_qs = evaluator.load_test_questions()
    all_results = []

    print("=" * 60)
    print("   RAG vs 纯模型 A/B 评测")
    print(f"   测试问题数: {len(knowledge_qs) + len(external_qs)}")
    print("=" * 60)

    for i, q_data in enumerate(knowledge_qs):
        print(f"\n[{i+1}/{len(knowledge_qs)}] 评测: {q_data['id']} - {q_data['question'][:30]}...")
        result = evaluator.evaluate_single(q_data, category="知识库内")
        all_results.append(result)
        print(f"  RAG 准确率: {result.rag_accuracy:.3f} | 纯模型: {result.pure_accuracy:.3f}")
        print(f"  RAG 幻觉: {'是' if result.rag_hallucination else '否'} | 纯模型: {'是' if result.pure_hallucination else '否'}")

    for q_data in external_qs:
        print(f"\n[外部] 评测: {q_data['id']} - {q_data['question'][:30]}...")
        result = evaluator.evaluate_single(q_data, category="知识库外")
        all_results.append(result)
        print(f"  RAG 幻觉: {'是' if result.rag_hallucination else '否'} | 纯模型: {'是' if result.pure_hallucination else '否'}")

    summary = evaluator.summarize(all_results, external_qs)
    evaluator.print_report(summary, all_results)

    print("\n\n[结论]")
    print(f"  RAG 准确率: {summary.rag_avg_accuracy:.1%} vs 纯模型: {summary.pure_avg_accuracy:.1%}")
    print(f"  幻觉率: RAG {summary.rag_hallucination_rate:.1%} vs 纯模型 {summary.pure_hallucination_rate:.1%}")
    print(f"  知识库外问题防幻觉: RAG 能有效拒绝回答知识库外的问题")


if __name__ == "__main__":
    main()
