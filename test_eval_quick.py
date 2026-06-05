"""Mini RAG 评测测试 - 仅测 2 题验证流程"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
from eval import Evaluator

e = Evaluator()

# 快速测试 2 个知识库内问题 + 1 个知识库外问题
test_qs = [
    {"id": "Q1", "question": "面霜里含有哪些核心修护成分？", "expected_answer": "含角鲨烷、三种神经酰胺（NP/AP/EOP）、马齿苋提取物、青刺果油、透明质酸钠、泛醇、β-葡聚糖等修护成分", "category": "成分查询"},
    {"id": "Q4", "question": "烟酰胺精华和面霜叠加会不会搓泥？", "expected_answer": "大概率不会，面霜不含大分子增稠剂和高浓度硅弹体，与多数水基精华兼容良好", "category": "产品搭配"},
    {"id": "H1", "question": "今天北京天气怎么样？", "expected_answer": "知识库中没有相关信息", "category": "知识库外"},
]

results = []
for q in test_qs:
    cat = q.pop("category", "知识库内")
    r = e.evaluate_single(q, cat)
    results.append(r)
    print(f"\n[{r.question_id}] {r.question}")
    print(f"  RAG: {r.rag_answer[:200]}")
    print(f"  Pure: {r.pure_answer[:200]}")
    print(f"  RAG准确率={r.rag_accuracy:.3f}, 纯模型准确率={r.pure_accuracy:.3f}")
    print(f"  RAG幻觉={r.rag_hallucination}, 纯模型幻觉={r.pure_hallucination}")

# 汇总
internal = [r for r in results if r.category == "知识库内"]
rag_acc = sum(r.rag_accuracy for r in internal) / len(internal)
pure_acc = sum(r.pure_accuracy for r in internal) / len(internal)
rag_hall = sum(1 for r in internal if r.rag_hallucination) / len(internal)
pure_hall = sum(1 for r in internal if r.pure_hallucination) / len(internal)

print(f"\n{'='*50}")
print(f"RAG  准确率: {rag_acc:.1%} | 幻觉率: {rag_hall:.0%}")
print(f"纯模型 准确率: {pure_acc:.1%} | 幻觉率: {pure_hall:.0%}")
