"""快速测试 RAG Pipeline"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from rag_pipeline import RAGPipeline

p = RAGPipeline()

# 测试1: 知识库内问题
test_questions = [
    "敏感肌能用这款面霜吗？",
    "烟酰胺精华白天能用吗？",
    "精华和面霜叠加会搓泥吗？",
    "这款面霜含有什么修护成分？",
    "烟酰胺精华多久能看到效果？",
]

# 测试2: 知识库外问题
out_questions = [
    "今天北京天气怎么样？",
    "这款面霜和赫莲娜黑绷带哪个好？",
]

print("=" * 60)
print("   RAG 问答测试")
print("=" * 60)

for q in test_questions + out_questions:
    result = p.answer(q, mode="rag")
    print(f"\n[Q] {q}")
    print(f"[A] {result['answer'][:300]}")
    print(f"    检索片段数: {len(result['retrieved_chunks'])}")
