"""
带货直播间商品知识库 RAG 系统
================================
主入口：统一命令行接口

用法:
  python main.py build              # 构建 FAISS 向量库
  python main.py chat               # 交互式 RAG 问答
  python main.py eval               # A/B 对比评测
  python main.py finetune           # 精调理解说明
  python main.py demo               # 运行完整 Demo 流程
"""
import sys


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1].lower()

    if cmd == "build":
        print("[步骤 1/3] 构建 FAISS 向量知识库...\n")
        from build_vectorstore import main as build_main
        build_main()

    elif cmd == "chat":
        print("[步骤 2/3] 启动 RAG 交互式问答...\n")
        from rag_pipeline import interactive_qa
        interactive_qa()

    elif cmd == "eval":
        print("[步骤 3/3] 运行 A/B 对比评测...\n")
        from eval import main as eval_main
        eval_main()

    elif cmd == "finetune":
        print("[精调] PE 局限诊断 + LoRA 精调方案...\n")
        from fine_tuning import main as ft_main
        ft_main()

    elif cmd == "demo":
        print("=" * 60)
        print("   带货直播间 RAG Demo - 完整流程")
        print("=" * 60)
        print("\n[1/3] 构建向量库...")
        from build_vectorstore import main as build_main
        build_main()

        print("\n\n[2/3] 运行 A/B 评测...")
        from eval import main as eval_main
        eval_main()

        print("\n\n[3/3] 精调诊断 + 方案...")
        from fine_tuning import main as ft_main
        ft_main()

        print("\n\n✅ Demo 完整流程执行完毕！")
        print("  - 向量库构建完成 → vectorstore/faiss_index.*")
        print("  - A/B 评测完成 → eval_results/eval_report.json")
        print("  - 精调说明已输出 ↑")
        print("\n💡 尝试交互式问答: python main.py chat")

    else:
        print(f"未知命令: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()
