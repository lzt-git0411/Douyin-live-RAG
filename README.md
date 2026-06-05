# 🛍️ Douyin-live-RAG

> 带货直播间商品知识库 RAG 系统 — 基于 FAISS + Ollama 的本地 RAG 问答引擎，专为直播带货场景打造。

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FAISS](https://img.shields.io/badge/FAISS-CPU-green.svg)](https://github.com/facebookresearch/faiss)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-orange.svg)](https://ollama.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📖 项目简介

直播带货中，主播需要快速、准确地回答观众关于商品成分、功效、适用肤质、搭配禁忌等问题。仅靠大模型幻觉太多、仅靠文档太慢——**RAG（检索增强生成）是当前最务实的落地方案**。

这个项目搭建了一套完整的本地 RAG 系统：

- 📚 **知识库**：Markdown 格式的商品文档（面霜/精华/防晒）
- 🔍 **检索**：Ollama `nomic-embed-text` 嵌入 + FAISS 向量索引
- 🧠 **生成**：Ollama `qwen2.5:3b` 本地大模型，严格限定只基于检索内容回答
- 📊 **评测**：A/B 对比（RAG vs 纯模型），多维度量化评测
- 🎯 **精调**：PE 局限诊断 + LoRA 精调方案，解决"说教味重""不懂直播梗""安全边界不稳"三大顽疾

全程**本地运行**，无需 API Key，数据不出机器。

---

## 🏗️ 系统架构

```
用户提问 → Embedding 编码 → FAISS 检索 Top-K
                                ↓
                          [商品知识库文档]
                          面霜.md | 精华.md | 防晒.md
                                ↓
                      拼接 Prompt 模板
                                ↓
                      Ollama 大模型生成回答
                                ↓
                           返回给前端
```

```
项目结构：
├── main.py                  # 主入口：统一命令行接口
├── config.py                # 全局配置（路径/模型/切片/检索/评测）
├── build_vectorstore.py     # 构建 FAISS 向量知识库
├── embedding_utils.py       # Ollama Embedding 封装（nomic-embed-text）
├── rag_pipeline.py          # RAG 检索 + 生成管道（含纯模型对比）
├── eval.py                  # A/B 对比评测（RAG vs 纯模型）
├── fine_tuning.py           # PE 局限诊断 + LoRA 精调方案
├── test_rag_quick.py        # RAG 快速功能测试
├── test_eval_quick.py       # 评测快速功能测试
├── requirements.txt         # 依赖清单
├── data/                    # 商品知识库文档 & 测试问题
│   ├── product_01_面霜.md
│   ├── product_02_精华.md
│   ├── product_03_防晒.md
│   └── test_questions.json
├── vectorstore/             # FAISS 索引 & 元数据（自动生成）
└── eval_results/            # 评测报告（自动生成）
```

---

## 🚀 快速开始

### 环境要求

- Python 3.10+
- [Ollama](https://ollama.com/) 已安装并启动
- 已拉取所需模型：`nomic-embed-text`（嵌入）、`qwen2.5:3b`（生成）

```bash
# 安装 Ollama 模型
ollama pull nomic-embed-text
ollama pull qwen2.5:3b

# 验证
ollama list
```

### 安装

```bash
# 克隆仓库
git clone https://github.com/lzt-git0411/Douyin-live-RAG.git
cd Douyin-live-RAG

# 安装依赖
pip install -r requirements.txt
```

### 使用

```bash
# 完整 Demo 流程（构建 → 评测 → 精调诊断）
python main.py demo

# 或分步执行：

# 1. 构建 FAISS 向量库
python main.py build

# 2. 交互式 RAG 问答
python main.py chat

# 3. A/B 对比评测
python main.py eval

# 4. 精调诊断 & 方案
python main.py finetune
```

### 交互式问答演示

```
============================================================
   带货直播间 RAG 智能问答系统
   模型: qwen2.5:3b | 知识库: 156 条向量
============================================================
输入问题开始 (输入 'quit' 退出, 'toggle' 切换对比模式)

观众> 面霜敏感肌能用吗？

主播AI (1.2s):
敏敏肌的宝子们放心冲！这款水润修护面霜就是为你们研发的，
三重神经酰胺直接把屏障修到位，含马齿苋提取物和青刺果油，
临床测试可降低皮肤敏感性。无香精无酒精无色素的配方，
上脸不会刺痛泛红～

(参考 4 条知识库片段)
```

---

## 📊 评测体系

### A/B 对比：RAG vs 纯模型

| 指标 | RAG 模式 | 纯模型 | 说明 |
|------|---------|--------|------|
| **准确率** | 语义相似度 | 语义相似度 | 与参考答案比较 |
| **幻觉率** | 是否捏造信息 | 是否捏造信息 | 简单 NLP 检测 |
| **完整性** | 答案覆盖度 | 答案覆盖度 | 与参考答案比较 |
| **响应时间** | 检索+生成耗时 | 纯生成耗时 | 秒级 |

### 测试问题覆盖

- **知识库内**：成分查询、安全性、适用肤质、产品搭配、使用场景、功效周期
- **知识库外**：故意问不存在的商品，评测"诚实拒绝"能力

### 运行评测

```bash
python main.py eval
# 报告输出到 eval_results/eval_report.json
```

---

## 🎯 PE + RAG 为什么不够？精调方案

项目核心洞察：**RAG 解决"知识准确"，PE 提供"表层指令"，但以下三类问题无法通过 RAG + PE 覆盖：**

| 问题 | 表现 | 根因 |
|------|------|------|
| **说教味太重** | 用户要"能用不？"，回答长篇成分解析 | 预训练语料以百科/教程为主，深层语言习惯改不了 |
| **不懂直播梗** | "绝绝子""蹲"等直播黑话无法理解 | PE 可模仿句式，但无法真正理解社交含义 |
| **安全边界不稳** | 该拒不拒、该放不放 | RLHF 对齐与直播场景存在结构性矛盾 |

→ **解决方案：LoRA 精调**（详见 `fine_tuning.py`）

```bash
python main.py finetune
# 输出完整诊断 + LoRA 训练方案 + 面试话术
```

**LoRA 优势**：每个主播风格仅 20MB，可插拔切换（李佳琦式激情 / 董宇辉式娓娓道来 / 专业成分党 / 母婴安全版），无需重载 14GB 模型。

---

## ⚙️ 配置说明

核心配置在 `config.py` 中：

```python
# 嵌入模型
EMBEDDING_MODEL_NAME = "nomic-embed-text"  # 768 维
EMBEDDING_DEVICE = "ollama"

# 生成模型
OLLAMA_MODEL = "qwen2.5:3b"     # 可换 qwen2.5:7b / llama3.2 等
OLLAMA_TEMPERATURE = 0.1         # RAG 用低温减少幻觉

# 切片策略
CHUNK_SIZE = 400                 # 每块最大字符数
CHUNK_OVERLAP = 80               # 块间重叠字符数

# 检索参数
TOP_K = 5                        # 返回 Top-K 片段
SIMILARITY_THRESHOLD = 0.3       # 相似度阈值，低于此值不采纳
```

---

## 🔧 技术栈

| 组件 | 技术选型 | 说明 |
|------|----------|------|
| 向量数据库 | FAISS (CPU) | Meta 开源，轻量高效 |
| 嵌入模型 | nomic-embed-text (Ollama) | 768 维，137M 参数，本地免费 |
| 大语言模型 | Qwen2.5:3B (Ollama) | 中文友好，消费级硬件可跑 |
| 文本切片 | LangChain RecursiveCharacterTextSplitter | 中文分隔符优化 |
| 评测指标 | scikit-learn cosine_similarity | 语义相似度对比 |

---

## 🗺️ 路线图

- [x] FAISS 向量知识库构建
- [x] RAG 检索 + 生成管道
- [x] A/B 对比评测体系
- [x] PE 局限诊断 + LoRA 精调方案
- [ ] 多商品知识库管理（动态增删改）
- [ ] Web UI / 直播间实时接入
- [ ] LoRA 实际训练 + 导出到 Ollama
- [ ] 多主播人设切换（运行时动态加载 LoRA）
- [ ] 用户反馈闭环（点赞/踩 → DPO 迭代优化）
- [ ] 流式输出（与直播弹幕实时交互）

---

## 📄 License

MIT License

---

## 🤝 贡献

欢迎 Issue / PR。如果你有直播带货场景的需求，或者对 RAG 技术有想法，欢迎交流！

---

*Built with ❤️ for the livestream commerce era.*
