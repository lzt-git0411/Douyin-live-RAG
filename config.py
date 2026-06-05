"""
全局配置文件
"""
import os

# FAISS + PyTorch OpenMP 冲突修复
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ==================== 路径配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")
EVAL_DIR = os.path.join(BASE_DIR, "eval_results")

# FAISS 索引存储路径
FAISS_INDEX_PATH = os.path.join(VECTORSTORE_DIR, "faiss_index")

# ==================== Embedding 配置 ====================
# 使用 Ollama 中的 nomic-embed-text 嵌入模型（768 维，137M 参数）
# 备选：如需用 BGE 可改为 "BAAI/bge-small-zh-v1.5"（需 HF 网络）
EMBEDDING_MODEL_NAME = "nomic-embed-text"
EMBEDDING_DIM = 768  # nomic-embed-text 的输出维度
EMBEDDING_DEVICE = "ollama"  # 使用 Ollama API 而非本地加载模型

# ==================== 切片配置 ====================
CHUNK_SIZE = 400          # 每个 chunk 最大字符数
CHUNK_OVERLAP = 80        # 相邻 chunk 重叠字符数

# ==================== Ollama 配置 ====================
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:3b"  # 可换成 qwen2.5:7b / llama3.2:1b 等
OLLAMA_TEMPERATURE = 0.1  # RAG 场景用低温减少幻觉
OLLAMA_MAX_TOKENS = 512

# ==================== 检索配置 ====================
TOP_K = 5                 # 检索返回的文档片段数
SIMILARITY_THRESHOLD = 0.3  # 相似度阈值，低于此值的片段不采纳

# ==================== 评测配置 ====================
TEST_QUESTIONS_FILE = os.path.join(BASE_DIR, "data", "test_questions.json")

# ==================== 确保目录存在 ====================
os.makedirs(VECTORSTORE_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)
