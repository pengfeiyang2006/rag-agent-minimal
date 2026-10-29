"""全局配置：从 .env 读取，支持 DeepSeek / 通义(百炼) 两种 OpenAI 兼容接口。"""
import os
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录（app 的上一级）
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# --- LLM ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek").strip().lower()
API_KEY = os.getenv("API_KEY", "").strip()

# 默认：DeepSeek 官方兼容接口
BASE_URL = os.getenv("BASE_URL", "https://api.deepseek.com/v1")
CHAT_MODEL = os.getenv("CHAT_MODEL", "deepseek-chat")

if LLM_PROVIDER == "qwen":
    # 通义千问(百炼) OpenAI 兼容模式
    BASE_URL = os.getenv("BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen-plus")

# --- RAG ---
DATA_DIR = ROOT / "data"
CORPUS_PATH = os.getenv("CORPUS_PATH", str(DATA_DIR / "sample_doc.md"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "300"))      # 切片字符数
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))  # 切片重叠字符数
TOP_K = int(os.getenv("TOP_K", "3"))                   # 检索返回片段数

# --- Agent ---
MAX_STEPS = int(os.getenv("MAX_STEPS", "5"))           # ReAct 最大步数（防死循环）
MOCK_MODE = os.getenv("MOCK_MODE", "0") == "1"         # 1=无 Key 演示
