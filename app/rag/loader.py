"""文档加载与切片。

切片策略：先按空行分段（保留语义完整），超长段落再按字符切，段间保留重叠。
chunk_size / chunk_overlap 可在 .env 调整，是 RAG 调优的第一组变量。
"""
from pathlib import Path

from .. import config


def read_doc(path=None):
    """读取语料文本。支持 .md / .txt，要求 UTF-8。"""
    path = Path(path or config.CORPUS_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"语料不存在: {path}。可在 .env 的 CORPUS_PATH 指定你自己的文档路径。"
        )
    return path.read_text(encoding="utf-8")


def split_chunks(text, chunk_size=None, overlap=None):
    """把整篇文本切成片段列表，返回非空 chunk。"""
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    buf = ""
    for p in paragraphs:
        # 超长段落内部再按字符切（带重叠）
        while len(p) > chunk_size:
            chunks.append(p[:chunk_size])
            p = p[chunk_size - overlap:]
        if buf and len(buf) + len(p) + 1 > chunk_size:
            chunks.append(buf)
            buf = p
        else:
            buf = (buf + "\n" + p).strip()
    if buf:
        chunks.append(buf)
    return [c for c in chunks if c.strip()]
