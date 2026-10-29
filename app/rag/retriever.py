"""BM25 检索器（jieba 中文分词）。

为什么用 BM25 而不是向量库：
- 小语料 + 中文场景下，BM25+jieba 命中率已够用
- 零成本、零外部依赖、可解释（词频/逆文档频率），面试能讲清原理
- 语料变大或需要语义匹配时再升级向量检索（通义 text-embedding 等），接口已按 search() 隔离
"""
import json
import math
import re
from collections import Counter

import jieba

from .. import config
from .loader import read_doc, split_chunks


class BM25Index:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_tokens = [self._tokenize(c) for c in chunks]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avgdl = sum(self.doc_len) / max(len(self.doc_len), 1)
        # 逆文档频率
        df = {}
        for tokens in self.doc_tokens:
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1
        self.N = len(self.doc_tokens)
        # Lucene 变体：+1 避免负数 idf
        self.idf = {term: math.log((self.N - n + 0.5) / (n + 0.5) + 1) for term, n in df.items()}

    @staticmethod
    def _tokenize(text):
        text = re.sub(r"[^\w\u4e00-\u9fff]+", " ", str(text).lower())
        return [w for w in jieba.cut(text) if w.strip()]

    def search(self, query, top_k=None):
        """返回按分数降序的命中片段（分数>0 才返回）。"""
        top_k = top_k or config.TOP_K
        q_count = Counter(self._tokenize(query))
        scores = []
        for i, tokens in enumerate(self.doc_tokens):
            tf_counter = Counter(tokens)
            score = 0.0
            for term, qf in q_count.items():
                tf = tf_counter.get(term, 0)
                if tf == 0:
                    continue
                idf = self.idf.get(term, 0)
                denom = tf + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                score += idf * qf * (tf * (self.k1 + 1)) / denom
            scores.append(score)
        ranked = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)
        return [
            {"index": i, "score": round(scores[i], 4), "text": self.chunks[i]}
            for i in ranked
            if scores[i] > 0
        ][:top_k]


_index = None  # 进程内单例：索引只构建一次


def get_index():
    global _index
    if _index is None:
        text = read_doc()
        chunks = split_chunks(text)
        _index = BM25Index(chunks)
    return _index


def search_chunks(query, top_k=None):
    """对外接口：Agent 的 search_docs 工具调用它。"""
    results = get_index().search(query, top_k)
    return {"hits": len(results), "results": results}
