"""
src/rag/retriever.py
====================
RAG 知識庫檢索模組。

【檢索策略】
  1. 向量相似度搜尋（Semantic Search）
     → ChromaDB cosine similarity，返回 top-K 候選段落
  2. 重排序（Reranking）
     → 使用 Cross-Encoder 模型精確計算查詢與候選段落的相關性
     → 可選：BAAI/bge-reranker-base（支援中文，含台語漢字）
  3. 結果過濾
     → 相似度低於門檻（0.3）的結果捨棄，避免注入不相關資訊

【為什麼需要 Reranking？】
  向量搜尋（Bi-encoder）速度快但精度有限，
  Cross-Encoder 在精度上更高（直接計算 query-doc 配對分數），
  兩者串聯即「Search → Rerank」是目前 RAG 最佳實踐。

【查詢擴展（台語特有需求）】
  長者輸入常使用口語台語：
    - 「腹肚痛」 ≠ ChromaDB 裡的「胃部疼痛」
    - 「頭殼暈」 ≠ 「頭暈目眩」
  本模組在查詢前進行「台語口語→書面語」對照擴展，
  提升跨語域一致性（cross-lingual retrieval）。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RetrievedChunk:
    """單一檢索結果。"""
    chunk_id: str
    text: str
    source: str
    similarity_score: float    # 向量相似度（0~1）
    rerank_score: float = 0.0  # Cross-Encoder 重排序分數（有執行才有值）
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def final_score(self) -> float:
        """若有 rerank_score 使用之，否則用 similarity_score。"""
        return self.rerank_score if self.rerank_score > 0 else self.similarity_score


# ── 台語口語→書面對照表（查詢擴展用）────────────────────────────
_TAILO_TO_FORMAL: dict[str, str] = {
    "腹肚痛": "胃部疼痛 腹痛 肚子痛",
    "頭殼暈": "頭暈 眩暈 頭暈目眩",
    "心悶": "胸悶 呼吸不順 胸部不適",
    "喘無過去": "呼吸困難 喘不過氣",
    "腳軟": "腳無力 行走困難 下肢無力",
    "歹眠": "睡眠困難 失眠",
    "食無落": "食慾不振 不想吃",
    "漚的感覺": "噁心感 反胃",
    "血壓懸": "高血壓",
    "血糖懸": "高血糖 糖尿病",
    "藥仔": "藥物 藥品 用藥",
    "病院": "醫院 診所",
    "醫生": "醫師 主治醫師",
}


def expand_query(query: str) -> str:
    """
    台語查詢擴展：將口語台語詞彙替換為書面語對照，
    擴展為多個同義詞加入查詢，提升 recall。

    例：「腹肚痛，真嚴重」→「腹肚痛 胃部疼痛 腹痛 肚子痛，真嚴重」
    """
    expanded = query
    for tailo, formal in _TAILO_TO_FORMAL.items():
        if tailo in query:
            expanded = expanded + " " + formal
    return expanded


class ChromaRetriever:
    """
    ChromaDB 向量檢索器（含 Reranking 選項）。

    使用方式
    --------
    >>> retriever = ChromaRetriever(persist_dir="./data/chroma_db")
    >>> results = await retriever.retrieve("腹肚痛，是愛食啥物藥仔？")
    >>> context = retriever.format_context(results)
    """

    def __init__(
        self,
        persist_dir: str = "./data/chroma_db",
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        collection_name: str = "tomni_care_knowledge",
        top_k: int = 5,                    # 向量搜尋候選數量
        rerank_top_n: int = 3,             # Reranking 後保留的最終數量
        min_similarity: float = 0.3,       # 最低相似度門檻
        use_reranker: bool = True,         # 是否啟用 Cross-Encoder 重排序
        reranker_model: str = "BAAI/bge-reranker-base",
    ):
        self.persist_dir = persist_dir
        self.embedding_model_name = embedding_model
        self.collection_name = collection_name
        self.top_k = top_k
        self.rerank_top_n = rerank_top_n
        self.min_similarity = min_similarity
        self.use_reranker = use_reranker
        self.reranker_model_name = reranker_model

        self._chroma_collection = None
        self._reranker = None

    def _get_collection(self):
        """延遲初始化 ChromaDB collection。"""
        if self._chroma_collection is None:
            try:
                import chromadb
                from chromadb.utils import embedding_functions
            except ImportError:
                raise ImportError("請安裝：pip install chromadb")

            client = chromadb.PersistentClient(path=self.persist_dir)
            ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model_name
            )
            self._chroma_collection = client.get_collection(
                name=self.collection_name,
                embedding_function=ef,
            )
            print(f"[RAG] ChromaDB 連線成功，{self._chroma_collection.count()} 筆資料")
        return self._chroma_collection

    def _get_reranker(self):
        """延遲初始化 Cross-Encoder Reranker。"""
        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder
                print(f"[RAG] 載入 Reranker：{self.reranker_model_name}")
                self._reranker = CrossEncoder(self.reranker_model_name)
                print("[RAG] Reranker 載入完成 ✓")
            except ImportError:
                print("[RAG] ⚠ sentence-transformers 未安裝，跳過 Reranking")
                self.use_reranker = False
        return self._reranker

    def retrieve_sync(
        self,
        query: str,
        use_query_expansion: bool = True,
    ) -> List[RetrievedChunk]:
        """
        同步檢索入口。

        流程：
          1. 查詢擴展（台語口語→書面對照）
          2. ChromaDB 向量搜尋（Bi-encoder）
          3. 相似度過濾
          4. Cross-Encoder Reranking（若啟用）
          5. 返回 top-N 結果
        """
        collection = self._get_collection()

        # ① 查詢擴展
        search_query = expand_query(query) if use_query_expansion else query

        # ② ChromaDB 向量搜尋
        results = collection.query(
            query_texts=[search_query],
            n_results=min(self.top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        if not results["documents"] or not results["documents"][0]:
            return []

        # ③ 轉換格式 + 相似度過濾
        # ChromaDB distance 是「1 - cosine_similarity」，需轉換
        chunks: List[RetrievedChunk] = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            similarity = 1.0 - dist   # cosine distance → similarity
            if similarity < self.min_similarity:
                continue
            chunks.append(
                RetrievedChunk(
                    chunk_id=meta.get("chunk_id", ""),
                    text=doc,
                    source=meta.get("file", "unknown"),
                    similarity_score=similarity,
                    metadata=meta,
                )
            )

        if not chunks:
            return []

        # ④ Cross-Encoder Reranking
        if self.use_reranker and len(chunks) > 1:
            reranker = self._get_reranker()
            if reranker:
                pairs = [(query, c.text) for c in chunks]
                scores = reranker.predict(pairs)
                for chunk, score in zip(chunks, scores):
                    chunk.rerank_score = float(score)
                # 依 rerank_score 降序排列
                chunks.sort(key=lambda c: c.rerank_score, reverse=True)

        # ⑤ 取 top-N
        return chunks[: self.rerank_top_n]

    async def retrieve(
        self,
        query: str,
        use_query_expansion: bool = True,
    ) -> List[RetrievedChunk]:
        """非同步檢索入口（FastAPI / LangGraph 整合用）。"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.retrieve_sync, query, use_query_expansion
        )

    @staticmethod
    def format_context(chunks: List[RetrievedChunk], max_chars: int = 800) -> str:
        """
        將檢索結果格式化為注入 Prompt 的上下文字串。

        格式化考量：
          - 加入來源標注，讓 LLM 知道資訊來源
          - 以 --- 分隔不同段落，降低 LLM 混淆機率
          - 截斷至 max_chars，避免 context window 超限
        """
        if not chunks:
            return ""

        lines = []
        total_chars = 0
        for i, chunk in enumerate(chunks, 1):
            header = f"【來源 {i}：{chunk.source}】"
            entry = f"{header}\n{chunk.text}"
            if total_chars + len(entry) > max_chars:
                break
            lines.append(entry)
            total_chars += len(entry)

        return "\n---\n".join(lines)

    def collection_stats(self) -> dict:
        """回傳 ChromaDB 統計資訊（除錯用）。"""
        collection = self._get_collection()
        return {
            "total_chunks": collection.count(),
            "collection_name": self.collection_name,
            "persist_dir": self.persist_dir,
            "embedding_model": self.embedding_model_name,
            "reranker_enabled": self.use_reranker,
        }
