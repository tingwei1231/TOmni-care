"""
src/rag/ingestion.py
====================
知識庫文件攝取（Ingestion）：Markdown / PDF 切塊 + Embedding。

【整體流程】
  文件（Markdown / PDF）
    → DocumentLoader（讀取、清理）
    → TextSplitter（依台語特性切塊）
    → EmbeddingModel（文字向量化）
    → ChromaDB（持久化儲存）

【切塊策略（台語文件特殊考量）】
  標準 RecursiveCharacterTextSplitter 對台語文件效果差，原因：
    - 台語漢字無空格分詞
    - 段落標題常為「第X節」而非英文標點
  本模組改用：
    - chunk_size=300 字元（約 150 個台語字，對應一段說明）
    - 以「\n\n」→「\n」→「。」→「！」→「？」層級切分
    - overlap=50 字元，保留跨段落的語意連貫性

【Embedding 模型】
  預設：paraphrase-multilingual-MiniLM-L12-v2
    - 支援繁體中文 + 台語漢字
    - 384 維向量，速度快
    - 本機 CPU 推論即可

  升級選項：
    - Alibaba-NLP/gte-multilingual-base（768 維，對台語效果更佳）
    - text-embedding-3-small（OpenAI API，需網路）
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class DocumentChunk:
    """單一知識庫文字片段。"""

    chunk_id: str              # 唯一 ID，格式：{source}_{index}
    text: str                  # 片段文字內容
    source: str                # 來源檔案名稱
    page: int = 0              # 頁碼（PDF 用），Markdown 預設 0
    metadata: Dict = field(default_factory=dict)  # 額外元資料（標題、章節等）


class TaiwaneseDocumentSplitter:
    """
    台語文件專用文字切塊器。

    針對台語漢字文件做了以下優化：
      1. 以段落分隔符（空行）優先切分，保留語意完整性
      2. chunk_size 以字元計，不用詞數（台語無空格分詞）
      3. 保留標題資訊作為 chunk metadata
    """

    # 切分優先順序：段落 > 換行 > 句尾標點
    _SEPARATORS = ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]

    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """
        遞迴切塊。

        先嘗試以最大分隔符（\n\n）切分，若片段仍超過 chunk_size，
        再嘗試下一層分隔符，直到所有片段都在大小範圍內。
        """
        return self._split_recursive(text, self._SEPARATORS)

    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        if not separators:
            return self._split_by_size(text)

        sep = separators[0]
        remaining = separators[1:]

        if sep == "":
            return self._split_by_size(text)

        splits = text.split(sep)
        chunks: List[str] = []
        current = ""

        for split in splits:
            candidate = (current + sep + split).strip() if current else split.strip()
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # 若單一 split 本身超過限制，遞迴處理
                if len(split) > self.chunk_size:
                    sub_chunks = self._split_recursive(split, remaining)
                    chunks.extend(sub_chunks)
                    current = ""
                else:
                    current = split.strip()

        if current:
            chunks.append(current)

        # 加入 overlap：每個 chunk 前附加上一個 chunk 的最後 overlap 字元
        if self.chunk_overlap > 0:
            overlapped = []
            for i, chunk in enumerate(chunks):
                if i > 0:
                    overlap_text = chunks[i - 1][-self.chunk_overlap :]
                    chunk = overlap_text + chunk
                overlapped.append(chunk)
            return overlapped

        return chunks

    def _split_by_size(self, text: str) -> List[str]:
        """強制以字元數切分（最後手段）。"""
        return [
            text[i : i + self.chunk_size]
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
        ]


class DocumentLoader:
    """Markdown 與 PDF 文件載入器。"""

    @staticmethod
    def load_markdown(file_path: str | Path) -> str:
        """
        載入 Markdown 文件，保留標題結構。
        去除 YAML frontmatter（---...---）、程式碼區塊等非語意內容。
        """
        with open(file_path, encoding="utf-8") as f:
            text = f.read()

        # 移除 YAML frontmatter
        text = re.sub(r"^---[\s\S]+?---\n", "", text)
        # 移除程式碼區塊（不含台語知識）
        text = re.sub(r"```[\s\S]*?```", "", text)
        # 移除 HTML 標籤
        text = re.sub(r"<[^>]+>", "", text)
        # 移除過多空行
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    @staticmethod
    def load_pdf(file_path: str | Path) -> str:
        """
        載入 PDF 文件。

        使用 pdfminer.six（輕量，不需 Java），
        若字型嵌入問題造成亂碼，可改用 pypdf 或 pdfplumber。
        """
        try:
            from pdfminer.high_level import extract_text as pdf_extract
            text = pdf_extract(str(file_path))
        except ImportError:
            try:
                import pypdf
                reader = pypdf.PdfReader(str(file_path))
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
            except ImportError:
                raise ImportError("請安裝：pip install pdfminer.six 或 pypdf")

        # 清理 PDF 常見垃圾字元
        text = re.sub(r"\x0c", "\n", text)       # 換頁符
        text = re.sub(r"[ \t]{2,}", " ", text)    # 多餘空白
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()


class KnowledgeIngestion:
    """
    知識庫攝取主流程控制器。

    使用方式
    --------
    # 初始化
    ingestor = KnowledgeIngestion(persist_dir="./data/chroma_db")

    # 攝取 Markdown 文件
    ingestor.ingest_file("data/knowledge/care_guide_tw.md")
    ingestor.ingest_directory("data/knowledge/")

    # 建立 ChromaDB 向量庫
    ingestor.build_vectorstore()
    """

    def __init__(
        self,
        persist_dir: str = "./data/chroma_db",
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        chunk_size: int = 300,
        chunk_overlap: int = 50,
    ):
        self.persist_dir = Path(persist_dir)
        self.embedding_model_name = embedding_model
        self.splitter = TaiwaneseDocumentSplitter(chunk_size, chunk_overlap)
        self.loader = DocumentLoader()
        self._chunks: List[DocumentChunk] = []
        self._embedding_model = None

    def _get_embedding_model(self):
        """延遲載入 Sentence-Transformers 模型。"""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                print(f"[RAG] 載入 Embedding 模型：{self.embedding_model_name}")
                self._embedding_model = SentenceTransformer(self.embedding_model_name)
                print("[RAG] Embedding 模型載入完成 ✓")
            except ImportError:
                raise ImportError("請安裝：pip install sentence-transformers")
        return self._embedding_model

    def ingest_file(self, file_path: str | Path) -> int:
        """
        攝取單一文件，返回切塊數量。

        支援：.md / .txt（Markdown）、.pdf
        """
        path = Path(file_path)
        print(f"[RAG] 攝取文件：{path.name}")

        if path.suffix.lower() in (".md", ".txt"):
            raw_text = self.loader.load_markdown(path)
        elif path.suffix.lower() == ".pdf":
            raw_text = self.loader.load_pdf(path)
        else:
            raise ValueError(f"不支援的檔案格式：{path.suffix}")

        # 切塊
        text_chunks = self.splitter.split_text(raw_text)

        for i, chunk_text in enumerate(text_chunks):
            if len(chunk_text.strip()) < 20:   # 過短片段跳過
                continue
            chunk = DocumentChunk(
                chunk_id=f"{path.stem}_{i}",
                text=chunk_text.strip(),
                source=path.name,
                metadata={"file": path.name, "chunk_index": i},
            )
            self._chunks.append(chunk)

        print(f"[RAG] {path.name} → {len(text_chunks)} 個片段")
        return len(text_chunks)

    def ingest_directory(self, dir_path: str | Path, extensions: tuple = (".md", ".txt", ".pdf")):
        """遞迴攝取目錄下的所有支援文件。"""
        dir_path = Path(dir_path)
        files = [f for ext in extensions for f in dir_path.rglob(f"*{ext}")]

        if not files:
            print(f"[RAG] ⚠ 在 {dir_path} 找不到文件")
            return

        for file in files:
            self.ingest_file(file)

        print(f"[RAG] 共攝取 {len(self._chunks)} 個知識片段")

    def build_vectorstore(self) -> "chromadb.Collection":
        """
        將所有攝取的片段向量化並存入 ChromaDB。

        ChromaDB 持久化：
          - 資料儲存於 persist_dir（SQLite + 向量檔）
          - 重啟後無需重新 embed，直接讀取
          - 適合開發測試；生產可遷移至 Qdrant 或 Weaviate
        """
        try:
            import chromadb
            from chromadb.utils import embedding_functions
        except ImportError:
            raise ImportError("請安裝：pip install chromadb")

        if not self._chunks:
            raise ValueError("尚未攝取任何文件，請先呼叫 ingest_file()")

        # 建立 ChromaDB 客戶端（持久化模式）
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=str(self.persist_dir))

        # 使用 Sentence-Transformers Embedding Function
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model_name
        )

        # 建立或取得 collection
        collection = chroma_client.get_or_create_collection(
            name="tomni_care_knowledge",
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},   # 向量相似度：餘弦相似度
        )

        # 批次插入（ChromaDB 建議每批 < 5000）
        batch_size = 500
        for i in range(0, len(self._chunks), batch_size):
            batch = self._chunks[i : i + batch_size]
            collection.upsert(
                ids=[c.chunk_id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[c.metadata for c in batch],
            )

        print(f"[RAG] ChromaDB 建立完成！共 {collection.count()} 筆資料")
        print(f"[RAG] 持久化路徑：{self.persist_dir.resolve()}")
        return collection
