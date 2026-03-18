"""
src/llm/pipeline.py
===================
第二階段整合管線：RAG + LLM 完整 QA Pipeline。

【架構】
  使用者台語輸入
    → ChromaRetriever（台語查詢擴展 + 向量搜尋 + Reranking）
    → PromptTemplate（System Prompt + RAG Context + Few-shot 注入）
    → LLMClient（Groq / OpenRouter / Ollama）
    → 台語回覆輸出

【使用方式】
  # 方式 A：直接對話（無 RAG）
  pipeline = RAGPipeline(use_rag=False)
  response = await pipeline.chat("腹肚痛，是按怎？")

  # 方式 B：RAG 增強（需先建立 ChromaDB）
  pipeline = RAGPipeline(use_rag=True)
  response = await pipeline.chat("食藥仔有啥物注意事項？")

  # 方式 C：守護情緒安撫模式
  response = await pipeline.chat("我真寂寞", comfort_mode=True)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import AsyncIterator, List, Optional

from .client import LLMBackend, LLMClient, LLMConfig, Message
from .prompts import PromptTemplate, build_comfort_prompt


@dataclass
class ConversationTurn:
    """單輪對話紀錄。"""
    user: str
    assistant: str
    rag_sources: List[str] = field(default_factory=list)  # 引用的知識庫來源


class RAGPipeline:
    """
    RAG + LLM 整合管線，第三階段 LangGraph Agent 的核心工具之一。
    """

    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        use_rag: bool = True,
        persist_dir: str = "./data/chroma_db",
        max_history_turns: int = 6,   # 保留最近 N 輪對話歷史
    ):
        # LLM 客戶端
        self.llm = LLMClient(llm_config or LLMConfig())
        self.prompt_template = PromptTemplate()

        # RAG 設定
        self.use_rag = use_rag
        self._retriever = None
        self.persist_dir = persist_dir

        # 對話歷史（多輪記憶）
        self.history: List[ConversationTurn] = []
        self.max_history_turns = max_history_turns

    def _get_retriever(self):
        """延遲初始化 ChromaRetriever。"""
        if self._retriever is None:
            from ..rag.retriever import ChromaRetriever
            self._retriever = ChromaRetriever(persist_dir=self.persist_dir)
        return self._retriever

    def _build_history_messages(self) -> List[Message]:
        """將對話歷史轉為 Message 列表。"""
        messages = []
        for turn in self.history[-self.max_history_turns :]:
            messages.append(Message(role="user", content=turn.user))
            messages.append(Message(role="assistant", content=turn.assistant))
        return messages

    async def chat(
        self,
        user_input: str,
        comfort_mode: bool = False,
        stream: bool = False,
    ) -> str:
        """
        單輪對話推論（含 RAG + 歷史 + 角色 Prompt）。

        Parameters
        ----------
        user_input : str
            使用者台語輸入
        comfort_mode : bool
            True = 情緒安撫模式（由 LangGraph 情緒節點觸發）
        stream : bool
            True = 串流輸出（Gradio / WebSocket 整合用）

        Returns
        -------
        str
            完整的台語回覆字串
        """
        # ① RAG 檢索
        rag_context = ""
        rag_sources: List[str] = []
        if self.use_rag:
            try:
                retriever = self._get_retriever()
                chunks = await retriever.retrieve(user_input)
                if chunks:
                    rag_context = retriever.format_context(chunks)
                    rag_sources = list({c.source for c in chunks})
            except Exception as e:
                print(f"[Pipeline] RAG 檢索失敗（{e}），以純 LLM 模式繼續")

        # ② 組裝 Prompt
        history_messages = self._build_history_messages()
        if comfort_mode:
            messages = build_comfort_prompt(user_input, rag_context)
            # 在 comfort prompt 後注入對話歷史
            messages = messages[:-1] + history_messages + [messages[-1]]
        else:
            messages = self.prompt_template.build(
                user_input,
                rag_context=rag_context,
                conversation_history=history_messages,
            )

        # ③ LLM 推論
        if stream:
            # 串流模式：逐 token 收集後存歷史
            tokens = []
            async for token in self.llm.stream(messages):
                tokens.append(token)
                print(token, end="", flush=True)  # 終端輸出（Gradio 替換為 yield）
            response = "".join(tokens)
        else:
            response = await self.llm.chat(messages)

        # ④ 儲存對話歷史
        self.history.append(
            ConversationTurn(
                user=user_input,
                assistant=response,
                rag_sources=rag_sources,
            )
        )

        return response

    async def stream_chat(
        self,
        user_input: str,
        comfort_mode: bool = False,
    ) -> AsyncIterator[str]:
        """
        串流版 chat，逐 token yield（供 Gradio / WebSocket 使用）。
        """
        # RAG 檢索
        rag_context = ""
        rag_sources: List[str] = []
        if self.use_rag:
            try:
                retriever = self._get_retriever()
                chunks = await retriever.retrieve(user_input)
                if chunks:
                    rag_context = retriever.format_context(chunks)
                    rag_sources = list({c.source for c in chunks})
            except Exception as e:
                print(f"[Pipeline] RAG 失敗：{e}")

        # 組裝 Prompt
        history_messages = self._build_history_messages()
        if comfort_mode:
            messages = build_comfort_prompt(user_input, rag_context)
            messages = messages[:-1] + history_messages + [messages[-1]]
        else:
            messages = self.prompt_template.build(
                user_input,
                rag_context=rag_context,
                conversation_history=history_messages,
            )

        # 串流推論，同時收集完整回覆
        full_response_tokens = []
        async for token in self.llm.stream(messages):
            full_response_tokens.append(token)
            yield token

        # 儲存歷史
        self.history.append(
            ConversationTurn(
                user=user_input,
                assistant="".join(full_response_tokens),
                rag_sources=rag_sources,
            )
        )

    def reset_history(self):
        """清除對話歷史（新對話開始時呼叫）。"""
        self.history = []


# ── 快速測試腳本（直接執行此檔案）────────────────────────────
async def _demo():
    """
    第二階段快速驗證腳本。
    需先設定 GROQ_API_KEY 環境變數。
    """
    import os
    from dotenv import load_dotenv
    load_dotenv()

    config = LLMConfig(
        backend=LLMBackend.GROQ,
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
    )

    # 不使用 RAG 的快速測試
    pipeline = RAGPipeline(llm_config=config, use_rag=False)

    test_inputs = [
        "腹肚痛，是按怎？",
        "我頭殼真暈。",
        "藥仔食完了，愛按怎辦？",
    ]

    print("=" * 50)
    print("TOmni-Care 第二階段 LLM Pipeline 測試")
    print("=" * 50)

    for user_input in test_inputs:
        print(f"\n📣 使用者：{user_input}")
        response = await pipeline.chat(user_input)
        print(f"🤖 TOmni-Care：{response}")
        print("-" * 40)


if __name__ == "__main__":
    asyncio.run(_demo())
