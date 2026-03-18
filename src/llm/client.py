"""
src/llm/client.py
=================
多後端 LLM 客戶端：Groq API / OpenRouter / Ollama（本地）

【後端選擇策略】
  優先：Groq API
    - 免費 tier：Llama-3.3-70B 每分鐘 30 次請求
    - 延遲極低（~200ms），適合即時對話
    - 不需本地 GPU

  次選：OpenRouter
    - 統一 API 接口，可呼叫 Llama-3-Taiwan、Qwen、Mistral 等
    - 按 token 付費，比 OpenAI 便宜

  備選：Ollama（本地 4-bit 量化）
    - 完全離線，適合醫療資料隱私需求
    - 需本地 GPU（至少 8GB VRAM）或 CPU 用量（速度慢）
    - 啟動：ollama run llama3.2

【Llama-3-Taiwan 說明】
  yentinglin/Llama-3-Taiwan 為台大林彥廷團隊基於 Meta Llama-3 微調的
  繁體中文模型，在台灣本土 NLP 任務（閩南語理解、繁中生成）遠優於原版。
  Groq 上可透過 llama-3.3-70b-versatile 替代（待官方上架 Taiwan 版本）。
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, List, Optional


class LLMBackend(str, Enum):
    GROQ = "groq"
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"


@dataclass
class Message:
    """單一對話訊息（OpenAI 相容格式）。"""
    role: str   # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMConfig:
    """LLM 客戶端設定。"""
    backend: LLMBackend = LLMBackend.GROQ
    # ── Groq ──────────────────────────────────────────────────
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_model: str = "llama-3.3-70b-versatile"   # 目前 Groq 上最接近 Taiwan 版本
    # ── OpenRouter ────────────────────────────────────────────
    openrouter_api_key: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY", "")
    )
    openrouter_model: str = "meta-llama/llama-3.1-70b-instruct"
    # ── Ollama（本地）─────────────────────────────────────────
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"   # ollama pull llama3.2 取得
    # ── 共用推論參數 ───────────────────────────────────────────
    temperature: float = 0.7        # 台語口語適合稍高 temperature 增加自然感
    max_tokens: int = 512
    top_p: float = 0.9
    stream: bool = True             # 預設串流輸出，降低首字延遲感


class LLMClient:
    """
    統一 LLM 客戶端，透過設定自動切換後端。

    使用方式
    --------
    >>> cfg = LLMConfig(backend=LLMBackend.GROQ)
    >>> client = LLMClient(cfg)
    >>> response = await client.chat([
    ...     Message("system", system_prompt),
    ...     Message("user", "腹肚痛，請問我愛按怎？"),
    ... ])
    >>> print(response)
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._groq_client = None
        self._openrouter_client = None
        self._ollama_client = None

    # ── 延遲初始化各後端客戶端 ────────────────────────────────
    def _get_groq(self):
        if self._groq_client is None:
            try:
                from groq import AsyncGroq
                if not self.config.groq_api_key:
                    raise ValueError("GROQ_API_KEY 未設定，請在 .env 填入")
                self._groq_client = AsyncGroq(api_key=self.config.groq_api_key)
            except ImportError:
                raise ImportError("請安裝：pip install groq")
        return self._groq_client

    def _get_openrouter(self):
        if self._openrouter_client is None:
            try:
                from openai import AsyncOpenAI   # OpenRouter 相容 OpenAI SDK
                if not self.config.openrouter_api_key:
                    raise ValueError("OPENROUTER_API_KEY 未設定")
                self._openrouter_client = AsyncOpenAI(
                    api_key=self.config.openrouter_api_key,
                    base_url="https://openrouter.ai/api/v1",
                    default_headers={
                        "HTTP-Referer": "https://github.com/TOmni-care",
                        "X-Title": "TOmni-Care Agent",
                    },
                )
            except ImportError:
                raise ImportError("請安裝：pip install openai")
        return self._openrouter_client

    def _get_ollama(self):
        if self._ollama_client is None:
            try:
                import ollama
                self._ollama_client = ollama.AsyncClient(host=self.config.ollama_host)
            except ImportError:
                raise ImportError(
                    "請安裝：pip install ollama\n"
                    "並啟動本地 Ollama：ollama serve\n"
                    "然後拉取模型：ollama pull llama3.2"
                )
        return self._ollama_client

    # ── 訊息格式轉換工具 ──────────────────────────────────────
    @staticmethod
    def _to_dicts(messages: List[Message]) -> List[dict]:
        return [{"role": m.role, "content": m.content} for m in messages]

    # ── 核心推論：Groq ────────────────────────────────────────
    async def _chat_groq(self, messages: List[Message]) -> str:
        client = self._get_groq()
        response = await client.chat.completions.create(
            model=self.config.groq_model,
            messages=self._to_dicts(messages),
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            stream=False,   # 非串流版，簡化首次實作
        )
        return response.choices[0].message.content

    async def _stream_groq(self, messages: List[Message]) -> AsyncIterator[str]:
        client = self._get_groq()
        stream = await client.chat.completions.create(
            model=self.config.groq_model,
            messages=self._to_dicts(messages),
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    # ── 核心推論：OpenRouter ──────────────────────────────────
    async def _chat_openrouter(self, messages: List[Message]) -> str:
        client = self._get_openrouter()
        response = await client.chat.completions.create(
            model=self.config.openrouter_model,
            messages=self._to_dicts(messages),
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return response.choices[0].message.content

    async def _stream_openrouter(self, messages: List[Message]) -> AsyncIterator[str]:
        client = self._get_openrouter()
        stream = await client.chat.completions.create(
            model=self.config.openrouter_model,
            messages=self._to_dicts(messages),
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    # ── 核心推論：Ollama（本地 4-bit）────────────────────────
    async def _chat_ollama(self, messages: List[Message]) -> str:
        """
        Ollama 本地推論。

        4-bit 量化方法：
          預設拉取的模型已是量化版。若需明確指定：
          ollama pull llama3.2:8b-instruct-q4_K_M
          （q4_K_M 為品質與速度最佳平衡的量化格式）
        """
        client = self._get_ollama()
        response = await client.chat(
            model=self.config.ollama_model,
            messages=self._to_dicts(messages),
            options={
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
                "top_p": self.config.top_p,
            },
        )
        return response["message"]["content"]

    async def _stream_ollama(self, messages: List[Message]) -> AsyncIterator[str]:
        client = self._get_ollama()
        async for chunk in await client.chat(
            model=self.config.ollama_model,
            messages=self._to_dicts(messages),
            stream=True,
        ):
            content = chunk.get("message", {}).get("content", "")
            if content:
                yield content

    # ── 公開 API ─────────────────────────────────────────────
    async def chat(self, messages: List[Message]) -> str:
        """
        一次性推論，返回完整回覆字串。
        適用於 RAG 查詢結果綜合、非即時場景。
        """
        backend = self.config.backend
        if backend == LLMBackend.GROQ:
            return await self._chat_groq(messages)
        elif backend == LLMBackend.OPENROUTER:
            return await self._chat_openrouter(messages)
        elif backend == LLMBackend.OLLAMA:
            return await self._chat_ollama(messages)
        raise ValueError(f"未知後端：{backend}")

    async def stream(self, messages: List[Message]) -> AsyncIterator[str]:
        """
        串流推論，逐 token yield 字串。
        適用於即時對話（降低首字延遲感）。
        """
        backend = self.config.backend
        if backend == LLMBackend.GROQ:
            async for token in self._stream_groq(messages):
                yield token
        elif backend == LLMBackend.OPENROUTER:
            async for token in self._stream_openrouter(messages):
                yield token
        elif backend == LLMBackend.OLLAMA:
            async for token in self._stream_ollama(messages):
                yield token
        else:
            raise ValueError(f"未知後端：{backend}")

    async def chat_with_fallback(self, messages: List[Message]) -> str:
        """
        自動降級（Groq → OpenRouter → Ollama）。
        當主要後端 API 限流或斷線時，自動嘗試下一個。
        """
        fallback_order = [LLMBackend.GROQ, LLMBackend.OPENROUTER, LLMBackend.OLLAMA]
        original_backend = self.config.backend

        for backend in fallback_order:
            try:
                self.config.backend = backend
                result = await self.chat(messages)
                return result
            except Exception as e:
                print(f"[LLM] {backend.value} 失敗：{e}，嘗試下一個後端...")

        self.config.backend = original_backend
        raise RuntimeError("所有 LLM 後端均失敗，請檢查 API 金鑰與網路連線")
