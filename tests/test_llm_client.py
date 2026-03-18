"""
tests/test_llm_client.py
========================
LLM 客戶端測試。

包含：
  - Mock 測試（不消耗 API token，CI/CD 可用）
  - 真實 API 測試（需 GROQ_API_KEY，手動執行）
  - Ollama 本地測試（需本地 Ollama 運行）

執行：
  # 只跑 mock 測試（快速）
  pytest tests/test_llm_client.py -v -m "not real_api"

  # 跑真實 Groq API 測試（需 GROQ_API_KEY）
  pytest tests/test_llm_client.py -v -m "real_api"
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from src.llm.client import LLMBackend, LLMClient, LLMConfig, Message
from src.llm.prompts import FEW_SHOT_EXAMPLES, PromptTemplate, SYSTEM_PROMPT_TW


# ══════════════════════════════════════════════════════════════
# PromptTemplate 單元測試（不需網路）
# ══════════════════════════════════════════════════════════════

class TestPromptTemplate:
    def test_build_basic(self):
        """基本 Prompt 組裝測試。"""
        template = PromptTemplate(use_few_shot=False)
        messages = template.build("腹肚痛，是按怎？")
        assert len(messages) >= 2  # system + user
        assert messages[0].role == "system"
        assert messages[-1].role == "user"
        assert "腹肚痛" in messages[-1].content

    def test_few_shot_injection(self):
        """Few-shot 範例注入測試。"""
        template = PromptTemplate(use_few_shot=True, n_few_shot=2)
        messages = template.build("我頭殼暈。")
        # system + 2 few-shot pairs (4 messages) + user = 6
        assert len(messages) == 6
        # Few-shot 格式應為 user/assistant 交替
        roles = [m.role for m in messages[1:-1]]
        assert roles == ["user", "assistant", "user", "assistant"]

    def test_rag_context_injection(self):
        """RAG 內容注入測試。"""
        template = PromptTemplate(use_few_shot=False)
        rag_text = "腹肚痛需補充水分，若持續超過2小時需就醫。"
        messages = template.build("腹肚痛", rag_context=rag_text)
        system_content = messages[0].content
        assert "相關知識庫資料" in system_content
        assert rag_text in system_content

    def test_rag_context_truncation(self):
        """RAG 內容超過 max_context_chars 時應截斷。"""
        template = PromptTemplate(use_few_shot=False, max_context_chars=50)
        long_rag = "X" * 200
        messages = template.build("test", rag_context=long_rag)
        system_content = messages[0].content
        # 截斷到 max_context_chars
        assert "X" * 51 not in system_content

    def test_conversation_history(self):
        """多輪對話歷史注入測試。"""
        template = PromptTemplate(use_few_shot=False)
        history = [
            Message(role="user", content="你好"),
            Message(role="assistant", content="你好！有啥物事？"),
        ]
        messages = template.build("腹肚痛", conversation_history=history)
        # system + 2 history + user = 4
        assert len(messages) == 4
        assert messages[1].content == "你好"
        assert messages[2].content == "你好！有啥物事？"

    def test_system_prompt_contains_taiwanese(self):
        """System Prompt 應包含台語關鍵詞彙。"""
        assert "台語" in SYSTEM_PROMPT_TW
        assert "腹肚痛" in SYSTEM_PROMPT_TW
        assert "病院" in SYSTEM_PROMPT_TW

    def test_few_shot_examples_format(self):
        """Few-shot 範例格式驗證。"""
        for ex in FEW_SHOT_EXAMPLES:
            assert "user" in ex
            assert "assistant" in ex
            assert len(ex["user"]) > 0
            assert len(ex["assistant"]) > 0


# ══════════════════════════════════════════════════════════════
# LLM 客戶端 Mock 測試（不消耗 API token）
# ══════════════════════════════════════════════════════════════

class TestLLMClientMock:
    """使用 Mock 模擬 API 回應，不需真實 API 金鑰。"""

    @pytest.mark.asyncio
    async def test_groq_chat_mock(self):
        """Mock Groq API 回應測試（不需 groq 套件已安裝）。"""
        mock_response_text = "唅，腹肚痛（pak-tóo thiànn）是真無爽的事。"

        # 構造模擬 Groq response 結構
        mock_choice = MagicMock()
        mock_choice.message.content = mock_response_text
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]

        mock_groq_instance = MagicMock()
        mock_groq_instance.chat.completions.create = AsyncMock(
            return_value=mock_completion
        )

        config = LLMConfig(
            backend=LLMBackend.GROQ,
            groq_api_key="mock-key-for-testing",
        )
        client = LLMClient(config)
        # 直接注入 mock，不透過 import patch（不需 groq 套件）
        client._groq_client = mock_groq_instance

        messages = [Message(role="user", content="腹肚痛，是按怎？")]
        result = await client._chat_groq(messages)
        assert result == mock_response_text


    @pytest.mark.asyncio
    async def test_fallback_mechanism(self):
        """測試多後端自動降級邏輯。"""
        config = LLMConfig(backend=LLMBackend.GROQ)
        client = LLMClient(config)

        call_order = []

        async def mock_chat_groq(messages):
            call_order.append("groq")
            raise ConnectionError("Groq 斷線")

        async def mock_chat_openrouter(messages):
            call_order.append("openrouter")
            raise ConnectionError("OpenRouter 斷線")

        async def mock_chat_ollama(messages):
            call_order.append("ollama")
            return "我是 Ollama 的回覆"

        client._chat_groq = mock_chat_groq
        client._chat_openrouter = mock_chat_openrouter
        client._chat_ollama = mock_chat_ollama

        messages = [Message(role="user", content="測試")]
        result = await client.chat_with_fallback(messages)

        assert call_order == ["groq", "openrouter", "ollama"]
        assert result == "我是 Ollama 的回覆"


# ══════════════════════════════════════════════════════════════
# 真實 API 測試（需 GROQ_API_KEY，手動執行）
# ══════════════════════════════════════════════════════════════

@pytest.mark.real_api
class TestLLMClientRealAPI:
    """
    真實 Groq API 測試。
    執行前需設定環境變數：export GROQ_API_KEY=your_key
    執行：pytest tests/test_llm_client.py -m "real_api" -v
    """

    @pytest.mark.asyncio
    async def test_groq_simple_chat(self):
        """真實 Groq API 基本對話測試。"""
        import os
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            pytest.skip("GROQ_API_KEY 未設定")

        config = LLMConfig(
            backend=LLMBackend.GROQ,
            groq_api_key=api_key,
            max_tokens=100,
        )
        client = LLMClient(config)
        template = PromptTemplate(use_few_shot=True, n_few_shot=1)
        messages = template.build("你好，你是啥物？")

        response = await client.chat(messages)
        assert isinstance(response, str)
        assert len(response) > 10
        print(f"\n✅ Groq 回覆：{response}")
