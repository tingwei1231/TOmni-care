"""
src/agent/tools.py — Tool Calling 工具集

工具說明：
  search_knowledge_base — 查詢 ChromaDB 台語照護知識庫（醫療/生活問題觸發）
  get_current_time      — 台語格式時間（長者常問幾點了）
  get_emergency_contacts— 緊急聯絡資訊（119/1966/1925）

get_tool_registry() 返回 LangChain Tool 列表，供 LangGraph ToolNode 或 bind_tools 使用。
"""
from __future__ import annotations
from datetime import datetime
from typing import Optional


def search_knowledge_base(query: str, top_k: int = 3) -> str:
    """搜尋台語長者照護知識庫，返回最相關段落。"""
    try:
        import asyncio
        from ..rag.retriever import ChromaRetriever
        retriever = ChromaRetriever(top_k=top_k)
        chunks = asyncio.get_event_loop().run_until_complete(retriever.retrieve(query))
        return retriever.format_context(chunks) if chunks else "知識庫中查無相關資料，建議諮詢醫師。"
    except Exception as e:
        return f"知識庫查詢失敗：{e}"


def get_current_time() -> str:
    """取得當前台灣時間（台語格式）。"""
    now = datetime.now()
    wd = ["禮拜一","禮拜二","禮拜三","禮拜四","禮拜五","禮拜六","禮拜日"][now.weekday()]
    h = now.hour
    period = ("半暝" if h<6 else "早起" if h<12 else "中晝" if h<14 else
              "下晡" if h<18 else "暗時" if h<22 else "暝頭")
    return f"今仔日是 {now.year} 年 {now.month} 月 {now.day} 日，{wd}，{period} {h} 點 {now.minute:02d} 分。"


def get_emergency_contacts(situation: str = "general") -> str:
    """取得緊急聯絡資訊（medical/elderly_care/general）。"""
    contacts = {
        "medical": "🚨 請撥 119（救護車）\n非緊急：1966（長照服務）",
        "elderly_care": "長照服務：1966\n老人諮詢：1925（週一~五 8:00-20:00）",
        "general": "緊急：119 ｜ 警察：110 ｜ 長照：1966 ｜ 心理：1925",
    }
    return contacts.get(situation, contacts["general"])


def get_tool_registry() -> list:
    """
    返回 LangChain Tool 列表。

    用法：
      # ToolNode 自動呼叫
      from langgraph.prebuilt import ToolNode
      tool_node = ToolNode(get_tool_registry())

      # 直接 bind 給 LLM（Function Calling）
      from langchain_groq import ChatGroq
      llm = ChatGroq(model="llama-3.3-70b-versatile").bind_tools(get_tool_registry())
    """
    try:
        from langchain_core.tools import tool
        return [tool(search_knowledge_base), tool(get_current_time), tool(get_emergency_contacts)]
    except ImportError:
        return [search_knowledge_base, get_current_time, get_emergency_contacts]
