"""Async Tavily search wrappers for worker agents."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from tavily import AsyncTavilyClient


_tavily_client: AsyncTavilyClient | None = None


def get_tavily_client() -> AsyncTavilyClient:
    global _tavily_client
    if _tavily_client is None:
        api_key = os.environ.get("TAVILY_API_KEY", "")
        if not api_key:
            raise RuntimeError("TAVILY_API_KEY environment variable not set")
        _tavily_client = AsyncTavilyClient(api_key=api_key)
    return _tavily_client


async def search_web(
    query: str,
    search_depth: str = "basic",
    topic: str = "general",
    max_results: int = 5,
    include_answer: bool = True,
) -> dict[str, Any]:
    client = get_tavily_client()
    return await client.search(
        query=query,
        search_depth=search_depth,
        topic=topic,
        max_results=max_results,
        include_answer=include_answer,
    )


async def search_news(
    query: str,
    max_results: int = 5,
) -> dict[str, Any]:
    client = get_tavily_client()
    return await client.search(
        query=query,
        topic="news",
        max_results=max_results,
        include_answer=True,
    )


async def extract_url(url: str, query: str = "") -> dict[str, Any]:
    client = get_tavily_client()
    return await client.extract(urls=url, query=query)


async def batch_search(queries: list[str], **kwargs: Any) -> list[dict[str, Any]]:
    results = await asyncio.gather(
        *[search_web(q, **kwargs) for q in queries],
        return_exceptions=True,
    )
    # Replace exceptions with empty results
    clean: list[dict[str, Any]] = []
    for r in results:
        if isinstance(r, Exception):
            clean.append({"query": "", "results": [], "answer": None})
        else:
            clean.append(r)
    return clean


def format_search_results(search_responses: list[dict[str, Any]]) -> str:
    """Convert raw Tavily responses into clean text context for LLM prompts."""
    sections: list[str] = []
    for i, resp in enumerate(search_responses, 1):
        if not resp.get("results"):
            continue
        answer = resp.get("answer")
        if answer:
            sections.append(f"[Search {i} Summary] {answer}")
        for j, result in enumerate(resp.get("results", []), 1):
            title = result.get("title", "")
            url = result.get("url", "")
            content = result.get("content", "")
            sections.append(f"[{i}.{j}] {title}\n  URL: {url}\n  {content}")
    return "\n\n".join(sections) if sections else "No search results found."
