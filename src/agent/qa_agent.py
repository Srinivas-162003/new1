import json
from typing import Dict, List, Optional, Tuple

import google.generativeai as genai

from config import GEMINI_API_KEY, GEMINI_MODEL_TEXT, MAX_CONTEXT_CHARS, TOP_K_DEFAULT, require_api_key
from models import AgentAnswer, DocumentRecord, RetrievalResult, ToolCall
from retrieval.vector_store import VectorStore
from tools.arxiv_tool import format_arxiv_results, search_arxiv
from agent.prompts import SYSTEM_PROMPT, TOOL_ROUTER_PROMPT


def answer_query(
    query: str,
    store: VectorStore,
    documents: List[DocumentRecord],
    enable_arxiv: bool = True,
    top_k: int = TOP_K_DEFAULT,
) -> AgentAnswer:
    require_api_key()
    genai.configure(api_key=GEMINI_API_KEY)

    tool_calls: List[ToolCall] = []
    extra: Dict[str, str] = {}

    if enable_arxiv:
        tool_call = _route_tool_call(query)
        if tool_call and tool_call.tool == "arxiv_search":
            tool_calls.append(tool_call)
            results = search_arxiv(tool_call.args.get("query", query))
            extra["arxiv"] = format_arxiv_results(results)

    results = _retrieve_context(query, store, documents, top_k)
    context_text, citations = _build_context(results)

    model = genai.GenerativeModel(GEMINI_MODEL_TEXT, system_instruction=SYSTEM_PROMPT)
    response = model.generate_content(
        [
            f"Context:\n{context_text}",
            f"Question: {query}",
            "Answer with citations in the form [doc_id:page].",
        ]
    )
    answer_text = response.text.strip() if response.text else ""

    return AgentAnswer(answer=answer_text, citations=citations, tool_calls=tool_calls, extra=extra or None)


def _route_tool_call(query: str) -> Optional[ToolCall]:
    model = genai.GenerativeModel(GEMINI_MODEL_TEXT, system_instruction=TOOL_ROUTER_PROMPT)
    response = model.generate_content([f"Query: {query}"])
    raw = response.text.strip() if response.text else ""

    try:
        payload = json.loads(raw)
        tool = payload.get("tool", "none")
        args = payload.get("args", {})
        if tool == "arxiv_search":
            if "query" not in args:
                args["query"] = query
            return ToolCall(tool=tool, args=args)
    except json.JSONDecodeError:
        return None

    return None


def _retrieve_context(
    query: str, store: VectorStore, documents: List[DocumentRecord], top_k: int
) -> List[RetrievalResult]:
    doc_filter = _detect_doc_filter(query, documents)
    if not doc_filter:
        return store.search(query, top_k)

    initial_results = store.search(query, max(top_k * 3, top_k))
    filtered = [result for result in initial_results if result.chunk.metadata.get("doc_id") == doc_filter]
    return filtered[:top_k]


def _detect_doc_filter(query: str, documents: List[DocumentRecord]) -> Optional[str]:
    q = query.lower()
    for doc in documents:
        if doc.title.lower() in q or doc.doc_id.lower() in q:
            return doc.doc_id
    return None


def _build_context(results: List[RetrievalResult]) -> Tuple[str, List[str]]:
    blocks: List[str] = []
    citations: List[str] = []
    total_len = 0

    for result in results:
        meta = result.chunk.metadata
        citation = f"{meta.get('doc_id')}:{meta.get('page')}"
        text = result.chunk.text
        block = f"[{citation}]\n{text}"
        if total_len + len(block) > MAX_CONTEXT_CHARS:
            break
        blocks.append(block)
        citations.append(citation)
        total_len += len(block)

    return "\n\n".join(blocks), citations
