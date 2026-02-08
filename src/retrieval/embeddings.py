from typing import List
import google.generativeai as genai

from config import GEMINI_API_KEY, GEMINI_MODEL_EMBED, require_api_key


def embed_texts(texts: List[str]) -> List[List[float]]:
    require_api_key()
    genai.configure(api_key=GEMINI_API_KEY)
    vectors: List[List[float]] = []
    for text in texts:
        result = genai.embed_content(
            model=GEMINI_MODEL_EMBED,
            content=text,
            task_type="RETRIEVAL_DOCUMENT",
        )
        vectors.append(result["embedding"])
    return vectors


def embed_query(text: str) -> List[float]:
    require_api_key()
    genai.configure(api_key=GEMINI_API_KEY)
    result = genai.embed_content(
        model=GEMINI_MODEL_EMBED,
        content=text,
        task_type="RETRIEVAL_QUERY",
    )
    return result["embedding"]
