from typing import List


def normalize_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    cleaned = [line for line in lines if line]
    return "\n".join(cleaned)


def chunk_text(text: str, max_chars: int, overlap_chars: int) -> List[str]:
    if max_chars <= 0:
        return [text]

    chunks: List[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + max_chars, text_len)
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= text_len:
            break
        start = max(0, end - overlap_chars)

    return chunks
