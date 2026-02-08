from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SectionRecord:
    title: str
    page: int
    content: str
    tables: str = ""
    vision_notes: str = ""


@dataclass
class DocumentRecord:
    doc_id: str
    title: str
    path: str
    sections: List[SectionRecord] = field(default_factory=list)


@dataclass
class Chunk:
    text: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "metadata": self.metadata}

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Chunk":
        return Chunk(text=data["text"], metadata=data["metadata"])


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float


@dataclass
class ToolCall:
    tool: str
    args: Dict[str, Any]


@dataclass
class AgentAnswer:
    answer: str
    citations: List[str] = field(default_factory=list)
    tool_calls: List[ToolCall] = field(default_factory=list)
    extra: Optional[Dict[str, Any]] = None
