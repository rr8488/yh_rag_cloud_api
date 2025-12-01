from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field

class TocNodeIn(BaseModel):
    id: Optional[str] = None               # optional; server can assign if missing
    toc_code: str                          # 'ROOT','1.0','1.1',...
    title: str
    level: int
    sort_key: str = Field(default="9999")
    children: List["TocNodeIn"] = Field(default_factory=list)

TocNodeIn.model_rebuild()

class DocumentIn(BaseModel):
    id: str               # e.g., 'report_2025_10'
    title: str            # e.g., 'October Report'
    month_label: str      # e.g., '2025-10'
    root: TocNodeIn       # full tree

class TocNodeOut(BaseModel):
    id: str
    toc_code: str
    title: str
    level: int
    children: list["TocNodeOut"] = Field(default_factory=list)

TocNodeOut.model_rebuild()

class DocumentOut(BaseModel):
    id: str
    title: str
    month_label: str
    root: TocNodeOut

class RagReq(BaseModel):
    doc_id: str
    question: str
    top_k: int = 15
    llm: Optional[str] = "ollama"

from enum import Enum

class ImpactArea(str, Enum):
    """Enumeration of the 5 allowed Impact Areas."""
    education = "Education"
    community = "Community Development"
    environment = "Environment & Sustainability"
    arts = "Arts & Culture"
    social_enterprise = "Social Enterprise"

class GlobalSearchReq(BaseModel):
    query: str
    top_k: int = 10