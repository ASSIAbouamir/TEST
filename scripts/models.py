from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
import json

@dataclass
class LegalNode:
    node_id: str
    text: str
    summary: str
    country: str
    law_name: str
    authority_level: float = 0.5  # 0.0 to 1.0 (e.g., Constitution=1.0, Decree=0.5)
    valid_from: str = "Unknown"
    valid_to: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    qa_support: List[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

@dataclass
class LegalEdge:
    source_id: str
    target_id: str
    relation_type: str  # defines, sanctions, references, excepts
    description: Optional[str] = None
    weight: float = 1.0

    def to_dict(self):
        return asdict(self)

@dataclass
class RetrievalTrace:
    query: str
    bm25_hits: List[Dict[str, Any]] = field(default_factory=list)
    dense_hits: List[Dict[str, Any]] = field(default_factory=list)
    graph_hops: List[Dict[str, Any]] = field(default_factory=list)
    final_scores: Dict[str, float] = field(default_factory=dict)
    reasoning: str = ""

    def to_json(self):
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

@dataclass
class LegalWorkflow:
    workflow_id: str
    steps: List[Dict[str, Any]]
    jurisdiction: str
    description: str

    def to_dict(self):
        return asdict(self)
