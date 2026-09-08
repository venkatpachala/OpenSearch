"""Conservative methodology extraction from real paper text.

An optional structured LLM callable may be supplied.  The deterministic
fallback only records values it can substantiate from text.
"""
from __future__ import annotations
import re
from typing import Any, Callable
from ..memory.models import Methodology

class MethodologyExtractor:
    def __init__(self, llm: Callable[[str], dict[str, Any]] | None = None): self.llm=llm
    def extract(self, paper_id: str, sections: dict[str, str]) -> Methodology:
        text="\n".join(sections.get(k,"") for k in ("methodology","training","results"))
        if self.llm:
            data=self.llm(text); return Methodology(paper_id=paper_id, **{k:v for k,v in data.items() if k in Methodology.model_fields and k != "paper_id"})
        hyper={}
        patterns={"batch_size":r"batch(?: size)?\s*(?:of|=|:)\s*(\d+)","learning_rate":r"(?:learning rate|lr)\s*(?:of|=|:)\s*([0-9.]+)","epochs":r"(\d+)\s*epochs?"}
        for k,p in patterns.items():
            m=re.search(p,text,re.I)
            if m: hyper[k]=float(m.group(1)) if "." in m.group(1) else int(m.group(1))
        metrics={}
        for m in re.finditer(r"(accuracy|f1|precision|recall)\s*(?:of|=|:)\s*(0?\.\d+|\d+(?:\.\d+)?)", text, re.I): metrics[m.group(1).lower()]=float(m.group(2))
        return Methodology(paper_id=paper_id, hyperparameters=hyper, reported_metrics=metrics, training_details={"source": "paper_text"})

