"""PDF-to-sections extraction; no synthetic paper text is generated."""
from __future__ import annotations
from pathlib import Path
import re

class PdfSectionParser:
    SECTION_NAMES = {"abstract": ("abstract",), "methodology": ("method", "approach", "architecture"), "training": ("training", "experimental setup", "implementation details"), "results": ("results", "ablation", "evaluation")}
    def extract(self, pdf_path: Path) -> dict[str, str]:
        try:
            import fitz
        except ImportError as exc: raise RuntimeError("PyMuPDF is required to parse papers") from exc
        doc=fitz.open(pdf_path); pages=[p.get_text("text") for p in doc]; full="\n\n".join(pages)
        sections={"full_text":full}
        headings=re.compile(r"(?im)^\s*((?:\d+(?:\.\d+)*\s+)?[A-Z][A-Za-z /_-]{3,80})\s*$")
        matches=list(headings.finditer(full))
        for i,m in enumerate(matches):
            title=m.group(1).strip().lower(); body=full[m.end():matches[i+1].start() if i+1<len(matches) else len(full)].strip()
            for key, aliases in self.SECTION_NAMES.items():
                if any(a in title for a in aliases) and key not in sections: sections[key]=body
        if "abstract" not in sections: sections["abstract"] = full[:4000]
        return sections

