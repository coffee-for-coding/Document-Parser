from pathlib import Path
from typing import Iterator, Tuple


def parse_pdf(path: Path) -> Iterator[Tuple[int, str]]:
    import fitz  # PyMuPDF
    with fitz.open(str(path)) as doc:
        for i, page in enumerate(doc, start=1):
            yield i, page.get_text("text")


def parse_docx(path: Path) -> Iterator[Tuple[int, str]]:
    """DOCX has no real pages; split on explicit rendered page breaks,
    falling back to a single logical page if none exist."""
    import docx
    d = docx.Document(str(path))
    page, buf = 1, []
    for para in d.paragraphs:
        buf.append(para.text)
        has_break = False
        for run in para.runs:
            try:
                if "lastRenderedPageBreak" in run.element.xml:
                    has_break = True
                    break
            except Exception:
                continue
        if has_break:
            yield page, "\n".join(buf)
            buf.clear()
            page += 1
    if buf:
        yield page, "\n".join(buf)


def parse_txt(path: Path, chars_per_page: int = 3000) -> Iterator[Tuple[int, str]]:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    page = 1
    for i in range(0, len(text), chars_per_page):
        yield page, text[i:i + chars_per_page]
        page += 1


def parse(path: Path) -> Iterator[Tuple[int, str]]:
    path = Path(path)
    suf = path.suffix.lower()
    if suf == ".pdf":
        yield from parse_pdf(path)
    elif suf == ".docx":
        yield from parse_docx(path)
    elif suf in (".txt", ".md"):
        yield from parse_txt(path)
    else:
        raise ValueError(f"Unsupported file type: {suf}")
