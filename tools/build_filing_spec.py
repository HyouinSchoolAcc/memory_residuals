"""Build filing-ready provisional specification PDF (internal notes stripped)."""
from __future__ import annotations

import re
import subprocess
from datetime import date
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "PATENT_PROVISIONAL_DRAFT.md"
OUT_DIR = ROOT / "patent_filing"
OUT_MD = OUT_DIR / "PROVISIONAL_SPECIFICATION.md"
OUT_HTML = OUT_DIR / "PROVISIONAL_SPECIFICATION.html"
OUT_PDF = OUT_DIR / "PROVISIONAL_SPECIFICATION.pdf"

# Must match the inventor name on your Patent Center Web ADS exactly.
INVENTOR_NAME = "Yueze Liu"
FILING_DATE = date.today().strftime("%B %d, %Y")  # e.g. May 29, 2026

EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
if not EDGE.exists():
    EDGE = Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe")


def clean_source(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    skip_until_title = True
    for line in lines:
        if skip_until_title:
            if line.strip() == "## TITLE OF THE INVENTION":
                skip_until_title = False
                out.append(line)
            continue
        if line.strip() == "## END OF PROVISIONAL DRAFT":
            break
        out.append(line)
    body = "\n".join(out).strip() + "\n"
    # Drop internal figure path hints in BRIEF DESCRIPTION section only.
    body = re.sub(
        r"\[Inventor note: drawings to be included from[\s\S]*?properly cross-referenced from the specification\.\]",
        "[Drawings are provided as a separate PDF.]",
        body,
        count=1,
    )
    return body


def finalize_declaration(body: str) -> str:
    """Replace placeholder declaration with dated typed electronic signature."""
    signed = f"""## DECLARATION

The undersigned hereby declares that:

(a) the foregoing specification describes the invention as conceived
    by the named inventor(s) as of the filing date hereof;

(b) the inventor(s) have not previously caused the invention to be
    publicly disclosed in a manner that would bar patentability under
    35 U.S.C. § 102, except as may be subject to the inventor's
    one-year grace period under 35 U.S.C. § 102(b)(1);

(c) the inventor(s) acknowledge the duty of candor and good faith
    under 37 C.F.R. § 1.56 in dealings with the United States Patent
    and Trademark Office.

Inventor(s):

  /s/ {INVENTOR_NAME}
  Date: {FILING_DATE}
"""
    return re.sub(
        r"## DECLARATION[\s\S]*?\[add additional inventors if any\]\s*",
        signed,
        body,
        count=1,
    )


def add_filing_header(body: str) -> str:
    header = (
        f"**Provisional Patent Application — Specification**\n\n"
        f"**Inventor:** {INVENTOR_NAME}  \n"
        f"**Date:** {FILING_DATE}\n\n---\n\n"
    )
    if body.startswith("## TITLE OF THE INVENTION"):
        return header + body
    return body


def md_to_html(md_text: str) -> str:
    body = markdown.markdown(md_text, extensions=["extra", "sane_lists"])
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Provisional Patent Specification</title>
<style>
  @page {{ margin: 1in; }}
  body {{
    font-family: "Times New Roman", Times, serif;
    font-size: 12pt;
    line-height: 1.35;
    max-width: 6.5in;
    margin: 0 auto;
    color: #000;
  }}
  h1, h2, h3 {{ page-break-after: avoid; }}
  h2 {{ font-size: 14pt; margin-top: 1.2em; }}
  h3 {{ font-size: 12pt; margin-top: 1em; }}
  p, li {{ text-align: justify; }}
  blockquote {{
    margin: 0.5em 0;
    padding-left: 0.75em;
    border-left: 2px solid #ccc;
    font-style: italic;
  }}
  hr {{ border: none; border-top: 1px solid #999; margin: 1.5em 0; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def html_to_pdf(html_path: Path, pdf_path: Path) -> None:
    if not EDGE.exists():
        raise RuntimeError("Microsoft Edge not found for PDF export")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(EDGE),
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path}",
        html_path.resolve().as_uri(),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cleaned = clean_source(SRC.read_text(encoding="utf-8"))
    cleaned = finalize_declaration(cleaned)
    cleaned = add_filing_header(cleaned)
    OUT_MD.write_text(cleaned, encoding="utf-8")
    OUT_HTML.write_text(md_to_html(cleaned), encoding="utf-8")
    html_to_pdf(OUT_HTML, OUT_PDF)
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
