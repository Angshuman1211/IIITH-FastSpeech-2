"""Technical_Report.md -> Technical_Report.pdf, via pandoc and headless Chrome.

No LaTeX on this machine, so pandoc writes HTML and Chrome prints it.

    python scripts/make_report_pdf.py
"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT / "project" / "Report" / "Technical_Report.md"
HTML = ROOT / "project" / "Report" / "_report.html"
PDF = ROOT / "project" / "Report" / "Technical_Report.pdf"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@page { size: A4; margin: 20mm 18mm; }
body { font-family: Georgia, "Times New Roman", serif; font-size: 10.5pt;
       line-height: 1.5; color: #111; max-width: none; }
h1 { font-size: 20pt; margin: 0 0 4pt; }
h2 { font-size: 14pt; margin: 20pt 0 6pt; border-bottom: 1px solid #ccc;
     padding-bottom: 3pt; page-break-after: avoid; }
h3 { font-size: 11.5pt; margin: 14pt 0 4pt; page-break-after: avoid; }
p, li { orphans: 3; widows: 3; }
code, pre { font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 9pt; }
pre { background: #f6f6f6; padding: 8pt; border-radius: 3px;
      white-space: pre-wrap; page-break-inside: avoid; }
code { background: #f1f1f1; padding: 0 2px; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; margin: 8pt 0; font-size: 9.5pt;
        page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 3pt 7pt; text-align: left; }
th { background: #f0f0f0; }
blockquote { margin: 8pt 0; padding: 6pt 10pt; background: #fafafa;
             border-left: 3px solid #999; }
hr { border: none; border-top: 1px solid #ddd; margin: 16pt 0; }
"""


def main():
    if not MD.exists():
        sys.exit(f"{MD} not found")
    if not shutil.which("pandoc"):
        sys.exit("pandoc not installed: brew install pandoc")
    if not Path(CHROME).exists():
        sys.exit(f"Chrome not found at {CHROME}")

    # fragment, not -s: pandoc's own title block would duplicate the H1
    body = subprocess.run(
        ["pandoc", str(MD), "-f", "gfm", "-t", "html5"],
        check=True, capture_output=True, text=True,
    ).stdout
    HTML.write_text(
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>FastSpeech 2 for English and Telugu</title>"
        f"<style>{CSS}</style></head><body>{body}</body></html>",
        encoding="utf-8",
    )
    subprocess.run(
        [CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
         f"--print-to-pdf={PDF}", HTML.as_uri()],
        check=True, capture_output=True,
    )
    if not PDF.exists():
        sys.exit("Chrome did not write the PDF")

    HTML.unlink()
    print(f"{PDF}  ({PDF.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
