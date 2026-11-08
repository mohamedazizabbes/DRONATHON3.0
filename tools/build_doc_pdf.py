"""Build docs/SKYFORGE_Technical_Documentation.pdf from docs/TECHNICAL_DOCUMENTATION.md.

    pip install markdown
    python tools/build_doc_pdf.py [--browser PATH]

Markdown -> styled print HTML (A4, cover with the banner, MathJax formulas, diagrams)
-> PDF with a headless Chromium browser (Edge or Chrome).
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
SRC = DOCS / "TECHNICAL_DOCUMENTATION.md"
OUT = DOCS / "SKYFORGE_Technical_Documentation.pdf"

BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser", "/usr/bin/microsoft-edge",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]

CSS = """
@page { size: A4; margin: 16mm 15mm 18mm 15mm; }
:root { --ink:#0f1c24; --muted:#4a5b63; --teal:#0b6f7a; --orange:#e2632a; --line:#d7e1e4; --soft:#f3f7f8; }
* { box-sizing: border-box; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; color: var(--ink); font-size: 10.2pt; line-height: 1.5; margin: 0; }
h1 { font-size: 22pt; margin: 6mm 0 2mm; letter-spacing: .2px; }
h2 { font-size: 15.5pt; color: var(--teal); border-bottom: 2px solid var(--teal); padding-bottom: 2mm;
     margin: 0 0 4mm; break-before: page; }
h3 { font-size: 12pt; margin: 6mm 0 2mm; color: #12323b; }
h4 { font-size: 10.5pt; margin: 4mm 0 1.5mm; }
p, li { orphans: 3; widows: 3; }
a { color: var(--teal); text-decoration: none; }
code { font-family: "Cascadia Mono", Consolas, "IBM Plex Mono", monospace; font-size: 8.8pt; background: var(--soft);
       border: 1px solid var(--line); border-radius: 3px; padding: 0 3px; }
pre { background: #0f1c24; color: #e9f1f2; border-radius: 6px; padding: 3mm 4mm; overflow: hidden; white-space: pre-wrap;
      word-break: break-word; break-inside: avoid; font-size: 8.4pt; line-height: 1.45; }
pre code { background: none; border: 0; color: inherit; padding: 0; font-size: inherit; }
table { border-collapse: collapse; width: 100%; margin: 2.5mm 0 4mm; font-size: 8.9pt; break-inside: auto; }
tr { break-inside: avoid; }
th { background: var(--teal); color: #fff; text-align: left; font-weight: 600; padding: 1.6mm 2.2mm; }
td { border-bottom: 1px solid var(--line); padding: 1.4mm 2.2mm; vertical-align: top; }
tbody tr:nth-child(even) td { background: var(--soft); }
blockquote { border-left: 3px solid var(--orange); margin: 3mm 0; padding: 1mm 4mm; color: var(--muted); background: #fff7f2; }
img { max-width: 100%; break-inside: avoid; }
p > img, p img { display: inline-block; }
hr { border: 0; border-top: 1px solid var(--line); margin: 5mm 0; }
.math { margin: 3mm 0; break-inside: avoid; overflow-x: hidden; }
/* cover */
.cover { height: 260mm; display: flex; flex-direction: column; break-after: page; }
.cover .banner { width: 100%; border-radius: 6px; }
.cover h1 { font-size: 30pt; margin-top: 16mm; }
.cover .sub { font-size: 14pt; color: var(--muted); margin-top: 1mm; }
.cover .meta { margin-top: auto; border-top: 2px solid var(--teal); padding-top: 4mm; font-size: 10pt; color: var(--muted); }
.cover .meta b { color: var(--ink); }
/* first h2 (Contents) must not add a second blank page after the cover */
.first-h2 { break-before: avoid; }
"""

MATHJAX = """
<script>
window.MathJax = { tex: { displayMath: [['\\\\[','\\\\]']], inlineMath: [['\\\\(','\\\\)']] },
                   svg: { fontCache: 'global' }, startup: { typeset: true } };
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg.js"></script>
"""


def protect_math(md: str) -> tuple[str, list[str]]:
    """Keep $$...$$ blocks away from the Markdown parser (underscores, backslashes)."""
    blocks: list[str] = []

    def repl(m):
        blocks.append(m.group(1).strip())
        return f"\n\nMATHBLOCK{len(blocks) - 1}END\n\n"

    return re.sub(r"\$\$(.+?)\$\$", repl, md, flags=re.S), blocks


LIST_RE = re.compile(r"^(\s*)([-*+]|\d+\.)\s")


def normalize_lists(md: str) -> str:
    """GitHub-flavoured lists -> Python-Markdown: a blank line before a list that follows
    a paragraph, and 2-space nested indentation doubled to 4 spaces."""
    out, in_code, prev = [], False, ""
    for line in md.splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
        m = None if in_code else LIST_RE.match(line)
        if m:
            indent = len(m.group(1))
            if indent:
                line = " " * (indent * 2) + line.lstrip()
            prev_is_text = prev.strip() and not LIST_RE.match(prev) and not prev.lstrip().startswith(("|", "<", "#"))
            if indent == 0 and prev_is_text and not prev.startswith(" "):
                out.append("")
        elif not in_code and line.startswith("  ") and out and LIST_RE.match(out[-1] if out[-1] else ""):
            line = "    " + line.lstrip()  # continuation line of a list item
        out.append(line)
        prev = line
    return "\n".join(out)


def build_html() -> str:
    md = normalize_lists(SRC.read_text(encoding="utf-8"))
    # drop the README-style banner paragraph and H1 at the top: the cover replaces them
    md = re.sub(r'^<p align="center"><img src="assets/banner.svg"[^\n]*\n+', "", md)
    md, maths = protect_math(md)
    body = markdown.markdown(md, extensions=["tables", "fenced_code", "attr_list", "md_in_html", "sane_lists", "toc"],
                             output_format="html5")
    for i, tex in enumerate(maths):
        body = body.replace(f"<p>MATHBLOCK{i}END</p>", f'<div class="math">\\[{tex}\\]</div>')
    # the first <h1> becomes part of the cover; the first <h2> (Contents) follows the cover directly
    body = re.sub(r"<h1[^>]*>.*?</h1>", "", body, count=1, flags=re.S)
    body = body.replace("<h2", '<h2 class="first-h2"', 1)
    cover = f"""
<section class="cover">
  <img class="banner" src="assets/banner.svg" alt="SKYFORGE">
  <h1>SKYFORGE<br>Technical Documentation</h1>
  <div class="sub">Autonomous coastal rescue drone · Dronathon 3.0 · Mission SKYFORGE</div>
  <div class="meta">
    <b>IEEE AESS/RAS ENSIT Student Branch Chapters</b> · Tunis, 07–08 November 2026<br>
    Stack: ROS 2 Jazzy · Gazebo Harmonic · PX4 SITL · LangGraph · YOLO11 · Python<br>
    Source: <code>docs/TECHNICAL_DOCUMENTATION.md</code> (repository <code>DRONATHON3.0</code>, branch <code>main</code>)
  </div>
</section>"""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>SKYFORGE Technical Documentation</title>
<base href="{DOCS.as_uri()}/">
<style>{CSS}</style>{MATHJAX}</head><body>{cover}{body}</body></html>"""


def find_browser(explicit: str | None) -> str:
    for c in ([explicit] if explicit else []) + BROWSERS:
        if c and Path(c).exists():
            return c
    for name in ("msedge", "google-chrome", "chromium", "chrome"):
        p = shutil.which(name)
        if p:
            return p
    sys.exit("no Chromium-based browser found; pass --browser")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--browser", default=None)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    html = build_html()
    with tempfile.TemporaryDirectory() as td:
        page = Path(td) / "doc.html"
        page.write_text(html, encoding="utf-8")
        cmd = [find_browser(a.browser), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
               "--virtual-time-budget=15000", "--run-all-compositor-stages-before-draw",
               f"--print-to-pdf={os.path.abspath(a.out)}", page.as_uri()]
        subprocess.run(cmd, check=True, capture_output=True, timeout=180)
    print(f"written {a.out} ({Path(a.out).stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
