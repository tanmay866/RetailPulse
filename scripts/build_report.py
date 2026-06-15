"""Render REPORT.md to a polished PDF (REPORT.pdf).

Markdown -> HTML (python-markdown) -> styled HTML -> PDF (Playwright + Chrome).
Run from the repo root:  python scripts/build_report.py
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

import markdown
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "REPORT.md"
OUT = ROOT / "REPORT.pdf"
MERMAID_LIB = ROOT / "scripts" / "lib" / "mermaid.min.js"

# Monochrome theme so diagrams match the black-and-white report.
MERMAID_INIT = """
mermaid.initialize({
  startOnLoad: true,
  theme: 'base',
  themeVariables: {
    fontFamily: 'Segoe UI, Helvetica, Arial, sans-serif',
    fontSize: '15px',
    primaryColor: '#ffffff',
    primaryBorderColor: '#1a1a1a',
    primaryTextColor: '#1a1a1a',
    lineColor: '#1a1a1a',
    secondaryColor: '#f4f4f4',
    secondaryBorderColor: '#1a1a1a',
    tertiaryColor: '#ffffff',
    tertiaryBorderColor: '#1a1a1a',
    clusterBkg: '#ffffff',
    edgeLabelBackground: '#ffffff'
  },
  flowchart: { curve: 'basis', nodeSpacing: 45, rankSpacing: 50, htmlLabels: true, useMaxWidth: true }
});
"""

CSS = """
:root { --ink:#1a1a1a; --muted:#5b6b7b; --line:#e2e8f0; --accent:#111111; }

@page { size: A4; margin: 16mm 15mm; }

* { box-sizing: border-box; }
body {
  font-family: "Segoe UI", Helvetica, Arial, sans-serif;
  color: var(--ink); font-size: 10.5pt; line-height: 1.7; margin: 0;
}

/* ---- Cover (minimal editorial) ---- */
@page coverpage { margin: 0; }
.cover {
  page: coverpage;
  background: #ffffff; color: #14181f;
  width: 210mm; height: 297mm; padding: 28mm 26mm 24mm;
  display: flex; flex-direction: column; justify-content: space-between;
  page-break-after: always;
}

.cv-head {
  display: flex; justify-content: space-between; align-items: baseline;
  padding-bottom: 12px; border-bottom: 2px solid #14181f;
}
.cv-head .cv-brand { font-weight: 700; letter-spacing: 3px; font-size: 11pt; }
.cv-head .cv-meta { font-size: 10pt; color: #6b7280; letter-spacing: .5px; }

.cv-body { padding-top: 6mm; }
.cv-kicker { font-size: 10.5pt; letter-spacing: 5px; text-transform: uppercase; color: #9aa3af; margin: 0 0 16px; }
.cv-title { font-size: 74pt; font-weight: 800; letter-spacing: -2.5px; line-height: .95; margin: 0; color: #14181f; border: 0; }
.cv-rule { width: 84px; height: 4px; background: #111111; margin: 24px 0; }
.cv-tagline { font-family: Georgia, "Times New Roman", serif; font-size: 19pt; color: #2b3744; margin: 0 0 18px; line-height: 1.4; max-width: 150mm; }
.cv-lede { font-size: 11pt; color: #5b6b7b; line-height: 1.6; margin: 0; max-width: 150mm; }

.cv-foot { }
.cv-stats {
  list-style: none; display: flex; gap: 0; padding: 18px 0; margin: 0 0 18px;
  border-top: 1px solid #e2e8f0; border-bottom: 1px solid #e2e8f0;
}
.cv-stats li { flex: 1; }
.cv-stats b { display: block; font-size: 21pt; font-weight: 800; letter-spacing: -.5px; color: #14181f; line-height: 1; }
.cv-stats span { display: block; font-size: 8.5pt; letter-spacing: .8px; text-transform: uppercase; color: #8b97a4; margin-top: 7px; }

.cv-byline { display: flex; justify-content: space-between; align-items: flex-end; }
.cv-author b { font-size: 13pt; display: block; margin-bottom: 3px; }
.cv-author span { font-size: 10pt; color: #6b7280; }
.cv-demo { font-size: 10pt; color: #111111; font-weight: 600; }

/* ---- Headings ---- */
h1, h2, h3 { color: var(--ink); line-height: 1.25; }
h2 {
  font-size: 18pt; margin: 0 0 22px; padding-bottom: 11px;
  border-bottom: 3px solid var(--accent); page-break-before: always;
}
h2:first-of-type { page-break-before: avoid; }
h3 { font-size: 12.5pt; margin: 30px 0 12px; color: var(--accent); }

/* ---- Tables ---- */
table { width: 100%; border-collapse: collapse; margin: 16px 0 24px; font-size: 9.5pt; page-break-inside: avoid; }
th, td { border: 1px solid var(--line); padding: 9px 13px; text-align: left; vertical-align: top; line-height: 1.5; }
th { background: #1a1a1a; color: #fff; font-weight: 600; }
tr:nth-child(even) td { background: #f6f9fc; }

/* ---- Code / ASCII art ---- */
pre {
  background: #1a1a1a; color: #e6e6e6; padding: 14px 16px; border-radius: 8px;
  font-size: 8pt; line-height: 1.35; overflow: hidden; page-break-inside: avoid;
  font-family: "Cascadia Mono", Consolas, monospace;
}
code { font-family: "Cascadia Mono", Consolas, monospace; }

/* ---- Images ---- */
img {
  max-width: 100%; display: block; margin: 8px auto 4px;
  border: 1px solid var(--line); border-radius: 6px; page-break-inside: avoid;
}

/* Keep each caption + screenshot together on one page */
figure.figure { margin: 16px 0; page-break-inside: avoid; break-inside: avoid; }
figure.figure figcaption { margin: 0 0 6px; }
figure.figure img { margin: 0 auto; }

/* Mermaid diagram */
.mermaid { text-align: center; margin: 18px 0 24px; page-break-inside: avoid; }
.mermaid svg { max-width: 100%; height: auto; }

p { margin: 11px 0; }
ul { margin: 12px 0 18px; padding-left: 24px; }
li { margin: 8px 0; padding-left: 4px; line-height: 1.65; }
li::marker { color: var(--accent); }
hr { border: 0; border-top: 1px solid var(--line); margin: 28px 0; }
a { color: var(--accent); text-decoration: none; }
em { color: var(--muted); }
"""


def fix_lists(md: str) -> str:
    """Markdown needs a blank line before a list. Insert one when a list item
    directly follows a paragraph, so bullets don't render as literal '- '.
    Fenced code blocks are left untouched."""
    import re

    marker = re.compile(r"^(\s*)([-*+] |\d+\. )")
    out: list[str] = []
    in_fence = False
    for line in md.split("\n"):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        if not in_fence and marker.match(line) and out:
            prev = out[-1]
            if prev.strip() and not marker.match(prev):
                out.append("")
        out.append(line)
    return "\n".join(out)


def render_mermaid_blocks(html: str) -> str:
    """Turn ```mermaid fenced blocks into <div class="mermaid"> so Mermaid.js
    renders them. Markdown escapes the code, so unescape it back first."""
    import html as ihtml
    import re

    return re.sub(
        r'<pre><code class="language-mermaid">(.*?)</code></pre>',
        lambda m: f'<div class="mermaid">{ihtml.unescape(m.group(1))}</div>',
        html,
        flags=re.DOTALL,
    )


def group_figures(html: str) -> str:
    """Wrap each caption paragraph + the image right after it in a <figure>
    so a page break can't separate a screenshot from its title."""
    import re

    # Markdown merges a caption line and the image line that follows it into a
    # single <p>: "<p><strong>Title</strong> — desc <img ...></p>".
    pattern = re.compile(
        r"<p>(?P<cap>(?:(?!</p>)(?!<img).)*?)\s*(?P<img><img\b[^>]*>)\s*</p>",
        re.DOTALL,
    )
    return pattern.sub(
        lambda m: f'<figure class="figure"><figcaption>{m.group("cap")}</figcaption>{m.group("img")}</figure>',
        html,
    )


def inline_images(html: str) -> str:
    """Embed local images as base64 data URIs so the PDF is self-contained."""
    import re

    def repl(match: re.Match) -> str:
        src = match.group(1)
        if src.startswith(("http://", "https://", "data:")):
            return match.group(0)
        path = (ROOT / src).resolve()
        if not path.exists():
            print(f"  ! missing image: {src}")
            return match.group(0)
        mime = mimetypes.guess_type(str(path))[0] or "image/png"
        data = base64.b64encode(path.read_bytes()).decode()
        return f'src="data:{mime};base64,{data}"'

    return re.sub(r'src="([^"]+)"', repl, html)


def main() -> None:
    text = fix_lists(SRC.read_text(encoding="utf-8"))
    body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "attr_list", "md_in_html"],
    )
    body = render_mermaid_blocks(body)
    body = group_figures(body)
    body = inline_images(body)

    has_mermaid = "mermaid" in body
    scripts = ""
    if has_mermaid:
        mermaid_js = MERMAID_LIB.read_text(encoding="utf-8")
        scripts = f"<script>{mermaid_js}</script><script>{MERMAID_INIT}</script>"

    html = (
        f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style>"
        f"{scripts}</head><body>{body}</body></html>"
    )

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        page = browser.new_page()
        page.set_content(html, wait_until="networkidle")
        if has_mermaid:
            page.wait_for_selector(".mermaid svg", timeout=20000)
            page.wait_for_timeout(400)
        page.pdf(
            path=str(OUT),
            print_background=True,
            prefer_css_page_size=True,
        )
        browser.close()

    print(f"Wrote {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
