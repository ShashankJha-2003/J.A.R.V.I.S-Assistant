"""Build the standalone GitHub Pages static bundle (`docs/`) from `jarvis/web/static/`."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "jarvis" / "web" / "static"
DOCS = ROOT / "docs"


def build_pages() -> None:
    if DOCS.exists():
        shutil.rmtree(DOCS)
    DOCS.mkdir(parents=True, exist_ok=True)

    # Copy core assets (CSS, JS, 10 expressions + blinking, 72 turntable frames, 35 real-video mouth frames)
    for item in ("styles.css", "app.js"):
        shutil.copy2(SRC / item, DOCS / item)
    for folder in ("avatar", "turntable", "mouth_seq"):
        src_dir = SRC / folder
        if src_dir.exists():
            shutil.copytree(src_dir, DOCS / folder)

    # Rewrite index.html to use relative paths and activate standalone GitHub Pages mode
    html = (SRC / "index.html").read_text(encoding="utf-8")
    html = html.replace('href="/static/styles.css', 'href="./styles.css')
    html = html.replace('src="/static/app.js', 'src="./app.js')
    html = html.replace(
        "<script src=\"./app.js",
        "<script>window.JARVIS_STATIC_MODE = true;</script>\n  <script src=\"./app.js",
    )
    (DOCS / "index.html").write_text(html, encoding="utf-8")
    (DOCS / ".nojekyll").write_text("", encoding="utf-8")

    total_bytes = sum(p.stat().st_size for p in DOCS.rglob("*") if p.is_file())
    file_count = sum(1 for p in DOCS.rglob("*") if p.is_file())
    print(f"Built GitHub Pages bundle in {DOCS}: {file_count} files ({total_bytes / 1e6:.2f} MB)")


if __name__ == "__main__":
    build_pages()
