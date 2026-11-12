"""Export the three figures of analysis/wal_design.html as standalone SVG files (light theme), for the thesis.

    python analysis/export_wal_figures.py   # writes analysis/figures/wal_fig1_architecture.svg, ..._checks.svg, ..._claims.svg

The page styles its inline SVGs with CSS classes and theme tokens; each file here gets the light values and the class
rules embedded. Convert to PDF elsewhere if needed (e.g. rsvg-convert -f pdf, or Inkscape).
"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
page = (HERE / "wal_design.html").read_text()
css = page[page.index("<style>") + 7:page.index("</style>")]
tokens = dict(re.findall(r"(--[\w-]+):\s*([^;]+);", css[css.index(":root {"):css.index("}", css.index(":root {"))]))
rules = css[css.index("/* SVG text and shapes */"):css.index("/* level table */")]


def resolve(text):
    for _ in range(3):  # tokens may refer to tokens
        text = re.sub(r"var\((--[\w-]+)\)", lambda m: tokens[m.group(1)].strip(), text)
    return text


names = ["architecture", "checks", "claims"]
out = HERE / "figures"
out.mkdir(exist_ok=True)
for i, svg in enumerate(re.findall(r"<svg .*?</svg>", page, re.S), 1):
    style = f"<style>svg {{ background: {resolve('var(--surface)')}; }}\n{resolve(rules)}</style>"
    svg = resolve(svg).replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ', 1)
    svg = re.sub(r"(<svg [^>]*>)", lambda m: m.group(1) + "\n" + style, svg, count=1)
    path = out / f"wal_fig{i}_{names[i - 1]}.svg"
    path.write_text('<?xml version="1.0" encoding="UTF-8"?>\n' + svg + "\n")
    print("written", path.relative_to(HERE.parent))
