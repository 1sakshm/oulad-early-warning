"""Assemble template.html from parts/ and the chart modules in template.v1.html. Usage: python site/assemble.py"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
old = (HERE / "template.v1.html").read_text(encoding="utf8").split("\n")
helpers = "\n".join(old[422:536])
mods = "\n".join(old[628:1112])
head = (HERE / "parts" / "head_body.html").read_text(encoding="utf8")
new = (HERE / "parts" / "new.js").read_text(encoding="utf8")
(HERE / "template.html").write_text(head + "\n<script>\n" + helpers + "\n\n" + mods + "\n\n" + new + "\n</script>\n</body>\n</html>\n", encoding="utf8")
print("wrote site/template.html")
