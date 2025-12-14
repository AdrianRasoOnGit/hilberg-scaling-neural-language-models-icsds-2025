
import re
import bz2
from pathlib import Path
from lxml import etree


_REDIRECT_RE = re.compile(
    r"^\s*(#REDIRECT|#REDIRECT:|#\s*REDIRECT)\b",
    flags=re.IGNORECASE | re.DOTALL,
)


def iterate_xml(dump_path):
    dump_path = Path(dump_path)
    print(f"[extract.py] Streaming XML from {dump_path}...")

    with bz2.open(dump_path, "rb") as f:
        context = etree.iterparse(
            f,
            events=("end",),
            tag="{*}page",
            remove_comments=True,
            recover=True,
            huge_tree=True,
        )

        for _, elem in context:
            page_id = elem.findtext("./{*}id[1]")
            title = elem.findtext("./{*}title")
            ns = elem.findtext("./{*}ns")

            if ns != "0":
                _clear_element(elem)
                continue

            text_elem = elem.find("./{*}revision/{*}text")
            if text_elem is not None:
                text = "".join(t for t in text_elem.itertext() if t)
            else:
                text = ""

            if _looks_like_redirect(text):
                _clear_element(elem)
                continue

            try:
                page_id_int = int(page_id)
            except (TypeError, ValueError):
                page_id_int = None

            if page_id_int is not None and text.strip():
                yield {
                    "id": page_id_int,
                    "title": title or "",
                    "text": text,
                }

            _clear_element(elem)


def _looks_like_redirect(text):
    if not text:
        return False
    cleaned = text.lstrip()
    return bool(_REDIRECT_RE.match(cleaned))


def _clear_element(elem):
    parent = elem.getparent()
    if parent is None:
        elem.clear()
        return

    prev = elem.getprevious()
    while prev is not None:
        try:
            parent.remove(prev)
        except Exception:
            break
        prev = elem.getprevious()

    try:
        parent.remove(elem)
    except Exception:
        pass

    elem.clear()
