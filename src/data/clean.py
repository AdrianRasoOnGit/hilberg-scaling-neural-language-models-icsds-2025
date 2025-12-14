
import re
import mwparserfromhell

def clean_wikitext(wikitext: str) -> str:

    if not wikitext:
        return ""

    try:
        wikicode = mwparserfromhell.parse(wikitext)
    except Exception:
        return wikitext

    for template in wikicode.filter_templates():
        wikicode.remove(template)

    for comment in wikicode.filter_comments():
        wikicode.remove(comment)

    for ext_link in wikicode.filter_external_links():
        wikicode.remove(ext_link)

    for tag in wikicode.filter_tags():
        wikicode.remove(tag)

    text = wikicode.strip_code(normalize=True, collapse=True)

    # Additional clean layer with regex

    text = re.sub(r"\[\[([^|\]]+)\|([^]]+)\]\]", r"\2", text)   
    text = re.sub(r"\[\[([^]]+)\]\]", r"\1", text)            

    text = re.sub(r"=+\s*(.*?)\s*=+", r"\1\n", text)

    text = re.sub(r"('{2,5})", "", text)

    text = re.sub(r"(?i)(?:category|file|image):\s*[^ \n]+", "", text)

    text = re.sub(r"&[a-zA-Z]+;", " ", text)

    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = text.strip()

    return text
