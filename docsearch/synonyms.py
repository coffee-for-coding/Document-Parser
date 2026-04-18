"""Seed DE<->EN synonym dictionary. Extend freely; also mirrored to
analysis/de_en_synonyms.txt for Elasticsearch."""

SYNONYMS = {
    # time / day
    "morgen": ["tomorrow", "morning"],
    "tomorrow": ["morgen"],
    "morning": ["morgen", "früh"],
    "heute": ["today"],
    "today": ["heute"],
    "gestern": ["yesterday"],
    "yesterday": ["gestern"],
    "abend": ["evening"],
    "evening": ["abend"],
    "nacht": ["night"],
    "night": ["nacht"],

    # place of birth
    "heimatstadt": ["hometown", "birthplace", "home town"],
    "hometown": ["heimatstadt", "geburtsort", "heimat"],
    "geburtsort": ["birthplace", "hometown", "place of birth"],
    "birthplace": ["geburtsort", "heimatstadt"],
    "heimat": ["home", "hometown"],
    "geboren": ["born"],
    "born": ["geboren"],
    "stadt": ["city", "town"],
    "city": ["stadt"],
    "town": ["stadt"],

    # dwelling
    "haus": ["house", "home"],
    "house": ["haus"],
    "home": ["zuhause", "heim", "haus"],
    "wohnung": ["apartment", "flat"],
    "apartment": ["wohnung"],

    # common
    "auto": ["car"],
    "car": ["auto"],
    "buch": ["book"],
    "book": ["buch"],
    "arbeit": ["work", "job"],
    "work": ["arbeit"],
    "job": ["arbeit"],
    "schule": ["school"],
    "school": ["schule"],
    "freund": ["friend"],
    "friend": ["freund"],
    "familie": ["family"],
    "family": ["familie"],
}


def expand_term(term: str):
    term = term.lower()
    return SYNONYMS.get(term, [])


# Explicit language buckets so we can classify query terms reliably,
# without depending on langdetect for single-word inputs.
DE_WORDS = {
    "morgen", "heute", "gestern", "abend", "nacht", "heimatstadt",
    "geburtsort", "heimat", "geboren", "stadt", "haus", "wohnung",
    "auto", "buch", "arbeit", "schule", "freund", "familie", "zuhause",
    "heim", "früh", "morgens",
}
EN_WORDS = {
    "tomorrow", "morning", "today", "yesterday", "evening", "night",
    "hometown", "birthplace", "home", "born", "city", "town", "house",
    "apartment", "flat", "car", "book", "work", "job", "school",
    "friend", "family", "place", "of",
}


def classify(term: str) -> str:
    """Return 'de' or 'en' for a single word, falling back to character hints."""
    t = term.lower()
    if t in DE_WORDS:
        return "de"
    if t in EN_WORDS:
        return "en"
    import re
    if re.search(r"[äöüß]", t):
        return "de"
    return "en"
