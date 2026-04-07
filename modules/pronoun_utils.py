"""
Shared pronoun inference utility.

Uses spaCy NER and morphological features to determine the correct pronoun
for a given noun token.  This replaces the per-module hardcoded heuristics
that only worked for a handful of pre-listed names.
"""

from utils import nlp  # noqa: F401 – keeps consistent import style

# ── optional gender-guesser ──────────────────────────────────────────
try:
    import gender_guesser.detector as _gender_detector
    _GENDER_DETECTOR = _gender_detector.Detector(case_sensitive=False)
except Exception:
    _GENDER_DETECTOR = None

# ── small curated set of common human-role nouns ─────────────────────
HUMAN_NOUN_HINTS = {
    "doctor", "nurse", "teacher", "student", "boy", "girl", "man", "woman",
    "father", "mother", "brother", "sister", "person", "people", "child",
    "children", "scientist", "researcher", "engineer", "professor", "leader",
    "worker", "driver", "player", "actor", "actress", "author", "writer",
    "president", "king", "queen", "prince", "princess", "soldier", "officer",
    "lawyer", "judge", "artist", "musician", "singer", "dancer", "chef",
    "pilot", "firefighter", "detective", "spy", "monk", "nun", "priest",
    "coach", "athlete", "surgeon", "therapist", "architect", "designer",
    "photographer", "journalist", "reporter", "editor", "politician",
    "diplomat", "ambassador", "senator", "governor", "mayor", "citizen",
    "neighbor", "friend", "colleague", "partner", "spouse", "husband", "wife",
    "uncle", "aunt", "cousin", "grandfather", "grandmother", "nephew", "niece",
}

# Gendered role nouns where the word itself signals gender unambiguously.
MALE_ROLE_NOUNS = {
    "boy", "man", "father", "brother", "husband", "king", "prince",
    "gentleman", "uncle", "grandfather", "nephew", "monk", "priest",
    "actor",
}
FEMALE_ROLE_NOUNS = {
    "girl", "woman", "mother", "sister", "wife", "queen", "princess",
    "lady", "aunt", "grandmother", "niece", "nun", "actress",
}

# Entity types that are clearly non-human → always "It"
NON_HUMAN_ENTITY_TYPES = {
    "GPE", "LOC", "FAC", "ORG", "EVENT", "PRODUCT", "WORK_OF_ART",
    "LAW", "LANGUAGE", "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY",
    "ORDINAL", "CARDINAL", "NORP",
}


def infer_pronoun(noun_token, context_span=None):
    """
    Given a spaCy Token that is the head noun of a noun phrase, return the
    most appropriate subject pronoun string: "He", "She", "They", or "It".

    The logic is:
      0. Context hints → if context_span contains strong "she/he" pronouns, use them.
      1. Plural  →  "They"
      2. Non-human NER type (GPE/LOC/ORG/…) or common non-human noun  →  "It"
      3. PERSON entity  →  try gender_guesser on the first PROPN; fall back to "They"
      4. Human-role noun  →  check gendered role sets; fall back to "They"
      5. Everything else  →  "It"
    """

    # ── 0. Context hints (if explicit pronoun used in the same sentence) ──
    if context_span:
        lower_tokens = {t.text.lower() for t in context_span}
        if any(t in {"her", "hers", "she", "herself"} for t in lower_tokens):
            return "She"
        if any(t in {"his", "him", "he", "himself"} for t in lower_tokens):
            return "He"

    # ── 1. Plurality ─────────────────────────────────────────────────
    if noun_token.tag_ in {"NNS", "NNPS"} or noun_token.morph.get("Number") == ["Plur"]:
        return "They"

    # ── 2. Entity-type based (most reliable for non-human) ───────────
    ent_type = noun_token.ent_type_

    # If the token itself has no ent_type, check its immediate compound noun modifiers
    # (avoid searching the whole subtree to prevent picking up entities in relative clauses).
    if not ent_type:
        for t in noun_token.children:
            if t.dep_ == "compound" and t.ent_type_:
                ent_type = t.ent_type_
                break

    if ent_type in NON_HUMAN_ENTITY_TYPES:
        return "It"

    # ── 3. PERSON entity → attempt gender detection ──────────────────
    if ent_type == "PERSON":
        return _gender_for_person(noun_token)

    # ── 4. Human-role nouns (no NER, but clearly a person) ───────────
    lemma = noun_token.lemma_.lower()
    if lemma in HUMAN_NOUN_HINTS:
        if lemma in MALE_ROLE_NOUNS:
            return "He"
        if lemma in FEMALE_ROLE_NOUNS:
            return "She"
        # Default for gender-ambiguous human roles (doctor, teacher, etc.)
        return "He"

    # ── 5. If there's a PROPN in subtree, it might be an un-tagged person ─
    propn_tokens = [t for t in noun_token.subtree if t.pos_ == "PROPN"]
    if propn_tokens:
        if noun_token.pos_ == "PROPN":
            # Try gender_guesser on the token itself first (handles surnames like "Shakespeare")
            gender = _guess_gender_from_name(noun_token.text)
            if gender:
                return gender
            # Try the first PROPN in the subtree
            gender = _guess_gender_from_name(propn_tokens[0].text)
            if gender:
                return gender
            # Unknown proper noun with no NER label — safer to use "It".
            return "It"

    # ── 6. Default: non-human ────────────────────────────────────────
    return "It"


def _gender_for_person(noun_token):
    """Try to determine He/She for a PERSON entity; fall back to They."""
    # Collect proper-noun tokens in the subject phrase
    propn_tokens = [t for t in noun_token.subtree if t.pos_ == "PROPN"]

    if propn_tokens:
        first_name = propn_tokens[0].text
        gender = _guess_gender_from_name(first_name)
        if gender:
            return gender

    # Check if the head noun itself is a gendered role word
    lemma = noun_token.lemma_.lower()
    if lemma in MALE_ROLE_NOUNS:
        return "He"
    if lemma in FEMALE_ROLE_NOUNS:
        return "She"

    # Gender-neutral fallback for people
    return "He"


def _guess_gender_from_name(name):
    """
    Use the gender_guesser library (if available) to infer gender from a
    first name.  Returns "He", "She", or None.
    """
    if _GENDER_DETECTOR is None:
        return None

    guess = _GENDER_DETECTOR.get_gender(name)
    if guess in {"female", "mostly_female"}:
        return "She"
    if guess in {"male", "mostly_male"}:
        return "He"
    return None
