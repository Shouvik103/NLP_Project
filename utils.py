import spacy
import sys

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading language model...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def get_subject(verb):
    """
    Find the subject of a verb.
    """
    for child in verb.children:
        if child.dep_ in ("nsubj", "nsubjpass"):
            return child
    return None
