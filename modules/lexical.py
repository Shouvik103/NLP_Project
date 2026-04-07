from utils import nlp
import nltk
from nltk.corpus import wordnet

# Static Dictionary (Fast & Accurate for Known Words)
COMPLEX_TO_SIMPLE = {
    "utilize": "use", "utilizes": "uses", "utilized": "used", "utilizing": "using",
    "demonstrate": "show", "demonstrates": "shows", "demonstrated": "showed", "demonstrating": "showing",
    "attempt": "try", "attempts": "tries", "attempted": "tried", "attempting": "trying",
    "approximately": "about",
    "subsequently": "later",
    "consequently": "so",
    "furthermore": "also",
    "nevertheless": "however",
    "moreover": "also",
    "reside": "live", "resides": "lives", "resided": "lived",
    "commence": "start", "commences": "starts", "commenced": "started",
    "terminate": "end", "terminates": "ends", "terminated": "ended",
    "regarding": "about",
    "facilitate": "help", "facilitates": "helps", "facilitated": "helped",
    "manufacture": "make", "manufactures": "makes", "manufactured": "made",
    "assistance": "help",
    "require": "need", "requires": "needs", "required": "needed",
    "objective": "goal",
    "component": "part", "components": "parts",
    "location": "place",
    "verify": "check", "verifies": "checks", "verified": "checked",
    "initiate": "start", "initiates": "starts", "initiated": "started",
    "difficult": "hard",
}

def get_wordnet_synonym(word, pos_tag):
    """
    Finds a simpler synonym using WordNet.
    Criteria: Shorter length, commonness (heuristic).
    """
    # Map Spacy POS to WordNet POS
    wn_pos = None
    if pos_tag.startswith('V'): wn_pos = wordnet.VERB
    elif pos_tag.startswith('N'): wn_pos = wordnet.NOUN
    elif pos_tag.startswith('J'): wn_pos = wordnet.ADJ
    elif pos_tag.startswith('R'): wn_pos = wordnet.ADV
    
    if not wn_pos:
        return None
        
    try:
        synsets = wordnet.synsets(word, pos=wn_pos)
    except Exception:
        return None
        
    if not synsets:
        return None
        
    # Collect candidate synonyms
    candidates = set()
    for syn in synsets:
        for lemma in syn.lemmas():
            cand_word = lemma.name().replace('_', ' ')
            if cand_word.lower() != word.lower():
                candidates.add(cand_word)
                
    if not candidates:
        return None
        
    # Selection Heuristic:
    # 1. Shorter is better.
    # 2. Must not be the word itself.
    # 3. Avoid very rare words (hard to filter without frequency list, typically short ones are okay).
    
    best_cand = None
    min_len = len(word)
    
    for cand in candidates:
        if len(cand) < min_len:
            min_len = len(cand)
            best_cand = cand
            
    # Threshold: Must be significantly shorter or known simple
    # Example: "ameliorate" (11) -> "improve" (7) -> Good.
    return best_cand

def simplify_lexical(text):
    """
    Replaces complex words with simpler ones using a hybrid approach.
    1. Static Dictionary Lookup
    2. WordNet Fallback for long words (> 7 chars)
    """
    doc = nlp(text)
    new_tokens = []
    
    for token in doc:
        word_lower = token.text.lower()
        replacement = None
        
        # 1. Check Static Dictionary
        if word_lower in COMPLEX_TO_SIMPLE:
            replacement = COMPLEX_TO_SIMPLE[word_lower]
            
        # 2. Check Dynamic WordNet (Only for longer words to avoid noise)
        # Threshold increased to 7 to avoid "results" -> "events"
        # DISABLING for now as it causes "syllabus" -> "programs" and "preparing" -> "set"
        # elif len(word_lower) > 7 and token.pos_ in ['VERB', 'ADJ', 'NOUN']: 
        #     replacement = get_wordnet_synonym(word_lower, token.tag_)
            
        #     # Additional check: don't replace if replacement is longer or same
        #     # if replacement and len(replacement) >= len(word_lower):
        #     #      replacement = None
        pass
        
        # Apply replacement
        if replacement:
            # Match Case (Capitalization)
            if token.text[0].isupper():
                replacement = replacement.capitalize()
            
            # Basic Inflection Fallback for Dynamic replacements
            if word_lower not in COMPLEX_TO_SIMPLE:
                 # Try to preserve -ed, -s. 
                 # LIMITATION: Irregular verbs (buy -> buyed) are hard to fix without a library like pattern or lemminflect.
                 # We will be conservative: Only add S suffixes. Avoid ED for now to prevent "buyed".
                 # Context: User wants "everything" to work, but broken grammar is bad.
                 
                 if word_lower.endswith('s') and not replacement.endswith('s') and not word_lower.endswith('ss'):
                     replacement += 's'
                 elif word_lower.endswith('ed'):
                     # Skip replacement for past tense dynamic words to avoid "buyed"
                     # Unless we know the specific mapping (which we don't dynamically)
                     # So we ABORT replacement if we can't inflect safely.
                     replacement = None 
                     
            if replacement:
                new_tokens.append(replacement + token.whitespace_)
            else:
                new_tokens.append(token.text_with_ws)
        else:
            new_tokens.append(token.text_with_ws)
            
    return "".join(new_tokens).strip()
