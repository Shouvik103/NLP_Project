from utils import nlp
from modules.pronoun_utils import infer_pronoun
import re


def simplify_appositives(text):
    """
    Extracts appositives into separate sentences.
    Example: "Elon Musk, the CEO of Tesla, bought Twitter." 
    -> "Elon Musk bought Twitter."
    -> "He was the CEO of Tesla."
    """
    doc = nlp(text)
    
    # We look for 'appos' dependency.
    appos_token = None
    for token in doc:
        if token.dep_ == 'appos':
            appos_token = token
            break
            
    if not appos_token:
        return [text]
        
    # Found an appositive.
    head_token = appos_token.head
    
    # Span of Appositive
    appos_span_start = min([t.i for t in appos_token.subtree])
    appos_span_end = max([t.i for t in appos_token.subtree])
    
    # We also need to remove enclosing commas usually.
    remove_start = appos_span_start
    remove_end = appos_span_end
    
    if remove_start > 0 and doc[remove_start - 1].text == ',':
        remove_start -= 1
    
    if remove_end < len(doc) - 1 and doc[remove_end + 1].text == ',':
        remove_end += 1
        
    # 1. Main Sentence (Removal)
    main_sent_tokens = []
    for i, token in enumerate(doc):
        if i >= remove_start and i <= remove_end:
            continue
            
        t_text = token.text_with_ws
        if i == remove_start - 1:
             if not token.whitespace_:
                 t_text = token.text + " "
                 
        main_sent_tokens.append(t_text)
    
    main_sent = "".join(main_sent_tokens).strip()
    main_sent = re.sub(r'\s+', ' ', main_sent).strip()
    main_sent = re.sub(r'\s+([,.])', r'\1', main_sent)
    
    # 2. Extracted Sentence (Construction)
    # "[Pronoun] is/was [Appositive]."
    
    # Get Head Text (full noun phrase minus the appositive subtree)
    head_subtree_idxs = [t.i for t in head_token.subtree]
    appos_subtree_idxs = [t.i for t in appos_token.subtree]
    
    exclude_idxs = set(appos_subtree_idxs)
    if remove_start < appos_span_start: exclude_idxs.add(remove_start)
    if remove_end > appos_span_end: exclude_idxs.add(remove_end)
    
    head_phrase_tokens = []
    for token in doc:
        if token.i in head_subtree_idxs and token.i not in exclude_idxs:
             head_phrase_tokens.append(token.text_with_ws)
             
    head_text = "".join(head_phrase_tokens).strip()
    head_text = re.sub(r'\s+', ' ', head_text).strip()
    
    # Get Appositive Text
    appos_text = "".join([t.text_with_ws for t in appos_token.subtree]).strip()
    
    # Determine verb tense
    # For people: match main verb tense (past actions → "was")
    # For things/places: always use present tense "is" (definitional properties)
    main_root = None
    for token in doc:
        if token.dep_ == "ROOT":
            main_root = token
            break
    
    # Infer pronoun to determine if subject is a person or thing
    pronoun = infer_pronoun(head_token, context_span=doc)
    is_person = pronoun in ("He", "She", "They")
    
    # Determine tense
    verb = "is"
    if is_person and main_root:
        # Check if main verb is past tense
        main_is_past = main_root.tag_ in ['VBD', 'VBN']
        if not main_is_past:
            # Check auxiliaries
            for child in main_root.children:
                if child.dep_ in ["aux", "auxpass"] and child.tag_ in ['VBD']:
                    main_is_past = True
                    break
        if main_is_past:
            verb = "was"

    # Plurality agreement
    if head_token.tag_ in ['NNS', 'NNPS']:
        verb = "are" if verb == "is" else "were"

    # Use pronoun as subject
    subject = pronoun
        
    extracted_sent = f"{subject} {verb} {appos_text}."
    
    # Capitalize
    if main_sent:
        main_sent = main_sent[0].upper() + main_sent[1:]
    extracted_sent = extracted_sent[0].upper() + extracted_sent[1:]
    
    return [main_sent, extracted_sent]
