from utils import nlp
from modules.pronoun_utils import infer_pronoun


def _normalize_np(text):
    return " ".join((text or "").strip().lower().split())


def _subject_span(token):
    span_tokens = list(token.subtree)
    start = min(t.idx for t in span_tokens)
    end = max(t.idx + len(t.text) for t in span_tokens)
    return start, end


def _replace_span(text, start, end, replacement):
    return text[:start] + replacement + text[end:]


def _extract_subject_text(doc):
    for token in doc:
        if token.dep_ == 'nsubj' and token.pos_ in ['PROPN', 'NOUN']:
            span_min = min(t.i for t in token.subtree)
            span_max = max(t.i for t in token.subtree)
            return doc[span_min: span_max + 1].text
    return None

def resolve_coreference(sentences):
    """
    Simple heuristic coreference resolution for split sentences.
    Replaces initial pronouns in subsequent sentences with the subject of the previous sentence.
    
    Args:
        sentences (list of str): List of simplified sentences.
        
    Returns:
        list of str: Sentences with pronouns replaced.
    """
    final_sents = []
    last_subject = None
    
    for sent_text in sentences:
        doc = nlp(sent_text)
        
        # 1. Check if we need to replace the subject
        # We look at the first nominal subject
        target_token = None
        for token in doc:
            if token.dep_ == 'nsubj':
                target_token = token
                break
        
        replaced = False
        replaced_repeated_subject_with_pronoun = False
        if target_token and target_token.pos_ == 'PRON' and last_subject:
            pron = target_token.text.lower()
            # Only truly ambiguous demonstratives should be replaced.
            # Personal pronouns (he/she/they/it/we/you) are kept as-is
            # since they were correctly assigned by earlier pipeline stages.
            ambiguous_pronouns = {'this', 'that', 'these', 'those'}
            if pron not in ambiguous_pronouns:
                # Keep explicit pronouns for natural readability.
                final_sents.append(sent_text)
                replaced = True

            # Check for expletive "It" (e.g., "It is important...")
            # Simple heuristic: If "It" + "be" + "ADJ", unlikely to be coreferent to a noun.
            is_expletive = False
            if target_token.text.lower() == 'it':
                # Check head
                head = target_token.head
                if head.lemma_ == 'be':
                    # Check for adjective child
                    for child in head.children:
                        if child.pos_ == 'ADJ':
                            is_expletive = True
                            break
            
            if not replaced and not is_expletive:
                # Perform replacement
                # We need to handle case (Capitalize if start of sentence)
                replacement = last_subject
                if target_token.i == 0: # Start of sentence
                    replacement = replacement[0].upper() + replacement[1:] if replacement else replacement
                
                new_sent = _replace_span(
                    sent_text,
                    target_token.idx,
                    target_token.idx + len(target_token.text),
                    replacement,
                )
                final_sents.append(new_sent)
                replaced = True

        elif target_token and target_token.pos_ in ['PROPN', 'NOUN'] and last_subject:
            # If split sentences repeat the same subject phrase, prefer a pronoun in subsequent sentence.
            span_start, span_end = _subject_span(target_token)
            current_subject_text = sent_text[span_start:span_end]

            if _normalize_np(current_subject_text) == _normalize_np(last_subject):
                pronoun = infer_pronoun(target_token, context_span=doc)
                new_sent = _replace_span(sent_text, span_start, span_end, pronoun)
                final_sents.append(new_sent)
                replaced = True
                replaced_repeated_subject_with_pronoun = True
        
        if not replaced:
            final_sents.append(sent_text)
            
        # 2. Update last_subject for the NEXT sentence
        # If we replaced repeated noun by pronoun, keep antecedent for continuity.
        if not replaced_repeated_subject_with_pronoun:
            current_doc = nlp(final_sents[-1])
            found_subj = _extract_subject_text(current_doc)
            if found_subj:
                last_subject = found_subj
            
    return final_sents
