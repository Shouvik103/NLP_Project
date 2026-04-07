from utils import nlp
import re

# Map subordinating conjunctions to discourse markers for the main clause
DISCOURSE_MARKERS = {
    "although": "However, ",
    "though": "However, ",
    "even though": "However, ",
    "while": "However, ",
    "whereas": "However, ",
    "because": "",
    "since": "",
    "as": "",
    "when": "Then, ",
    "after": "After that, ",
    "before": "Before that, ",
    "so": "As a result, ",
    "until": "",
    "once": "",
    "if": "",
    "unless": "",
}


def _get_discourse_marker(marker_text):
    """Return the appropriate discourse marker for the main clause."""
    return DISCOURSE_MARKERS.get(marker_text.lower().strip(), "")


def simplify_adverbial_clauses(text):
    """
    Handles: "Because it was raining, we stayed inside."
    -> "It was raining.", "We stayed inside."
    
    Also handles discourse markers:
    - "Although X, Y" -> "X.", "However, Y."
    - "When X, Y" -> "X.", "Then, Y."
    - "After X, Y" -> "X.", "After that, Y."
    - "Before X, Y" -> "X.", "Before that, Y."
    """
    doc = nlp(text)
    new_sents = []
    
    for sent in doc.sents:
        advcls = [t for t in sent if t.dep_ == "advcl"]
        
        if not advcls:
            # Fallback: check if the sentence starts with a known subordinator
            # that spaCy didn't parse as advcl
            result = _fallback_subordinator_split(sent)
            if result:
                new_sents.extend(result)
            else:
                new_sents.append(sent.text)
            continue
            
        # Heuristic: Split at the adverbial clause
        verb_head = sent.root
        advcl_verb = advcls[0]
        
        # Check marker (Because, Since, Although, When, After, Before)
        # spaCy labels 'When/After/Before' as advmod (not mark), so check both
        marker = None
        for child in advcl_verb.children:
            if child.dep_ == "mark":
                marker = child
                break
            if child.dep_ == "advmod" and child.pos_ == "SCONJ" and child.text.lower() in DISCOURSE_MARKERS:
                marker = child
                break
        
        if marker:
            # Get span of advcl
            advcl_tokens = list(advcl_verb.subtree)
            if not advcl_tokens: # Safety check
                 new_sents.append(sent.text)
                 continue

            start_i = min(t.i for t in advcl_tokens)
            end_i = max(t.i for t in advcl_tokens)

            # Get the discourse marker for the main clause
            discourse_prefix = _get_discourse_marker(marker.text)

            # Helper to reconstruct subject for participial phrases
            def reconstruct_clause(tokens, original_verb, marker_node, main_root):
                raw_text = " ".join([t.text for t in tokens if t != marker_node]).strip()
                
                # Check if we need reconstruction
                if original_verb.tag_ == 'VBG' and not any(c.dep_ == 'nsubj' for c in original_verb.children):
                    subject = None
                    subject_node = None
                    for child in main_root.children:
                        if child.dep_ in ['nsubj', 'nsubjpass']:
                            subject = "".join([t.text_with_ws for t in child.subtree]).strip()
                            subject_node = child
                            break
                    
                    if subject and subject_node:
                        aux = "was"
                        is_past = True
                        
                        if main_root.tag_ in ['VBP', 'VBZ']:
                            aux = "is"
                            is_past = False
                        elif main_root.tag_ in ['VBD', 'VBN']:
                            aux = "was"
                            is_past = True
                            
                        is_plural = False
                        if subject_node.tag_ in ['NNS', 'NNPS']:
                            is_plural = True
                        elif subject_node.pos_ == 'PRON' and subject_node.text.lower() in ['they', 'we', 'you']:
                            is_plural = True
                        elif subject_node.morph.get("Number") == ["Plur"]:
                             is_plural = True

                        if is_plural:
                            if is_past:
                                aux = "were"
                            else:
                                aux = "are"
                        
                        reconstructed = f"{subject} {aux} {raw_text}"
                        return reconstructed.capitalize()
                
                return raw_text.capitalize()
            
            # If advcl starts the sentence
            if start_i == sent.start:
                if marker.i == start_i:
                    part1 = reconstruct_clause(advcl_tokens, advcl_verb, marker, verb_head)
                    if part1.endswith(','): part1 = part1[:-1]
                    if not part1.endswith('.'): part1 += "."
                    
                    part2_span = sent.doc[end_i + 1 : sent.end]
                    part2 = part2_span.text.strip()
                    if part2.startswith(','): part2 = part2[1:].strip()
                    
                    # Apply discourse prefix
                    if discourse_prefix and part2:
                        # Keep proper-case for pronouns like 'I', proper nouns, etc.
                        first_word = part2.split()[0] if part2 else ""
                        if first_word and first_word[0].isupper() and len(first_word) > 1 and not first_word.isupper():
                            part2 = discourse_prefix + part2[0].lower() + part2[1:]
                        else:
                            part2 = discourse_prefix + part2
                    elif part2:
                        part2 = part2[0].upper() + part2[1:]
                    
                    new_sents.append(part1)
                    new_sents.append(part2)
                else:
                    new_sents.append(sent.text)
            else:
                # "We stayed inside because it was raining."
                part1_span = sent.doc[sent.start : start_i]
                part1 = part1_span.text.strip()
                if part1.endswith(','): part1 = part1[:-1]
                if not part1.endswith('.'): part1 += "."
                
                part2 = reconstruct_clause(advcl_tokens, advcl_verb, marker, verb_head)
                if not part2.endswith('.'): part2 += "."
                
                # For end-of-sentence advcl (e.g., "X because Y"), the discourse marker
                # goes on part1 (the main clause) only if it's a contrastive marker.
                # But for "so" the marker goes on part2.
                if discourse_prefix:
                    first_char = part2[0] if part2 else ""
                    if first_char.isupper() and len(part2) > 1:
                        part2 = discourse_prefix + part2[0].lower() + part2[1:]
                    else:
                        part2 = discourse_prefix + part2
                
                new_sents.append(part1)
                new_sents.append(part2)
        else:
            new_sents.append(sent.text)
            
    return new_sents


def _fallback_subordinator_split(sent):
    """
    Fallback for sentences where spaCy doesn't detect advcl but the sentence
    starts with a known subordinating conjunction followed by a comma-separated
    main clause.
    e.g. "When the fire broke out, firefighters arrived within minutes."
    """
    sent_text = sent.text.strip()
    
    # Check if starts with a known subordinator
    lower_text = sent_text.lower()
    matched_marker = None
    for marker in ["although", "though", "even though", "because", "since",
                    "when", "after", "before", "until", "once", "while", "whereas"]:
        if lower_text.startswith(marker + " "):
            matched_marker = marker
            break
    
    if not matched_marker:
        return None
    
    # Find the comma that separates subordinate clause from main clause
    comma_idx = sent_text.find(",")
    if comma_idx == -1:
        return None
    
    # Extract subordinate clause (without the marker) and main clause
    sub_clause = sent_text[len(matched_marker):comma_idx].strip()
    main_clause = sent_text[comma_idx + 1:].strip()
    
    if not sub_clause or not main_clause:
        return None
    
    # Format subordinate clause
    part1 = sub_clause[0].upper() + sub_clause[1:]
    if not part1.endswith("."): part1 += "."
    
    # Get discourse marker
    discourse_prefix = _get_discourse_marker(matched_marker)
    
    # Format main clause with discourse prefix
    if discourse_prefix:
        part2 = discourse_prefix + main_clause[0].lower() + main_clause[1:]
    else:
        part2 = main_clause[0].upper() + main_clause[1:]
    
    # Ensure period
    if not part2.endswith("."): part2 += "."
    
    return [part1, part2]
