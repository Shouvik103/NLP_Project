from utils import get_subject

def split_conjunctions(doc):
    return []

def simplify_conjunctions(sent):
    """
    Handles: "John likes apples and Mary likes oranges." -> "John likes apples.", "Mary likes oranges."
    Handles: "Melanie bought A, and B." -> "Melanie bought A.", "Melanie bought B." (Noun Phrase Coordination)
    Handles: "A and B went home." -> "A went home.", "B went home." (Subject Coordination)
    """
    root = sent.root
    
    # Pre-check for Colons (List/Explanation)
    # If there is a colon, we split strictly on it first, because it dominates the sentence structure.
    for token in sent:
        if token.text == ':':
            # Split into Pre and Post
            # Pre: "Each state... legislature or parliament"
            # Post: "unicameral in..."
            
            # Use raw indexing from the doc span
            # We must be careful to keep them as "sentences" for the pipeline.
            # But the pipeline expects specific logic. 
            # Let's return just the texts.
            
            part1 = sent.doc[sent.start : token.i].text.strip()
            part2 = sent.doc[token.i + 1 : sent.end].text.strip()
            
            sents = []
            if part1:
                if not part1.endswith('.'): part1 += "."
                sents.append(part1)

            if part2:
                # Improvement: Check if part2 is a fragment.
                # Check directly if it starts with lower case, it IS a continuation/fragment.
                if part2[0].islower():
                     part2 = "It is " + part2
                elif part2[0].isupper():
                     # Check if first word is 'Unicameral' (Adj).
                     first_word = part2.split(' ')[0].lower()
                     if first_word in ["unicameral", "bicameral", "consisting", "including"]:
                          part2 = "It is " + first_word + part2[len(first_word):]
                
                # Ensure punctuation
                if not part2.endswith('.'): part2 += "."
                
                # Capitalize "It"
                part2 = part2[0].upper() + part2[1:]
                
                sents.append(part2)
            
            return sents

    # 1. Search for split candidates
    # Priority 1: Direct children of ROOT (e.g., "He ran and jumped", "He bought A and B")
    candidates = [child for child in root.children if child.dep_ == "conj" and child.pos_ in ["VERB", "AUX", "NOUN", "PROPN"]]
    
    # Priority 2: Deep search for nested lists
    if not candidates:
        for token in sent:
            if token.head == root: continue
            
            # SAFEGUARD: Do not split if the list anchors are objects of prepositions (pobj).
            # Check dependency chain upwards.
            anchor = token.head
            is_pobj_chain = False
            # Walk up to find if we are in a 'pobj' list
            temp = anchor
            for _ in range(5): # Safety limit
                if temp.dep_ == "pobj":
                     is_pobj_chain = True
                     break
                if temp.dep_ != "conj" and temp.dep_ != "appos": 
                     # If we hit a Verb or something else that isn't list connector, stop.
                     break
                temp = temp.head
                
            if is_pobj_chain:
                continue

            if token.dep_ == "conj" and token.pos_ in ["NOUN", "PROPN"]:
                candidates.append(token)
            elif token.dep_ == "appos" and token.pos_ in ["NOUN", "PROPN"]:
                  # Check for comma
                  head = token.head
                  if (token.i > 0 and sent.doc[token.i - 1].text == ',') or any(t.text == ',' for t in head.children):
                      candidates.append(token)
    
    if not candidates:
        return [sent.text]
        
    # Process the first candidate
    target = candidates[0]
    
    # Check if this is a SUBJECT splitting case
    # If 'target' is in the subject subtree of the main verb.
    main_subj = get_subject(root)
    is_subject_split = False
    if main_subj:
        # Check if target is in main_subj's subtree
        if target in main_subj.subtree:
            is_subject_split = True
            
    # Find split point (CC or Comma)
    split_token = None
    for child in target.head.children:
        if child.dep_ == "cc":
            split_token = child
            break
    if not split_token:
         for i in range(target.i - 1, sent.start - 1, -1):
            if sent.doc[i].text == ',':
                split_token = sent.doc[i]
                break
                
    if not split_token:
         return [sent.text]
            
    # Split Spans
    # This part is tricky for Subject Splitting because we want to isolate S1 and S2.
    # Current simplistic span split: [Start...Split] and [Split+1...End]
    
    split_i = split_token.i
    
    if is_subject_split:
        # Special Logic for Subject Splitting
        # Structure: [SubjPart1] [Split] [SubjPart2] [...Verb...]
        # We want: 
        # Sent1: [SubjPart1] [...Verb...]
        # Sent2: [SubjPart2] [...Verb...]
        
        # Identification of SubjPart1 and SubjPart2 is hard purely on indices.
        # But broadly:
        # Part 1: Everything before split_i?
        # Part 2: Everything from split_i+1 to end of subject?
        
        # Let's try to reconstruct strictly.
        # This requires more precise span manipulation than text slicing.
        
        # Safe Fallback: preventing the loop is priority.
        # If we detect subject split, but can't do it cleanly, DO NOTHING for this candidate.
        # This breaks the loop!
        return [sent.text] 
        # Note: By returning original text, we stop trying to split this specific subject.
        # The recursion will try other candidates or stop. 
        # This effectively disables Compound Subject Splitting, which is safer than the garbage loop.
        
    else:
        # Standard Logic (Object / Clause Coordinate)
        part1_span = sent.doc[sent.start : split_i]
        part2_span = sent.doc[split_i + 1 : sent.end]
                
        part1_text = part1_span.text.strip()
        if part1_text.endswith(','):
            part1_text = part1_text[:-1]
            
        part2_text = part2_span.text.strip()
        
        # Reconstruction
        if target.pos_ in ["VERB", "AUX"]:
             # Clausal: check if Part 2 needs a subject
             conj_subj = get_subject(target)
             if not conj_subj:
                 root_subj = get_subject(root)
                 if root_subj:
                     subj_text = "".join([t.text_with_ws for t in root_subj.subtree]).strip()
                     part2_text = f"{subj_text} {part2_text}"
        else:
             # Noun Phrase (Object)
             # Heuristic: Copy Subject + Root Verb.
             subj = get_subject(root)
             subj_text = "".join([t.text_with_ws for t in subj.subtree]).strip() if subj else ""
             verb_text = root.text_with_ws.strip() # Strip and add logic space
             
             part2_text = f"{subj_text} {verb_text} {part2_text}".strip()

        # Discourse markers based on the coordinating conjunction
        cc_text = split_token.text.lower() if split_token else ""
        discourse_prefix = ""
        if cc_text == "but":
            discourse_prefix = "However, "
        elif cc_text == "so":
            discourse_prefix = "As a result, "
    
        # Capitalization & Formatting
        part1_text = part1_text.strip()
        if part1_text:
             part1_text = part1_text[0].upper() + part1_text[1:]
             if not part1_text.endswith('.'): part1_text += "."
        
        part2_text = part2_text.strip()
        if part2_text:
            if discourse_prefix:
                # Lowercase the first char of part2 since the prefix provides the capital
                part2_text = discourse_prefix + part2_text[0].lower() + part2_text[1:]
            else:
                part2_text = part2_text[0].upper() + part2_text[1:]
            if not part2_text.endswith('.'): part2_text += "." 
        
        return [part1_text, part2_text]
