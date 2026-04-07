from utils import nlp
from modules.pronoun_utils import infer_pronoun


def simplify_relative_clauses(text):
    """
    Handles: "The boy, who is wearing a red hat, is my brother." 
    -> "The boy is my brother.", "The boy is wearing a red hat."
    """
    doc = nlp(text)
    new_sents = []
    
    for sent in doc.sents:
        # Look for relative clauses (relcl)
        relcls = [t for t in sent if t.dep_ == "relcl"]
        
        if not relcls:
            new_sents.append(sent.text)
            continue
            
        # For simplicity, handle the first relative clause found
        relcl_verb = relcls[0]
        noun_head = relcl_verb.head # The noun being modified
        
        rel_marker = None
        for child in relcl_verb.children:
            if child.tag_ in ["WDT", "WP", "WP$"] or child.dep_ == "mark": # who, which, that
                rel_marker = child
                break
        
        if rel_marker:
            # Span of relative clause
            rc_span_tokens = list(relcl_verb.subtree)
            rc_start = min(t.i for t in rc_span_tokens)
            rc_end = max(t.i for t in rc_span_tokens)
            
            # Sentence 1: text before RC + text after RC
            pre_text = sent.doc[sent.start : rc_start].text.strip()
            if pre_text.endswith(','): pre_text = pre_text[:-1]
            
            post_text = sent.doc[rc_end+1 : sent.end].text.strip()
            if post_text.startswith(','): post_text = post_text[1:].strip()
            
            sent1 = f"{pre_text} {post_text}".strip()
            sent1 = sent1.replace("  ", " ")
            if not sent1.endswith('.'): sent1 += "."
            
            # Sentence 2: The relative clause, with marker replaced by noun head
            head_noun_phrase = ""
            head_subtree = []
            for t in noun_head.subtree:
                if t.i < rc_start or t.i > rc_end:
                    head_subtree.append(t)
            
            head_text = noun_head.text
            for child in noun_head.children:
                if child.dep_ == "det":
                     head_text = f"{child.text} {head_text}"
            
            pronoun_subject = infer_pronoun(noun_head, context_span=sent)

            rc_text_parts = []
            for t in rc_span_tokens:
                if t == rel_marker:
                    rc_text_parts.append(pronoun_subject)
                else:
                    rc_text_parts.append(t.text)
            
            sent2 = " ".join(rc_text_parts)
            if sent2:
                sent2 = sent2[0].upper() + sent2[1:]
            if not sent2.endswith('.'): sent2 += "."
            
            new_sents.append(sent1)
            new_sents.append(sent2)
        else:
            new_sents.append(sent.text)
            
    return new_sents
