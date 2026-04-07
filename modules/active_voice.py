import spacy
from utils import nlp
import lemminflect

def convert_to_active(text):
    """
    Attempts to convert passive voice to active voice.
    Example: "The ball was hit by John." -> "John hit the ball."
    """
    doc = nlp(text)
    
    # We need to find passive verbs.
    # In spaCy, this often looks like:
    #   auxpass (was) <--- head --- VERB (hit) --- agent ---> pobj (John)
    #                               |
    #                               |--- nsubjpass ---> ball
    
    # We will look for tokens with dep_ == 'auxpass'
    passive_spans = []
    
    for token in doc:
        if token.dep_ == 'auxpass':
            # This token typically modifies the main verb
            verb = token.head
            
            # Check if there is an agent ('by ...')
            agent = None
            for child in verb.children:
                if child.dep_ == 'agent' and child.text.lower() == 'by':
                    # The actual agent noun is usually the pobj of 'by'
                    for grandchild in child.children:
                        if grandchild.dep_ == 'pobj':
                            agent = grandchild
                            break
            
            # Check for the passive subject (the patient)
            patient = None
            for child in verb.children:
                if child.dep_ == 'nsubjpass':
                    patient = child
                    break
            
            if verb and agent and patient:
                # Found a full passive construction we can try to flip
                passive_spans.append({
                    'aux': token,
                    'verb': verb,
                    'agent': agent,
                    'patient': patient,
                    'by_node': agent.head # The 'by' token
                })
                
    # If no passive constructions found, return original
    if not passive_spans:
        return text
        
    # We proceed with the first one found to avoid overlapping complexities for now
    # (Recursion can handle multiple later if we re-process)
    
    case = passive_spans[0]
    
    # Construct new sentence pieces
    # 1. Agent subtree (New Subject)
    # 2. Verb (transformed)
    # 3. Patient subtree (New Object)
    
    # Get subtrees as text
    agent_text = "".join([t.text_with_ws for t in case['agent'].subtree]).strip()
    patient_text = "".join([t.text_with_ws for t in case['patient'].subtree]).strip()
    
    # Verb transformation
    # This is tricky without a conjugator. 
    # 'was hit' (past) -> 'hit' (past)
    # 'is eaten' (present) -> 'eats' (present)
    # 'has been seen' -> 'has seen'?
    
    # Simple heuristic:
    # If aux is 'was'/'were' -> convert verb to past tense.
    # If aux is 'is'/'are' -> convert verb to present tense (3rd person singular?).
    
    # For MVP, let's try to map the lemma of the verb to a simple form.
    # Ideally we'd use 'lemminflect' library but we might not have it installed.
    # Let's default to the verb's lemma + 'ed' or just the lemma if we can't do better, 
    # or rely on the fact that the 'verb' token in passive is usually a participle (VBN).
    
    # Heuristic for tense using lemminflect
    aux_text = case['aux'].text.lower()
    
    target_tag = 'VBD' # Default to past
    if aux_text in ['is', 'are']:
        target_tag = 'VBZ' # Present 3rd person singular
    elif aux_text in ['was', 'were']:
        target_tag = 'VBD' # Past tense
        
    # Get the inflected form
    new_verb = case['verb']._.inflect(target_tag)
    
    # lemminflect returns None if it fails? Actually it usually returns the form.
    # But just in case, or if it returns empty.
    if not new_verb:
        new_verb = case['verb'].lemma_

    # Reconstruct
    # We need to handle capitalization of the new start (Agent) and removing period if needed? 
    # Usually sentences end with period.
    
    # Capitalize first letter of agent
    agent_text = agent_text[0].upper() + agent_text[1:]
    
    # Lowercase the patient if it was the start? (It was nsubjpass, so commonly yes)
    # But names shouldn't be lowercased.
    # We'll trust the case is mostly fine or fix only if it was 'The table...' -> 'the table'
    if patient_text[0].isupper() and case['patient'].pos_ != 'PROPN':
         patient_text = patient_text[0].lower() + patient_text[1:]
    
    # Build: Agent + " " + NewVerb + " " + Patient + "."
    # We need to respect the original punctuation of the sentence end.
    # Let's assume a period for simplified output.
    
    new_sentence = f"{agent_text} {new_verb} {patient_text}."
    
    return new_sentence

