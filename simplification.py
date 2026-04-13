from utils import nlp
from modules.conjunctions import simplify_conjunctions
from modules.relative import simplify_relative_clauses
from modules.appositives import simplify_appositives
from modules.adverbial import simplify_adverbial_clauses
from modules.lexical import simplify_lexical
from modules.active_voice import convert_to_active
from modules.compression import simplify_compression
from modules.coreference import resolve_coreference
from modules.graph_based import graph_based_simplify

def track_change(original, new_text, tag):
    """Helper to detect change and return tag if changed."""
    if original != new_text:
        return tag
    return None

def overall_simplify(text, use_graph_based=True, graph_debug=False):
    """
    Apply pipeline of simplifications.
    Returns list of dicts: {'text': str, 'modifications': list}
    """
    # Initialize with 'Sentence Object' abstraction
    # current_items = [{'text': str, 'mods': []}]
    
    # Step 0: Pre-processing (Compression)
    compressed_text = simplify_compression(text)
    initial_mods = []
    if compressed_text != text:
        initial_mods.append("Compression (Removed fillers/parentheticals)")
        
    current_items = [{'text': compressed_text, 'mods': initial_mods}]

    # Step 0b: Graph-based splitting (optional)
    if use_graph_based:
        graph_items = []
        for item in current_items:
            try:
                g_results = graph_based_simplify(item['text'], debug=graph_debug)
            except Exception:
                graph_items.append(item)
                continue

            if not g_results:
                graph_items.append(item)
                continue

            # Treat no-change output as pass-through.
            if (
                len(g_results) == 1
                and g_results[0].get('text', '').strip() == item['text'].strip()
                and any('No Change' in m for m in g_results[0].get('modifications', []))
            ):
                graph_items.append(item)
                continue

            added_any = False
            for g in g_results:
                g_text = g.get('text', '').strip()
                if not g_text:
                    continue

                new_mods = item['mods'].copy()
                for m in g.get('modifications', []):
                    if m and 'No Change' not in m and m not in new_mods:
                        new_mods.append(m)

                graph_items.append({'text': g_text, 'mods': new_mods})
                added_any = True

            if not added_any:
                graph_items.append(item)

        current_items = graph_items

    # Step 1: Conjunctions (Recursive)
    # logic: iterate, if split happens, inherit mods and add 'Split'
    
    for _ in range(20): 
        next_items = []
        changed_any = False
        
        for item in current_items:
            doc = nlp(item['text'])
            # We assume simplify_conjunctions handles the whole text chunk if it's one sentence, 
            # or we iterate sents. existing logic iterated sents.
            
            # The existing logic was:
            # for sent in doc.sents: simplified = simplify_conjunctions(sent)
            # But item['text'] might be multiple sentences if spacy splits it so.
            # We strictly want to operate on the units we have.
            # But simplify_conjunctions expects a Span (sent).
            
            item_full_text_reconstructed = []
            item_was_split = False
            
            # We process each sentence within the current block (usually just 1, but maybe more)
            sents = list(doc.sents)
            if not sents: # Handle empty/whitespace
                next_items.append(item)
                continue
                
            for sent in sents:
                simplified_list = simplify_conjunctions(sent)
                
                # Check for split
                if len(simplified_list) > 1:
                    item_was_split = True
                    # Create new items
                    for s_text in simplified_list:
                        new_mods = item['mods'].copy()
                        new_mods.append("Split (Conjunctions)")
                        next_items.append({'text': s_text, 'mods': new_mods})
                    changed_any = True
                elif simplified_list and simplified_list[0] != sent.text:
                    # Changed but not split (e.g. rearrangement or removal)
                    # For conjunctions module, this is usually a split or reconstruct.
                    # We treat it as one item
                    new_mods = item['mods'].copy()
                    new_mods.append("Rephrased (Conjunctions)")
                    next_items.append({'text': simplified_list[0], 'mods': new_mods})
                    changed_any = True
                else:
                    # No change
                    # We might have multiple sentences in one item, if so we split them "implicitly" 
                    # by appending separately? 
                    # The original logic flattened everything.
                    # Let's flatten.
                     next_items.append({'text': sent.text, 'mods': item['mods'].copy()})
                     
            # Wait, if we had 1 item -> 2 sentences -> no change, 
            # we effectively split them into 2 items with same mods?
            # Yes, that is fine.
            
        current_items = next_items
        if not changed_any:
            break
            
    step1_items = current_items
    
    # Helper for 1-to-N steps (Relative, Appositive, Adverbial)
    def apply_splitting_step(items, func, step_name):
        new_items = []
        for item in items:
            results = func(item['text'])
            if len(results) > 1:
                for res in results:
                    n_mods = item['mods'].copy()
                    n_mods.append(f"Split ({step_name})")
                    new_items.append({'text': res, 'mods': n_mods})
            elif results and results[0] != item['text']:
                 n_mods = item['mods'].copy()
                 n_mods.append(f"Rephrased ({step_name})")
                 new_items.append({'text': results[0], 'mods': n_mods})
            else:
                 new_items.append(item)
        return new_items

    # Step 2: Relative Clauses
    step2_items = apply_splitting_step(step1_items, simplify_relative_clauses, "Relative Clause")
    
    # Step 2b: Appositives
    step2b_items = apply_splitting_step(step2_items, simplify_appositives, "Appositive")
        
    # Step 3: Adverbial Clauses (MUST run before active voice conversion)
    step3_items = apply_splitting_step(step2b_items, simplify_adverbial_clauses, "Adverbial Clause")
    
    # Step 4: Lexical & Active Voice (1-to-1 steps)
    
    step4_items = []
    for item in step3_items:
        txt = item['text']
        mods = item['mods'].copy()
        
        # Lexical
        lex_txt = simplify_lexical(txt)
        if lex_txt != txt:
            mods.append("Lexical Simplification")
            
        # Active Voice
        active_txt = convert_to_active(lex_txt)
        if active_txt != lex_txt:
            mods.append("Active Voice Conversion")
            
        step4_items.append({'text': active_txt, 'mods': mods})
        
    # Step 5: Coreference Resolution
    # This takes the WHOLE list.
    input_texts = [i['text'] for i in step4_items]
    resolved_texts = resolve_coreference(input_texts)
    
    final_result = []
    for i, res_txt in enumerate(resolved_texts):
        # If text changed from input_texts[i], add tag
        orig_txt = step4_items[i]['text']
        item_mods = step4_items[i]['mods']
        
        if res_txt != orig_txt:
            item_mods.append("Coreference Resolution")
            
        final_result.append({'text': res_txt, 'modifications': item_mods})
    
    return final_result


# Example Execution
if __name__ == "__main__":
    print("--- Syntactic Simplification Project ---")
    print(f"Loaded spaCy model: {nlp.meta['name']}\n")
    
    while True:
        try:
            user_input = input("\nEnter a sentence to simplify (or type 'exit' to quit): ").strip()
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("Exiting program.")
                break
            
            if not user_input:
                continue
                
            print(f"Original: {user_input}")
            simplified = overall_simplify(user_input)
            print("Simplified:")
            for simple in simplified:
                print(f"  - {simple}")
            print("-" * 40)
            
        except KeyboardInterrupt:
            print("\nExiting program.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
