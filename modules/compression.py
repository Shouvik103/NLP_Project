import re

def simplify_compression(text):
    """
    Removes non-essential phrases:
    1. Content within parentheses.
    2. Common conversational fillers at the start.
    """
    
    # 1. Remove text inside parentheses ( ... )
    # We want to remove the parens and the text inside.
    # Case: "The study (which was done in 2020) showed..." -> "The study showed..."
    # Note: We need to handle spacing correctly. "word (paren) word" -> "word word" -> "word word"
    
    # Regex for balanced parentheses (non-nested for simplicity)
    new_text = re.sub(r'\s*\([^)]*\)', '', text)
    
    # 2. Remove filler words at start
    # "Basically, ..."
    # "Actually, ..."
    # "However, ..." (Sometimes this is a connector, removing it might lose cohesion, but "simplification" often asks for independent sentences. Let's be careful.)
    # "Furthermore, ..."
    
    FILLERS = [
        "Basically", "Actually", "Literally", "Simply put", "In fact", 
        "To be honest", "As a matter of fact", "It is known that", 
        "Needless to say"
    ]
    
    for filler in FILLERS:
        # Check start of string, followed by optional comma and space
        pattern = r'^' + re.escape(filler) + r'[:,\s]*\s*'
        if re.match(pattern, new_text, re.IGNORECASE):
            new_text = re.sub(pattern, '', new_text, count=1, flags=re.IGNORECASE)
            # Capitalize the new start
            if new_text:
                new_text = new_text[0].upper() + new_text[1:]
                
    return new_text.strip()
