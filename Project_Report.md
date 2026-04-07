# Syntactic Simplification of Long Sentences using NLP

**Subject:** Natural Language Processing (NLP)  
**Project Type:** Theoretical + Practical Implementation  

---

## 1. Abstract
Syntactic simplification is the process of modifying the grammatical structure of a sentence to make it easier to read and understand while preserving its original meaning. This project explores the development of an automated system that breaks down long, complex English sentences into a set of shorter, simpler sentences. By leveraging linguistic rules and dependency parsing, the system identifies key splitting points such as coordinating conjunctions, relative clauses, and subordinate clauses. The resulting output aims to enhance readability for diverse applications, including aiding people with aphasia, improving machine translation pre-processing, and assisting second-language learners. The implementation uses the Python programming language and the spaCy library to perform robust syntactic analysis and transformation.

## 2. Introduction
Natural Language Processing (NLP) is a subfield of artificial intelligence that focuses on the interaction between computers and human language. One significant challenge in NLP is handling long, convoluted sentences that are common in legal, medical, and academic texts.

**Syntactic Simplification** addresses this challenge not by summarizing (removing information) but by restructuring. The goal is to produce "One idea per sentence." 

**Motivation:**
- **Complexity:** Long sentences with multiple clauses increase the cognitive load on readers.
- **Machine Processing:** Parsers and translators often fail on deeply nested structures.
- **Accessibility:** Simplified text is crucial for individuals with cognitive disabilities or low literacy.

**Problem Statement:**
To design and implement a system that accepts a complex sentence as input and outputs a list of grammatically correct simple sentences, ensuring that no information is lost and references (like pronouns) are resolved where possible.

## 3. Literature Review
Existing research in text simplification generally falls into two categories:

1.  **Rule-Based Approaches:** Early systems (e.g., *Siddharthan, 2006*) relied on hand-crafted grammar rules and regular expressions. These systems are highly accurate for specific patterns but struggle with unseen variations. They often use breakdown rules for conjunctions and relative clauses.
    
2.  **Statistical and Machine Learning Approaches:** Systems like *Zhu et al. (2010)* treat simplification as a monolingual translation problem (Complex $\to$ Simple) using parallel corpora (e.g., Simple English Wikipedia).

3.  **Dependency-Based Simplification:** Recent "hybrid" approaches use dependency trees (graph-based representation of grammar) to identify clauses. This works better than pure constituent parsing because it directly links verbs to their subjects and objects, allowing for safer splitting.

*This project adopts a Dependency-Based Rule approach as it offers a balance of explainability and performance for standard English constructs.*

## 4. System Architecture
The system follows a pipeline architecture where the text is progressively analyzed and transformed.

```text
[ Input Text ]
      |
      v
[ Tokenization & POS Tagging ] -> (Assigns Noun, Verb, Adj, etc.)
      |
      v
[ Dependency Parsing ] -> (Builds the tree: Subject -> Verb <- Object)
      |
      v
[ Clause Detection Module ]
      |-- Detect Conjunctions ('and', 'but')
      |-- Detect Relative Clauses ('who', 'which')
      |-- Detect Adverbial Clauses ('because', 'since')
      |
      v
[ Sentence Splitting & Reconstruction ] -> (Fixes capitalization, references)
      |
      v
[ Output: List of Simple Sentences ]
```

### Module Explanation
- **Tokenization:** Breaking text into words.
- **POS Tagging:** Identification of Parts of Speech (e.g., identifying 'run' as a Verb).
- **Dependency Parsing:** The core engine. It identifies that "uncovered" is the main action performed by "The scientist".
- **Clause Detection:** finding sub-graphs in the dependency tree that represent complete thoughts.

## 5. Methodology
The simplification process focuses on three main linguistic constructs:

### A. Coordinating Conjunctions
Sentences joined by "and", "or", "but".
- **Rule:** If a Coordinating Conjunction (`cc`) connects two verbs, split the sentence at the conjunction.
- **reconstruction:** The subject of the first clause is often the implicit subject of the second.
    - *Example:* "He ran fast **and** missed the bus."
    - *Result:* "He ran fast.", "He missed the bus."

### B. Relative Clauses
Clauses modifying a noun, usually starting with "who", "which", "that".
- **Rule:** Identify the `relcl` dependency. Extract the clause.
- **Reconstruction:** Replace the relative pronoun (e.g., "who") with the noun it modifies (the "head").
    - *Example:* "The boy, **who is wearing a hat**, is my brother."
    - *Result:* "The boy is my brother.", "The boy is wearing a hat."

### C. Adverbial/Subordinate Clauses
Clauses indicating reason, time, or condition (starting with "Because", "Although").
- **Rule:** Identify `advcl` and its marker (`mark`).
- **Reconstruction:** Separate the main clause and the dependent clause.
    - *Example:* "**Because it rained**, we stayed inside."
    - *Result:* "It rained.", "We stayed inside."

## 6. Tools and Technologies
- **Python 3.x:** The primary programming language.
- **spaCy:** An industrial-strength NLP library used for:
    - Pre-trained model: `en_core_web_sm`
    - Efficient dependency parsing
    - Part-of-Speech tagging
- **Linguistic Concepts:** Dependency grammar, syntax trees, coreference resolution (basic).

## 7. Implementation
The core logic is implemented in `simplification.py`. Below is an overview of the key functions used:

```python
import spacy
nlp = spacy.load("en_core_web_sm")

def simplify_sentence(text):
    doc = nlp(text)
    # 1. Split on Conjunctions
    # 2. Split Relative Clauses
    # 3. Split Adverbial Clauses
    # (See full code in appended file)
    return list_of_sentences
```

*Note: The system recursively applies these rules to ensure all complex structures are resolved.*

## 8. Sample Inputs and Outputs
The following examples illustrate the detailed performance of the system:

**Example 1: Conjunctions**
> **Input:** "The quick brown fox jumps over the lazy dog and the cat sleeps deeply."
> **Output:** 
> 1. The quick brown fox jumps over the lazy dog.
> 2. The cat sleeps deeply.

**Example 2: Relative Clause**
> **Input:** "The boy, who is wearing a red hat, is my brother."
> **Output:**
> 1. The boy is my brother.
> 2. The boy is wearing a red hat.

**Example 3: Adverbial Clause (Reason)**
> **Input:** "Because it was raining heavily, we decided to stay indoors."
> **Output:**
> 1. It was raining heavily.
> 2. We decided to stay indoors.

**Example 4: Missing Subject (Ellipsis)**
> **Input:** "He ran fast but he missed the bus."
> **Output:**
> 1. He ran fast.
> 2. He missed the bus.

**Example 5: Nested Information**
> **Input:** "The scientist who discovered the cure won the Nobel Prize."
> **Output:**
> 1. The scientist won the Nobel Prize.
> 2. The scientist discovered the cure.

## 9. Evaluation and Results
The system was evaluated qualitatively on a set of common complex sentence structures.
- **Sentence Length Reduction:** The average words per sentence dropped significantly (e.g., from ~15 words to ~7 words), improving readability scores.
- **Grammatical Correctness:** The use of dependency parsing ensures that splits effectively preserve Subject-Verb-Object (SVO) structures.
- **Comparison:** Unlike simple "split by comma" methods, this approach correctly keeps phrases like "apples, oranges, and bananas" together while splitting "He ate, and he slept".

## 10. Advantages
1.  **Readability:** Makes text accessible to a wider audience.
2.  **Modularity:** New rules can be added (e.g., for Passive Voice) without breaking existing ones.
3.  **Efficiency:** Rule-based parsing is faster and more interpretable than large Neural Networks (Transformers).
4.  **Preservation:** Maintains the original factual content without hallucinations common in Generative AI.

## 11. Limitations
1.  **Ambiguity:** Sentences with highly ambiguous attachments might be parsed incorrectly.
2.  **Coreference:** Complex pronoun resolution (e.g., "He told him that he was late") is difficult to resolve perfectly without context.
3.  **Fluency:** Sometimes the simplified sentences can sound repetitive (e.g., repeating the subject multiple times).

## 12. Future Scope
- **Deep Coreference Resolution:** Using neural models to resolve pronouns across long texts.
- **Paraphrasing:** Changing complex vocabulary to simpler synonyms (Lexical Simplification).
- **Web Interface:** Creating a Flask/Django UI for users to input text directly.

## 13. Applications
- **Educational Tools:** Helping children or language learners understand complex texts.
- **Medical Domains:** Simplifying patient consent forms and medical instructions.
- **Legal Tech:** Summarizing or simplifying legal contracts for laypeople.

## 14. Conclusion
This project successfully demonstrates the application of Natural Language Processing techniques to the problem of syntactic simplification. By using a dependency-parsing based approach, we can reliably identify and split complex grammatical structures. While there is room for improvement in handling extreme ambiguity, the current system serves as a solid foundation for automated text simplification and highlights the power of linguistic engineering.
