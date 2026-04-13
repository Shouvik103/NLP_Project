"""Graph-based sentence simplification module.

This module builds a lightweight dependency graph from spaCy parses,
detects clause roots, extracts clause subtrees, propagates missing
subjects when needed, and reconstructs simplified sentences.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Set, Tuple, Union

import spacy
from spacy.tokens import Doc, Span, Token


def _load_nlp() -> "spacy.language.Language":
  """Load and cache the spaCy English model used by this module."""
  try:
    return spacy.load("en_core_web_sm")
  except OSError as exc:
    raise RuntimeError(
      "spaCy model 'en_core_web_sm' is not installed. "
      "Install it with: python -m spacy download en_core_web_sm"
    ) from exc


_NLP = _load_nlp()


def build_dependency_graph(doc: Doc) -> Dict[int, Dict[str, Union[str, int, List[int]]]]:
  """Build a dependency graph representation keyed by token index.

  Each node stores:
  - token text
  - lemma
  - POS
  - dependency label
  - head index
  - children indices
  """
  graph: Dict[int, Dict[str, Union[str, int, List[int]]]] = {}
  for token in doc:
    graph[token.i] = {
      "text": token.text,
      "lemma": token.lemma_,
      "pos": token.pos_,
      "dep": token.dep_,
      "head": token.head.i,
      "children": [child.i for child in token.children],
    }
  return graph


def extract_clause_roots(doc: Doc) -> List[Tuple[Token, str]]:
  """Extract clause roots and their clause type from a parsed document.

  Clause types:
  - Conjunction Clause: dep_ in ("conj", "cc")
  - Relative Clause: dep_ == "relcl"
  - Adverbial Clause: dep_ == "advcl"
  """
  roots: List[Tuple[Token, str]] = []
  seen: Set[int] = set()

  for token in doc:
    if token.dep_ == "relcl":
      if token.i not in seen:
        roots.append((token, "Relative Clause"))
        seen.add(token.i)
      continue

    if token.dep_ == "advcl":
      if token.i not in seen:
        roots.append((token, "Adverbial Clause"))
        seen.add(token.i)
      continue

    # Keep conjunction splitting focused on predicate-level coordination
    # so noun lists (apples, oranges, bananas) are not broken.
    if token.dep_ == "conj" and token.pos_ in {"VERB", "AUX"}:
      if token.i not in seen:
        roots.append((token, "Conjunction Clause"))
        seen.add(token.i)
      continue

    # Include cc marker by mapping to its coordinating head when valid.
    if token.dep_ == "cc" and token.head.dep_ == "conj" and token.head.pos_ in {"VERB", "AUX"}:
      if token.head.i not in seen:
        roots.append((token.head, "Conjunction Clause"))
        seen.add(token.head.i)

  return roots


def extract_subtree_tokens(token: Token) -> List[Token]:
  """Extract a clause subtree rooted at the given token in original order."""
  return sorted(list(token.subtree), key=lambda t: t.i)


def _get_main_subject_tokens(doc: Doc) -> List[Token]:
  """Get the main clause subject tokens for subject propagation."""
  root = next((t for t in doc if t.dep_ == "ROOT"), None)
  if root is not None:
    for child in root.children:
      if child.dep_ in {"nsubj", "nsubjpass"}:
        return sorted(list(child.subtree), key=lambda t: t.i)

  # Fallback: first nominal subject in the sentence.
  for token in doc:
    if token.dep_ in {"nsubj", "nsubjpass"}:
      return sorted(list(token.subtree), key=lambda t: t.i)
  return []


def _contains_subject(tokens: Sequence[Token]) -> bool:
  """Check whether a token sequence already contains a subject."""
  return any(t.dep_ in {"nsubj", "nsubjpass", "csubj", "expl"} for t in tokens)


def _relative_anchor_tokens(clause_root: Token) -> List[Token]:
  """Build noun phrase anchor for relative clause subject replacement."""
  head = clause_root.head
  if head.i == clause_root.i:
    return []

  allowed_left_deps = {"det", "amod", "compound", "nummod", "poss"}
  left_mods = [t for t in head.lefts if t.dep_ in allowed_left_deps]
  phrase = sorted(left_mods + [head], key=lambda t: t.i)
  return phrase


def _trim_clause_markers(tokens: List[Token], clause_type: str) -> List[Token]:
  """Remove leading subordinators/relative markers from clause token lists."""
  trimmed = list(tokens)

  if clause_type == "Adverbial Clause":
    trimmed = [t for t in trimmed if t.dep_ not in {"mark"}]

  if clause_type == "Relative Clause":
    trimmed = [
      t
      for t in trimmed
      if not (
        t.tag_ in {"WDT", "WP", "WP$", "WRB"}
        or (t.dep_ == "mark" and t.text.lower() in {"that", "which", "who", "whom", "whose"})
      )
    ]

  return sorted(trimmed, key=lambda t: t.i)


def propagate_subject(doc: Doc, clause_tokens: Sequence[Token]) -> List[Union[Token, str]]:
  """Propagate main subject to clause tokens if the clause lacks a subject.

  Returns a list containing Tokens and/or string fragments used during
  sentence reconstruction.
  """
  clause_list = list(clause_tokens)
  if not clause_list:
    return []

  if _contains_subject(clause_list):
    return clause_list

  main_subj = _get_main_subject_tokens(doc)
  if not main_subj:
    return clause_list

  # Avoid duplicating near-identical leading content.
  clause_lemmas = {t.lemma_.lower() for t in clause_list}
  subj_lemmas = {t.lemma_.lower() for t in main_subj}
  if subj_lemmas and subj_lemmas.issubset(clause_lemmas):
    return clause_list

  return main_subj + clause_list


def reconstruct_sentence(tokens: Sequence[Union[Token, str]]) -> str:
  """Reconstruct a clean sentence from tokens while fixing spacing and punctuation."""
  if not tokens:
    return ""

  parts: List[str] = []
  for tok in tokens:
    if isinstance(tok, str):
      raw = tok.strip()
    else:
      raw = tok.text.strip()
    if raw:
      parts.append(raw)

  if not parts:
    return ""

  text = " ".join(parts)

  # Punctuation spacing cleanup.
  text = text.replace(" ,", ",")
  text = text.replace(" .", ".")
  text = text.replace(" ;", ";")
  text = text.replace(" :", ":")
  text = text.replace(" ?", "?")
  text = text.replace(" !", "!")

  # Normalize internal whitespace.
  text = " ".join(text.split()).strip()
  if not text:
    return ""

  # Capitalize initial letter.
  text = text[0].upper() + text[1:]

  # Ensure sentence-final period if no end punctuation exists.
  if text[-1] not in {".", "?", "!"}:
    text += "."

  return text


def _extract_main_clause_tokens(sent: Span, clause_roots: Sequence[Token]) -> List[Token]:
  """Build main clause tokens by removing subordinate clause subtrees."""
  if not clause_roots:
    return list(sent)

  removable: Set[int] = set()
  for root in clause_roots:
    # Remove subordinate clause subtree.
    if root.dep_ in {"relcl", "advcl"}:
      removable.update(t.i for t in root.subtree)
      # Also remove nearest comma separators around removed segment.
      left_idx = min(t.i for t in root.subtree)
      right_idx = max(t.i for t in root.subtree)
      if left_idx - 1 >= sent.start and sent.doc[left_idx - 1].text == ",":
        removable.add(left_idx - 1)
      if right_idx + 1 < sent.end and sent.doc[right_idx + 1].text == ",":
        removable.add(right_idx + 1)

    # For conjunction splits, remove coordinated subtree from main clause.
    if root.dep_ == "conj":
      removable.update(t.i for t in root.subtree)
      # Remove linked cc markers attached to same head.
      for child in root.head.children:
        if child.dep_ == "cc":
          removable.add(child.i)

  kept = [t for t in sent if t.i not in removable]
  return kept


def _dedupe_and_prune(sentences: Sequence[str]) -> List[str]:
  """Remove duplicates and obvious redundant overlaps."""
  unique: List[str] = []
  seen: Set[str] = set()

  for s in sentences:
    key = " ".join(s.lower().split())
    if not key or key in seen:
      continue
    seen.add(key)
    unique.append(s)

  # Remove very short sentences that are strict substrings of longer ones.
  keep = [True] * len(unique)
  for i, a in enumerate(unique):
    a_norm = " ".join(a.lower().split())
    a_len = len(a_norm.split())
    for j, b in enumerate(unique):
      if i == j:
        continue
      b_norm = " ".join(b.lower().split())
      b_len = len(b_norm.split())
      if a_norm in b_norm and a_len < b_len * 0.7:
        keep[i] = False
        break

  return [s for s, k in zip(unique, keep) if k]


def graph_based_simplify(text: str, debug: bool = False) -> List[Dict[str, Union[str, List[str]]]]:
  """Simplify a sentence using graph-based clause extraction.

  Args:
    text: Input sentence string.
    debug: If True, print detected roots and extracted subtrees.

  Returns:
    A list of dicts in the format:
    [{"text": "...", "modifications": ["Graph-Based Split (...)"]}]
  """
  cleaned = (text or "").strip()
  if not cleaned:
    return [{"text": "", "modifications": ["Graph-Based Split (No Change)"]}]

  try:
    doc = _NLP(cleaned)
  except Exception:
    return [{"text": cleaned, "modifications": ["Graph-Based Split (No Change)"]}]

  graph = build_dependency_graph(doc)
  clause_roots = extract_clause_roots(doc)

  if debug:
    print("[graph_based] Clause roots:")
    for root, ctype in clause_roots:
      print(f"  - {root.text} (idx={root.i}, dep={root.dep_}, type={ctype})")

  if not clause_roots:
    original = reconstruct_sentence(list(doc))
    return [{"text": original, "modifications": ["Graph-Based Split (No Change)"]}]

  outputs: List[Tuple[str, str]] = []

  # Sentence-level processing to support nested structures and multi-sentence inputs.
  for sent in doc.sents:
    sent_roots = [(r, t) for r, t in clause_roots if sent.start <= r.i < sent.end]
    if not sent_roots:
      continue

    # Main clause extraction.
    main_tokens = _extract_main_clause_tokens(sent, [r for r, _ in sent_roots])
    main_text = reconstruct_sentence(main_tokens)
    if main_text:
      outputs.append((main_text, "Graph-Based Split (Main Clause)"))

    for root, clause_type in sent_roots:
      clause_tokens = extract_subtree_tokens(root)

      if debug:
        print(
          f"[graph_based] Subtree {clause_type}: "
          f"{' '.join(t.text for t in clause_tokens)}"
        )

      clause_tokens = _trim_clause_markers(clause_tokens, clause_type)

      # Relative clause subject replacement with anchor noun phrase.
      if clause_type == "Relative Clause":
        anchor = _relative_anchor_tokens(root)
        if anchor:
          clause_tokens = anchor + clause_tokens

      clause_with_subject = propagate_subject(sent.doc, clause_tokens)
      clause_text = reconstruct_sentence(clause_with_subject)
      if clause_text:
        outputs.append((clause_text, f"Graph-Based Split ({clause_type})"))

  final_texts = _dedupe_and_prune([t for t, _ in outputs])

  if not final_texts:
    original = reconstruct_sentence(list(doc))
    return [{"text": original, "modifications": ["Graph-Based Split (No Change)"]}]

  # Preserve first-seen modification label per sentence.
  text_to_mod: Dict[str, str] = {}
  for t, m in outputs:
    key = " ".join(t.lower().split())
    if key not in text_to_mod:
      text_to_mod[key] = m

  return [
    {
      "text": s,
      "modifications": [text_to_mod.get(" ".join(s.lower().split()), "Graph-Based Split")],
    }
    for s in final_texts
  ]


