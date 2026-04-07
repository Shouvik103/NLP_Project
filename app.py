from flask import Flask, request, jsonify, send_from_directory
from simplification import overall_simplify
import os
from difflib import SequenceMatcher
import re

MODEL_DIRS = {
    "t5_small": "t5_small",
    "bart_base": "bart_base",
    "t5_base_lora": "t5_base_lora",
    "bart_large_lora": "bart_large_lora",
}

FALLBACK_STAGES = ["t5_small", "bart_base", "t5_base_lora", "bart_large_lora"]

_model_cache = {}

app = Flask(__name__)


def _load_seq2seq_model(model_key):
    """Lazy-load a local seq2seq model from the models directory."""
    if model_key in _model_cache:
        return _model_cache[model_key]

    if model_key not in MODEL_DIRS:
        raise ValueError(f"Unsupported model key: {model_key}")

    model_path = os.path.join(app.root_path, "models", MODEL_DIRS[model_key])
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    except ImportError as exc:
        raise RuntimeError(
            "transformers is not installed. Install it with: pip install transformers torch"
        ) from exc

    adapter_config = os.path.join(model_path, "adapter_config.json")

    # LoRA adapter folders contain adapter_config.json and need PEFT loading.
    if os.path.isfile(adapter_config):
        try:
            from peft import AutoPeftModelForSeq2SeqLM
        except ImportError as exc:
            raise RuntimeError(
                "peft is not installed. Install it with: pip install peft"
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoPeftModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)

    _model_cache[model_key] = (tokenizer, model)
    return tokenizer, model


def _run_model_simplification(text, model_key):
    """Run a local seq2seq simplification model and return generated text."""
    tokenizer, model = _load_seq2seq_model(model_key)

    prompt = text
    if model_key.startswith("t5"):
        prompt = f"simplify: {text}"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        num_beams=4,
        early_stopping=True,
    )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    return generated if generated else text


def _apply_neural_stages(items, stages):
    """Apply neural stages to each already-simplified sentence item."""
    warnings = []
    current_items = items

    stage_labels = {
        "t5_small": "Neural Simplification (T5-small)",
        "bart_base": "Neural Simplification (BART-base)",
        "t5_base_lora": "Neural Simplification (T5-base LoRA)",
        "bart_large_lora": "Neural Simplification (BART-large LoRA)",
    }

    for stage in stages:
        stage_failed = False
        next_items = []

        for item in current_items:
            text = item.get("text", "")
            mods = list(item.get("modifications", []))

            if not text.strip():
                next_items.append({"text": text, "modifications": mods})
                continue

            try:
                new_text = _run_model_simplification(text, stage)
                if new_text != text:
                    mods.append(stage_labels[stage])
                next_items.append({"text": new_text, "modifications": mods})
            except Exception as exc:  # noqa: BLE001 - keep API robust
                stage_failed = True
                warnings.append(f"{stage} stage skipped: {exc}")
                next_items.append({"text": text, "modifications": mods})

        current_items = next_items
        if stage_failed:
            # If this stage failed once, avoid repeated failures for remaining items.
            break

    return current_items, warnings


def _normalize_text(text):
    return " ".join((text or "").strip().lower().split())


def _flatten_items(items):
    return " ".join(item.get("text", "").strip() for item in items if item.get("text", "").strip()).strip()


def _content_overlap_ratio(source_text, candidate_text):
    """Rough content preservation ratio based on non-trivial token overlap."""
    stopwords = {
        "the", "a", "an", "and", "or", "but", "if", "then", "that", "this", "those", "these",
        "to", "of", "in", "on", "at", "for", "from", "with", "by", "as", "is", "are", "was", "were",
        "be", "been", "being", "it", "its", "he", "she", "they", "them", "their", "his", "her", "we",
        "you", "i", "do", "did", "does", "have", "has", "had", "not", "no", "so", "very", "also",
        "who", "which", "when", "where", "why", "how", "while", "although", "because", "after", "before",
    }
    src = {t for t in re.findall(r"[a-z0-9']+", (source_text or "").lower()) if len(t) > 2 and t not in stopwords}
    if not src:
        return 1.0
    cand = {t for t in re.findall(r"[a-z0-9']+", (candidate_text or "").lower()) if len(t) > 2 and t not in stopwords}
    return len(src & cand) / len(src)


def _choose_best_neural_output(original_text):
    """Run neural models on the original sentence and choose best candidate."""
    warnings = []
    candidates = []

    for stage in FALLBACK_STAGES:
        try:
            generated = _run_model_simplification(original_text, stage)
            candidates.append((stage, generated))
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"{stage} stage skipped: {exc}")

    if not candidates:
        return [{"text": original_text, "modifications": ["Fallback failed (kept original)"]}], warnings, None

    original_norm = _normalize_text(original_text)
    original_len = max(1, len(original_norm.split()))
    label_map = {
        "t5_small": "Neural Simplification (T5-small)",
        "bart_base": "Neural Simplification (BART-base)",
        "t5_base_lora": "Neural Simplification (T5-base LoRA)",
        "bart_large_lora": "Neural Simplification (BART-large LoRA)",
    }

    best_stage = None
    best_text = original_text
    best_score = -1.0

    for stage, cand_text in candidates:
        cand_norm = _normalize_text(cand_text)
        sim = SequenceMatcher(None, original_norm, cand_norm).ratio()
        overlap = _content_overlap_ratio(original_text, cand_text)
        length_ratio = len(cand_norm.split()) / original_len
        length_score = 1.0 - min(1.0, abs(length_ratio - 0.75))
        score = (0.35 * sim) + (0.45 * overlap) + (0.20 * length_score)

        if score > best_score:
            best_score = score
            best_stage = stage
            best_text = cand_text

    best_items = [{
        "text": best_text,
        "modifications": [label_map.get(best_stage, "Neural Simplification"), "Fallback (Rule-based low quality)"]
    }]
    return best_items, warnings, best_stage


def _rule_based_not_working(original_text, simplified_items):
    """
    Decide when to trigger neural fallback.
    We treat rule-based as "not working" if it produced no effective change.
    """
    if not simplified_items:
        return True

    combined_text = _flatten_items(simplified_items)
    if not combined_text:
        return True

    any_modification = any(item.get("modifications") for item in simplified_items)
    if not any_modification:
        if len(simplified_items) == 1 and _normalize_text(simplified_items[0].get("text", "")) == _normalize_text(original_text):
            return True

    original_norm = _normalize_text(original_text)
    combined_norm = _normalize_text(combined_text)

    # If output drops too much source content, fallback to neural.
    if _content_overlap_ratio(original_text, combined_text) < 0.55:
        return True

    # Very short outputs tend to omit meaning.
    if len(combined_norm.split()) < max(4, int(0.5 * len(original_norm.split()))):
        return True

    # Basic malformed-output checks.
    bad_patterns = ["..", "  ", " .", " ,", "However.", "Therefore.", "As a result,"]
    if any(p in combined_text for p in bad_patterns):
        return True

    return False

# Serve the index.html file
@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

# API endpoint for simplification
@app.route('/simplify', methods=['POST'])
def simplify():
    try:
        data = request.json
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']

        simplified_sentences = overall_simplify(text)
        warnings = []
        stages_applied = ['rule_based']

        if _rule_based_not_working(text, simplified_sentences):
            simplified_sentences, warnings, chosen_stage = _choose_best_neural_output(text)
            if chosen_stage:
                stages_applied.extend([chosen_stage, 'quality_fallback'])
            else:
                stages_applied.append('neural_failed')

        return jsonify(
            {
                'simplified': simplified_sentences,
                'pipeline_used': 'auto_fallback',
                'stages_applied': stages_applied,
                'warnings': warnings,
            }
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001, use_reloader=False)
