# NLP Sentence Simplification Project

This project simplifies complex English sentences using a hybrid pipeline:

- Rule-based syntactic simplification (spaCy dependency parsing)
- Neural fallback (local seq2seq models) when rule-based output is low quality

## Features

- Splits complex sentences into simpler sentences
- Handles conjunctions, relative clauses, appositives, adverbial clauses
- Applies lexical simplification and active voice conversion
- Performs basic coreference resolution
- Exposes a Flask API and simple web UI

## Project Structure

- `app.py`: Flask API, rule-first + neural-fallback orchestration
- `simplification.py`: Main rule-based pipeline
- `modules/`: Rule modules
- `models/`: Local model folders
- `Dataset/`: Dataset files for evaluation/training experiments
- `index.html`: Frontend

## Requirements

Install Python dependencies (preferably in a virtual environment):

```bash
pip install -r requirements.txt
```

If you use neural models, ensure these are installed:

- `torch`
- `transformers`
- `peft` (required for LoRA adapter models)

## Dataset and Models Setup

Large assets are not committed to GitHub (`Dataset/` and `models/` are gitignored), so after cloning you must place them manually in the project root.

Required local folders:

```text
project/
  Dataset/
  models/
```

Expected model subfolders under `models/`:

- `t5_small`
- `bart_base`
- `t5_base_lora`
- `bart_large_lora`

If these folders are missing:

- Rule-based simplification still runs.
- Neural fallback/model-based simplification will be skipped or raise loading warnings.

Tip: keep the exact folder names above so `app.py` can discover local models.

## Run the Project

From project root:

```bash
python app.py
```

Server starts at:

- `http://127.0.0.1:5001`

## API Usage

### Endpoint

- `POST /simplify`

### Sample Request

```bash
curl -s -X POST http://127.0.0.1:5001/simplify \
  -H 'Content-Type: application/json' \
  -d '{"text":"When the results were announced, the entire team celebrated their unexpected victory."}'
```

### Sample Response (shape)

```json
{
  "simplified": [
    {
      "text": "The results were announced.",
      "modifications": ["Split (Adverbial Clause)"]
    }
  ],
  "pipeline_used": "auto_fallback",
  "stages_applied": ["rule_based"],
  "warnings": []
}
```

## Simplification Flow

1. Rule-based pipeline runs first
2. Output is checked for quality/meaning retention
3. If rule-based output is weak, neural fallback is used
4. Best available output is returned

## Local Models

Current model directories detected in `models/`:

- `t5_small`
- `bart_base`
- `t5_base_lora`
- `bart_large_lora`

Note: Which models are actively used depends on `MODEL_DIRS` and `FALLBACK_STAGES` in `app.py`.

## Notes

- This is a development setup (Flask debug server)
- For production, use a production WSGI server
- Quality can vary by sentence structure; evaluation against dataset files is recommended
