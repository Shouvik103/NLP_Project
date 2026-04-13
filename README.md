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
- `notebooks/`: Jupyter notebooks for model training & evaluation
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

> **Note:** The trained model weights are too large to host on GitHub. Instead, the Jupyter notebooks used to train and evaluate every model are included in this repository so you can reproduce them yourself.

### Reproducing the Models

The following notebooks (run on Google Colab with GPU) contain all training and evaluation code:

| Notebook | Description |
|----------|-------------|
| `notebooks/NLP_Project.ipynb` | End-to-end project notebook — data prep, rule-based pipeline testing, and initial model experiments |
| `notebooks/Stage_2_seq2seq (1).ipynb` | Training and evaluation of the **T5-Small** and **BART-Base** seq2seq baselines |
| `notebooks/llm_finetuning_lora (2).ipynb` | LoRA fine-tuning of **T5-Base** and **BART-Large** using the PEFT library |
| `notebooks/stage3_analysis (2).ipynb` | Comparative analysis and metric computation across all four models |

After training, download the model checkpoints and place them under `models/` with the following folder names:

```text
models/
  t5_small/
  bart_base/
  t5_base_lora/
  bart_large_lora/
```

### What if the models are missing?

- **Rule-based simplification still works** — no model files are needed.
- **Neural fallback / model-based simplification** will be skipped or show loading warnings.

> **Tip:** Keep the exact folder names listed above so `app.py` can auto-discover local models.

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
