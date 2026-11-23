# Champion–Challenger Model Comparison with Hugging Face and MLflow

**Student USN:** 1MS25CI099 &nbsp;·&nbsp; Skill Lab (MLOps with MLflow)

## 1. Task and models

| | |
|---|---|
| Task | Binary sentiment classification (positive / negative) |
| Dataset | SST-2 validation split (`stanfordnlp/sst2`), movie-review sentences with human labels |
| Subset size | 200 sentences, sampled with `random_state=42` (saved in `data/eval_inputs.csv`) |
| Model A (current Champion) | `distilbert-base-uncased-finetuned-sst-2-english` (~67M params) |
| Model B (Challenger) | `siebert/sentiment-roberta-large-english` (~355M params) |

Both models output POSITIVE/NEGATIVE, which `hf_wrapper.py` maps to `positive`/`negative`.
Both are wrapped in a custom `mlflow.pyfunc.PythonModel` (as in Lab 9), so any logged
version can be loaded from MLflow and used without writing Hugging Face code.

## 2. How to run

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt

# terminal 1: tracking server (leave it running)
python -m mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db

# terminal 2
python evaluate.py --n 10          # quick debug run
python evaluate.py                 # final run, 200 examples
python register_champion.py        # registers A and B, sets champion/challenger aliases
python predict_from_mlflow.py      # loads models:/sentiment-model@champion, predicts on data/new_data.csv

# stretch goal: swap the aliases and rerun the SAME script without editing it
python register_champion.py --flip
python predict_from_mlflow.py
```

| File | What it does |
|---|---|
| `config.py` | USN, model IDs, dataset, seed, and the Champion rule constants |
| `hf_wrapper.py` | pyfunc wrapper around the Hugging Face pipeline + label mapping |
| `evaluate.py` | runs both models on the same 200 inputs, logs runs, metrics, artifacts, models and the decision |
| `register_champion.py` | finds the latest `compare_sentiment` run by name (no hard-coded run IDs), registers both models, sets aliases |
| `predict_from_mlflow.py` | loads the Champion by alias only; never imports transformers |

## 3. Champion rule (written before the final run)

1. **Primary metric:** macro-F1 on the 200-sentence evaluation set. **Higher is better.**
2. **Minimum improvement margin:** the Challenger replaces the Champion only if its macro-F1 is
   at least **1.0 percentage point** higher (delta ≥ 0.01). With 200 examples one sentence is
   0.5 pp, so a 1 pp margin means the Challenger must get at least two more sentences right;
   a smaller gap could be noise from which sentences happened to be sampled.
3. **Guardrail:** reject the Challenger if its average latency is more than **10×** the
   Champion's, even if it is more accurate. Model B is about 5× bigger, so some slowdown is
   expected and acceptable; 10× would be too slow for interactive use.
4. **Tie-breaker:** if the difference is inside the margin, the Champion stays, because it is
   smaller and faster.

Direction of every metric: accuracy, macro-F1, macro-precision, macro-recall → higher is better.
`avg_latency_ms`, `total_inference_s`, `latency_ratio_b_over_a` → lower is better.

## 4. Results

Final run: 200 SST-2 validation sentences, CPU, batch size 16 (from `outputs/metrics_summary.json`).

| Metric | A: distilbert-sst2 (Champion) | B: roberta-large (Challenger) | B − A |
|---|---|---|---|
| Accuracy ↑ | 0.9350 | 0.9450 | +0.0100 |
| **Macro-F1 (primary) ↑** | **0.9343** | **0.9444** | **+0.0101 (1.01 pp)** |
| Macro-precision ↑ | 0.9359 | 0.9460 | +0.0101 |
| Macro-recall ↑ | 0.9331 | 0.9432 | +0.0101 |
| Avg latency (ms/example) ↓ | 25.96 | 223.20 | 8.60× slower |
| Total inference (s) ↓ | 5.19 | 44.64 | +39.45 s |
| Model size | ~67M params | ~355M params | ~5.3× bigger |

The two models disagree on **18 of 200** sentences. Because the task is binary, exactly one
model is right on each disagreement: B is right on 10 and A on 8, a net gain of 2 sentences for B.

Plots: `outputs/compare_metrics.png`, `outputs/compare_latency.png`,
`outputs/confusion_matrix_*.png`. Disagreements: `outputs/disagreements.csv`.

**Registry result:** `sentiment-model` v1 = Model A (`@challenger`), v2 = Model B (`@champion`).
`predict_from_mlflow.py` loaded `models:/sentiment-model@champion` → v2
(`siebert/sentiment-roberta-large-english`) and matched my expected label on all 8 new sentences.

### Qualitative examples (5 cases where the models disagree)

*(from `outputs/disagreements.csv`; 5 of the 18 disagreements: B right in 3, A right in 2)*

| # | Sentence | Gold | A | B | Who was right | Comment |
|---|---|---|---|---|---|---|
| 1 | "oh come on ." | neg | pos | **neg** | B | Very short, no sentiment words at all. The negativity is purely in the tone, which the larger model picks up better. |
| 2 | "a bloated gasbag thesis grotesquely impressed by its own gargantuan aura of self-importance ..." | neg | pos | **neg** | B | Rare, insulting vocabulary. A seems misled by the word "impressed"; B understands the sarcastic phrasing. |
| 3 | "writer/director joe carnahan 's grimy crime drama is a manual of precinct cliches , but it moves fast enough to cover its clunky dialogue and lapses in logic ." | pos | neg | **pos** | B | Mostly negative words ("cliches", "clunky", "lapses") but the "but it moves fast enough" clause flips the overall verdict. B handles the contrast; A counts the negative words. |
| 4 | "fun , flip and terribly hip bit of cinematic entertainment ." | pos | **pos** | neg | A | "terribly" is used as an intensifier ("very"), not as criticism. B seems to read it as negative. |
| 5 | "it 's a bit disappointing that it only manages to be decent instead of dead brilliant ." | neg | **neg** | pos | A | Positive words ("decent", "brilliant") inside a negative judgement. B is fooled by them; A keys on "disappointing". |

Pattern: both models struggle with mixed or ironic sentences. B is better at sarcasm and
"X, but Y" contrasts, while A is sometimes better when a single strong word ("disappointing")
decides the label. Some SST-2 labels are also debatable (e.g. "nothing is sacred in this
gut-buster ." is labelled negative), so a few "errors" may be label noise.

## 5. Decision

**Champion: Model B (`siebert/sentiment-roberta-large-english`).** Applying the rule written
before the run, the Challenger's macro-F1 is 0.9444 against the Champion's 0.9343, a gain of
1.01 percentage points, which just meets the 1.0 pp margin. Its average latency is 8.6× the
Champion's, which is inside the 10× guardrail, so the rule says the Challenger replaces the
Champion, and the registry aliases were set that way.

The improvement is real but small. On 200 sentences, 1 pp is only 2 sentences: the models
disagree on 18 sentences, B is right on 10 of them and A on 8. With a sample this size, a
different random seed could easily shrink the gap below the margin, so I would not claim B is
clearly better from this experiment alone. A larger evaluation set (the full 872 validation
sentences) would be needed to be confident.

The trade-off is cost. B is about 5× bigger and took 223 ms per sentence against 26 ms, and
44.6 s against 5.2 s for the whole set. For an offline batch job, B's small accuracy gain is
worth it. For a real-time app on a CPU, A would be the more practical choice, and a tighter
latency guardrail (for example 5×) would have kept A as Champion. Because prediction goes
through the `@champion` alias, switching back only needs `register_champion.py --flip`, with no
code change.

## 6. Limitations and declaration

- 200 sentences is a small sample; differences of 1–2 pp could change with a different seed.
- SST-2 sentences are short movie-review snippets; results may not carry over to long reviews,
  tweets or Indian-English text.
- Model A was fine-tuned on the SST-2 *training* split, so it is evaluated on in-domain data;
  Model B was trained on a mix of 15 sentiment datasets. This favours Model A slightly.
- Latency was measured on my laptop CPU, batch size 16; numbers will differ on other machines.
- **AI assistance:** I used Claude (an AI assistant) to help draft the scripts and this README
  structure. I ran every script myself, checked the outputs, and can explain each line.
