"""Shared settings for the Champion-Challenger mini project.

STUDENT_USN = "1MS25CI099"

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "hf-sentiment-champion-challenger"
REGISTERED_MODEL_NAME = "sentiment-model"
TASK = "sentiment"

# Model A = current Champion, Model B = Challenger (assignment table #1 vs #2)
MODEL_A = {
    "hf_model_id": "distilbert-base-uncased-finetuned-sst-2-english",
    "short_name": "distilbert-sst2",
}
MODEL_B = {
    "hf_model_id": "siebert/sentiment-roberta-large-english",
    "short_name": "roberta-large",
}

# Evaluation data: SST-2 validation split (872 movie-review sentences, labelled 0=neg, 1=pos)
DATASET_NAME = "stanfordnlp/sst2"
DATASET_SPLIT = "validation"
EVAL_SAMPLES = 200
SEED = 42

# Inference settings (same for both models)
BATCH_SIZE = 16
MAX_LENGTH = 128

# ---- Champion rule (written BEFORE the final run, do not change after) ----
PRIMARY_METRIC = "macro_f1"      # higher is better
MARGIN = 0.01                    # Challenger must beat Champion by >= 1 percentage point
MAX_LATENCY_RATIO = 10.0         # guardrail: reject Challenger if avg latency > 10x Champion's
