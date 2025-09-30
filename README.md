# myntra-product-review-sentiment-analysis-using-nlp-machine-learning

Classify Myntra product reviews into sentiment classes (Positive / Neutral / Negative) using a clean, reproducible NLP/ML pipeline. The repo includes data prep, model training (classic ML & Transformer baselines), evaluation, and a simple API/CLI for inference.

✨ Features
End-to-end pipeline: ingest → clean → label → featurize → train → evaluate → serve
Multiple model options: Logistic Regression / SVM / XGBoost / DistilBERT
Text preprocessing: lowercasing, punctuation/URL/user mention removal, emoji handling, negation handling, lemmatization
Class imbalance handling (class weights / focal loss / undersampling/oversampling)
Rich metrics: accuracy, precision/recall/F1, ROC-AUC, PR-AUC, confusion matrix
Reproducible: config-driven runs + seeds + deterministic splits
Lightweight serving: FastAPI endpoint and CLI predictor

📦 Project Structure
myntra-sentiment/
├─ data/
│  ├─ raw/                # raw reviews (CSV/JSONL)
│  ├─ interim/            # after cleaning
│  └─ processed/          # train/dev/test splits
├─ notebooks/
│  ├─ 01_eda.ipynb
│  └─ 02_error_analysis.ipynb
├─ src/
│  ├─ config.py
│  ├─ data_utils.py
│  ├─ preprocess.py
│  ├─ features.py         # TF-IDF, embeddings
│  ├─ models/
│  │  ├─ classic.py       # LR/SVM/XGB
│  │  └─ transformers.py  # DistilBERT
│  ├─ train.py
│  ├─ evaluate.py
│  ├─ infer.py
│  └─ api.py              # FastAPI app
├─ scripts/
│  ├─ prepare_data.sh
│  ├─ train_classic.sh
│  └─ train_transformer.sh
├─ requirements.txt
├─ README.md
└─ LICENSE

🗂️ Data
Input: A CSV/JSONL containing at least:
review_text (string)
rating or label (int/string). If only rating exists, labels are derived (e.g., 1–2 = Negative, 3 = Neutral, 4–5 = Positive).
Place files in data/raw/ (e.g., myntra_reviews.csv).

⚠️ Ethics & TOS: If you scrape reviews, follow Myntra’s Terms of Service, robots.txt, and applicable laws. Prefer their official data exports or public datasets. Remove PII and respect rate limits.
Example CSV
review_id,review_text,rating
abc123,"Loved the fabric and fit!",5
def456,"Color faded after one wash.",2
ghi789,"It's okay for the price.",3

🧰 Environment Setup
# 1) Python
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)

# 2) Dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3) NLTK/Spacy assets (if used)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
python -m spacy download en_core_web_sm
requirements.txt (suggested)
pandas
numpy
scikit-learn
xgboost
nltk
spacy
emoji
matplotlib
seaborn
tqdm
fastapi
uvicorn
pydantic
transformers
torch
datasets

⚙️ Configuration
src/config.py exposes knobs like:
paths: RAW_PATH, PROCESSED_DIR
split ratios: train/dev/test
text cleaning flags: remove_urls, emojis, numbers, repeated chars
vectorizer: tfidf (ngram_range, max_df, min_df)
model: lr|svm|xgb|distilbert + hyperparameters
class imbalance: class_weight=balanced or sampling strategy
seeds for reproducibility
You can also pass overrides via CLI (see below).

🧹 Prepare Data
# Convert/clean/split data into processed/ train|dev|test
python -m src.preprocess \
  --input data/raw/myntra_reviews.csv \
  --text-col review_text \
  --label-col rating \
  --derive-labels-from-rating true \
  --rating-map "1:neg,2:neg,3:neu,4:pos,5:pos" \
  --lower true --strip-punct true --remove-urls true \
  --lemmatize true --remove-stopwords true \
  --output-dir data/processed
  
🧠 Train (Classic ML)
python -m src.train \
  --model lr \
  --featurizer tfidf \
  --tfidf-max-features 50000 \
  --tfidf-ngram-max 2 \
  --class-weight balanced \
  --train data/processed/train.csv \
  --dev data/processed/dev.csv \
  --save-dir artifacts/classic_lr
Alternate models
# Linear SVM
python -m src.train --model svm --featurizer tfidf --save-dir artifacts/classic_svm

# XGBoost
python -m src.train --model xgb --featurizer tfidf --save-dir artifacts/classic_xgb

🤗 Train (Transformer Baseline)
python -m src.train \
  --model distilbert \
  --pretrained distilbert-base-uncased \
  --max-length 256 \
  --batch-size 16 \
  --lr 2e-5 \
  --epochs 3 \
  --train data/processed/train.csv \
  --dev data/processed/dev.csv \
  --save-dir artifacts/distilbert
  
📊 Evaluate
python -m src.evaluate \
  --model-dir artifacts/classic_svm \
  --test data/processed/test.csv \
  --report-path artifacts/classic_svm/test_report.json \
  --plot-cm artifacts/classic_svm/confusion_matrix.png \
  --plot-pr artifacts/classic_svm/pr_curve.png
Outputs
Classification report (macro & weighted P/R/F1)
Confusion matrix plot
ROC/PR curves (per class, micro/macro)
Error buckets & hardest examples (optional)

🔮 Inference
CLI
python -m src.infer \
  --model-dir artifacts/distilbert \
  --text "Fabric quality is great but the size runs small."
Output
{
  "label": "neutral",
  "probs": {"negative": 0.22, "neutral": 0.51, "positive": 0.27}
}
API (FastAPI)
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
POST /predict
{
  "text": "Great fit and delivery was quick!"
}
Response
{
  "label": "positive",
  "probs": {"negative": 0.03, "neutral": 0.10, "positive": 0.87},
  "model": "artifacts/distilbert"
}
🧪 Tips for Better Performance
Imbalance: try class_weight=balanced, or RandomOverSampler/SMOTE (for classic ML).
Text length: cap at 256–320 tokens for DistilBERT.
Context: include product category/brand if available as additional features.
Noise: strip SKU codes, order IDs, and store names to reduce vocabulary bloat.
Domain shift: fine-tune on Myntra-specific data rather than generic product reviews.

📈 Example Results (placeholder)
Model	Val Macro F1	Test Macro F1
TF-IDF + LR	0.82	0.81
TF-IDF + SVM	0.83	0.82
DistilBERT	0.87	0.86
Replace with your actual numbers after src.evaluate.

🪲 Troubleshooting
CUDA not found: install the correct PyTorch build for your CUDA version, or run on CPU by setting CUDA_VISIBLE_DEVICES="".
Tokenizer mismatch: ensure --pretrained matches the tokenizer used at train time.
Non-English text: switch to multilingual models (e.g., distilbert-base-multilingual-cased) or add language detection + routing.

🔐 Privacy
Strip PII and order info during preprocessing.
Do not store raw user identifiers in logs or artifacts.
If sharing the dataset, share only anonymized text with consent (and within TOS).

🧪 Reproducibility
We fix numpy, torch, and sklearn seeds and save configs in artifacts/*/config.json.
Use --seed 42 (or your choice) consistently across runs.

🤝 Contributing
Fork & create a feature branch
Add/modify unit tests where possible
Format with black/ruff
Open a PR with a concise description and sample outputs

📜 License
Specify your license (e.g., MIT). See LICENSE.

📣 Citation
If you use this repo, consider citing:
@software{myntra_sentiment_2025,
  title  = {Myntra Product Review Sentiment Analysis (NLP + ML)},
  year   = {2025},
  author = {Janhavi Patil},
  url    = {https://github.com/yourname/myntra-sentiment}
}

🗺️ Roadmap
 Better neutral detection via calibration
 Product-aware multi-task head (sentiment + quality/size/fit tags)
 Distillation to a tiny on-device model
 Batch inference script + Dockerfile
 
Quickstart (one-liners)
# Prepare
python -m src.preprocess --input data/raw/myntra_reviews.csv --output-dir data/processed --derive-labels-from-rating true

# Train (SVM)
python -m src.train --model svm --featurizer tfidf --save-dir artifacts/classic_svm

# Evaluate
python -m src.evaluate --model-dir artifacts/classic_svm --test data/processed/test.csv

# Serve
uvicorn src.api:app --port 8000 --reload

https://16fad99e3c3b.ngrok-free.app
