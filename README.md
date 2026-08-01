# Fake Job Posting Detector — Streamlit App

An explainable UI for the model trained in your notebook. Paste in a job title +
description and get a REAL/FAKE gauge, colour-coded word-by-word contribution,
ranked top contributing words, and a flip/robustness analysis — the same
explainability layer as the notebook, wrapped in a web UI.

## Folder structure

```
fake_job_detector/
├── app.py                     # Main Streamlit app (UI + orchestration)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── .streamlit/
│   └── config.toml            # Dark theme config
├── models/
│   ├── best_fake_job_model.pkl   # <-- put your downloaded model here
│   └── PUT_MODEL_HERE.txt
└── utils/
    └── explain.py              # Leave-one-word-out explainability engine
```

## 1. Get your model file

From the training notebook, Step 14 saves `best_fake_job_model.pkl`
(a pickle containing `model_name`, the trained `model`, and the fitted
TF-IDF `vectorizer`). Download that file from Kaggle's Output tab and
place it at:

```
fake_job_detector/models/best_fake_job_model.pkl
```

> If your best model turned out to be one of the deep learning models
> (LSTM/BiLSTM/CNN/CNN-BiLSTM) instead of an ML model, the notebook saves
> `best_fake_job_model.keras` + `tokenizer.json` instead — that needs a
> slightly different loader. Let me know and I'll give you the DL version
> of `app.py`.

## 2. Install dependencies

```bash
cd fake_job_detector
pip install -r requirements.txt
```

If your saved model is **not** XGBoost or LightGBM, you can remove those
two lines from `requirements.txt` to keep the install lighter.

## 3. Run locally

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

## 4. Deploy for free (Streamlit Community Cloud)

1. Push this whole `fake_job_detector/` folder to a GitHub repo (include
   the `models/best_fake_job_model.pkl` file — GitHub allows files up to
   100MB; if your pickle is bigger, use [Git LFS](https://git-lfs.com) or
   load the model from cloud storage instead).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, click **New app**, pick the repo, and set the main file path
   to `app.py`.
3. Deploy — you'll get a public URL like
   `https://your-app-name.streamlit.app`.

## How the explainability works

`utils/explain.py` contains the same leave-one-word-out logic as the
notebook: for every word in the input, the model is asked to predict again
with that one word removed. The shift in REAL-probability is that word's
"impact" — no SHAP/LIME dependency needed, works with any model that
exposes `predict_proba` (or `decision_function` as a fallback).
