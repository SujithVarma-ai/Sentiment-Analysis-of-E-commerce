# SPL Project (Streamlit)

This project converts the original notebook-style script (`special_project.py`) into a runnable Streamlit app.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run streamlit_app.py
```

## Data format

Upload a dataset containing:

- a text column (commonly `content`)
- a label column (commonly `score`, e.g. 1–5)

The app supports:

- quick EDA charts (label distribution, review length distribution, word cloud)
- training a TF‑IDF + Logistic Regression model
- single and batch predictions
- export/import of the trained model as a `.pkl`

