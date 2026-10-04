# Customer Churn Prediction ML Pipeline

An end-to-end machine learning project that predicts customer churn, explains each
prediction with SHAP, and routes high-risk customers into automated business
actions through an n8n workflow.

```
Customer data -> FastAPI -> XGBoost model -> SHAP explanation -> recommended action -> n8n -> email
```

## Overview

Customer churn (a customer canceling their subscription) is costly to fix after the
fact and expensive to prevent for everyone equally. This project predicts churn risk
for individual telecom customers and converts that prediction into a tiered business
action, so that retention effort (and cost) is spent where it is likely to matter most.

**Pipeline:**
1. Train an XGBoost classifier on the [Kaggle Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
2. Serve predictions through a FastAPI endpoint.
3. Explain each prediction with SHAP (which factors pushed this customer toward churn).
4. Map the churn probability to a recommended action (discount, support message, usage tips, or no action).
5. Forward the result to an n8n workflow, which triggers a notification email.

## Tech stack

- **Model:** XGBoost (`XGBClassifier`)
- **Explainability:** SHAP (`TreeExplainer`)
- **Preprocessing:** scikit-learn (`ColumnTransformer`, `OneHotEncoder`, `SimpleImputer`)
- **API:** FastAPI + Pydantic
- **Automation:** n8n (webhook -> email)
- **Testing:** pytest

## Project structure

```
.
├── churn_model.py       # ChurnPredictor: feature engineering, training, SHAP, persistence
├── train_model.py       # CLI script: trains the model and writes evaluation artifacts
├── main.py               # FastAPI service exposing /predict
├── requirements.txt
├── tests/
│   ├── test_actions.py   # Unit tests for the risk-tier / action rules
│   └── test_api.py       # Tests for request validation and the /predict contract
└── artifacts/             # Created by train_model.py: metrics.json, test_predictions.csv
```

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
(7,043 customers, 21 columns, ~26.5% churn rate). Not included in this repo — download
it from Kaggle and place the CSV in the project root before training.

18 raw features are used (demographics, account info, and subscribed services);
`customerID` is dropped and two engineered features are added: `AvgMonthlySpend` and
`TotalServices` (see **Feature engineering** below).

## Setup

```bash
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Download the dataset from Kaggle and place it in the project root (or anywhere,
and point `--data` at it).

## Training

```bash
python train_model.py --data telco_customer_churn.csv --model-out churn_model.pkl --output-dir artifacts
```

This splits the data (80/20, stratified on churn), runs 5-fold cross-validation on the
training portion, fits the final model, evaluates it on the held-out test set, and writes:

- `churn_model.pkl` — the trained model, preprocessor, and metadata
- `artifacts/metrics.json` — all reported metrics in JSON form
- `artifacts/test_predictions.csv` — true labels and predicted probabilities for the test set

## Results

Measured on the held-out test set (20% of the data, never used for training or tuning):

| Metric | Value |
|---|---|
| ROC-AUC (test) | 0.841 |
| PR-AUC (test) | 0.655 |
| ROC-AUC (5-fold CV, training data) | 0.847 ± 0.009 |
| PR-AUC (5-fold CV, training data) | 0.666 ± 0.024 |

The cross-validation step is run on the training split only, before the model ever
sees the test set, so the CV score is a check on how stable the model is across
different data splits — not a number tuned against the test set.

**Precision / recall at different thresholds** (illustrative — see **Limitations**):

| Threshold | Precision | Recall |
|---|---|---|
| 0.30 | 0.532 | 0.762 |
| 0.40 | 0.576 | 0.639 |
| 0.50 | 0.651 | 0.508 |
| 0.60 | 0.721 | 0.388 |
| 0.70 | 0.791 | 0.243 |
| 0.75 | 0.810 | 0.171 |

Confusion matrix at threshold 0.50:

|  | Predicted: No churn | Predicted: Churn |
|---|---|---|
| **Actual: No churn** | 933 | 102 |
| **Actual: Churn** | 184 | 190 |

## Feature engineering

- **`AvgMonthlySpend`** — `TotalCharges / tenure`, falling back to `MonthlyCharges` for
  customers with `tenure == 0` to avoid a division-by-zero / missing-value artifact.
- **`TotalServices`** — count of subscribed add-on services (`OnlineSecurity`,
  `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`).
- Binary Yes/No columns are one-hot encoded with `drop="if_binary"` (one column
  instead of two perfectly collinear ones), which keeps the SHAP attribution for each
  feature clean instead of splitting it across redundant columns.
- All preprocessing (imputation, encoding) is fit **only on the training split**, then
  applied to validation/test data — this avoids data leakage.

## SHAP explanations

Each prediction includes the top 3 features (by absolute SHAP value) that pushed it
toward or away from churn. One-hot encoded columns are summed back to their original
feature (e.g. `Contract_Month-to-month` and `Contract_One year` are reported as a
single `Contract` contribution) so the explanation reflects the customer's actual
answer rather than internal encoding detail.

SHAP values are reported in **log-odds space**, not probability points — a value of
`0.42` means a positive push toward churn, not "+42% probability."

## Recommended actions

Each prediction is mapped to a business action based on the predicted risk and a
customer-value tag supplied by the caller:

| Risk level | Customer value | Action |
|---|---|---|
| High (≥ 0.60) | High | Send 10% discount offer |
| High (≥ 0.60) | Medium / Low | Send support message |
| Medium (≥ 0.40) | any | Send personalized usage tips |
| Low (< 0.40) | any | No action |

## Limitations

- **The risk thresholds (0.60 / 0.40) are hand-set, illustrative estimates — not
  tuned.** A production deployment should derive them from actual campaign costs and
  measured retention/success rates, selected on a validation split and reported on a
  separate test split, rather than picked by hand.
- **`customer_value` is a caller-supplied input**, not a model output. It is not
  learned from data in this project.
- SHAP explanations describe what the model learned from this dataset, not a causal
  claim about why a customer churns.
- The dataset is a well-known public dataset; results here demonstrate the pipeline,
  not performance on a live production population.

## API

Start the server:

```bash
uvicorn main:app --reload --port 8000
```

Interactive docs: `http://127.0.0.1:8000/docs`

### `POST /predict`

<details>
<summary>Example request</summary>

```json
{
  "customer_id": "C-1001",
  "tenure": 3,
  "MonthlyCharges": 89.5,
  "TotalCharges": 268.5,
  "SeniorCitizen": 0,
  "Contract": "Month-to-month",
  "InternetService": "Fiber optic",
  "TechSupport": "No",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "PaymentMethod": "Electronic check",
  "PaperlessBilling": "Yes",
  "Partner": "No",
  "Dependents": "No",
  "MultipleLines": "No",
  "PhoneService": "Yes",
  "customer_value": "High"
}
```
</details>

<details>
<summary>Example response</summary>

```json
{
  "churn_probability": 0.819,
  "risk_level": "High",
  "customer_value": "High",
  "recommended_action": "Send 10% Discount Offer",
  "top_churn_factors": [
    {"feature": "Contract", "value": "Month-to-month", "impact_direction": "Increases Churn", "shap_value": 0.91},
    {"feature": "Tenure", "value": 3, "impact_direction": "Increases Churn", "shap_value": 0.44},
    {"feature": "Internet Service", "value": "Fiber optic", "impact_direction": "Increases Churn", "shap_value": 0.21}
  ],
  "customer_id": "C-1001"
}
```
</details>

Request fields are validated against the dataset's known categories (`Contract`,
`InternetService`, `PaymentMethod`, etc.) — an unrecognized value returns `422`
instead of silently producing a prediction.

## n8n integration

On every `/predict` call, the result is sent as a background task (so the API
response is never delayed by it) to an n8n webhook:

```
N8N_WEBHOOK_URL=http://localhost:5678/webhook/customer-churn
```

The n8n workflow listens on that path and, in this project, sends a notification
email based on the result. If n8n is unreachable, the prediction still succeeds —
the delivery failure is only logged.

### Example notification email

<img width="1281" height="671" alt="Ekran görüntüsü 2026-10-04 191831" src="https://github.com/user-attachments/assets/2291664f-be1d-4b62-9871-dc112a0c9045" />


*Triggered automatically by the n8n workflow when a customer is flagged as high risk.*

## Testing

```bash
pytest
```

- `tests/test_actions.py` — checks the risk-tier / action-mapping rules at their boundaries.
- `tests/test_api.py` — checks request validation (rejects unknown categories) and the
  `/predict` response contract.

## Possible next steps

- Derive the risk thresholds from an explicit cost/benefit model instead of hand-setting them.
- Calibrate predicted probabilities (`CalibratedClassifierCV`) before using them for business decisions.
- Add a Dockerfile for containerized deployment.

## Author

İrem Mersin
[GitHub](https://github.com/iremmersin) · [LinkedIn](https://linkedin.com/in/irem-mersin-643b51343/)
