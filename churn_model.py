import os
import joblib
import numpy as np
import pandas as pd
import shap
import sklearn
import xgboost as xgb
import logging
import warnings
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class ChurnPredictor:

    # Thresholds are hand-set, illustrative estimates. They are not optimized.
    # In a real deployment they should come from campaign costs and measured
    # retention rates and be validated on data separate from the test set.
    SEED = 42
    TEST_SIZE = 0.20
    CV_FOLDS = 5
    HIGH_RISK_THRESHOLD = 0.60
    MEDIUM_RISK_THRESHOLD = 0.40

    XGB_PARAMS = {
        "objective": "binary:logistic",
        "max_depth": 4,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "random_state": SEED,
    }

    MODEL_FEATURES = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "SeniorCitizen",
        "Contract",
        "InternetService",
        "TechSupport",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "StreamingTV",
        "StreamingMovies",
        "PaymentMethod",
        "PaperlessBilling",
        "Partner",
        "Dependents",
        "MultipleLines",
        "PhoneService"
    ]

    SERVICE_COLUMNS = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    NUMERIC_FEATURES = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "SeniorCitizen",
        "AvgMonthlySpend",
        "TotalServices"
    ]

    CATEGORICAL_FEATURES = [
        "Contract",
        "InternetService",
        "TechSupport",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "StreamingTV",
        "StreamingMovies",
        "PaymentMethod",
        "PaperlessBilling",
        "Partner",
        "Dependents",
        "MultipleLines",
        "PhoneService"
    ]

    ENGINEERED_FEATURES = [
        "AvgMonthlySpend",
        "TotalServices"
    ]

    FEATURE_LABELS = {
        "tenure": "Tenure",
        "MonthlyCharges": "Monthly Charges",
        "TotalCharges": "Total Charges",
        "SeniorCitizen": "Senior Citizen",
        "AvgMonthlySpend": "Average Monthly Spend",
        "TotalServices": "Total Services",
        "Contract": "Contract",
        "InternetService": "Internet Service",
        "TechSupport": "Tech Support",
        "OnlineSecurity": "Online Security",
        "OnlineBackup": "Online Backup",
        "DeviceProtection": "Device Protection",
        "StreamingTV": "Streaming TV",
        "StreamingMovies": "Streaming Movies",
        "PaymentMethod": "Payment Method",
        "PaperlessBilling": "Paperless Billing",
        "Partner": "Partner",
        "Dependents": "Dependents",
        "MultipleLines": "Multiple Lines",
        "PhoneService": "Phone Service",
    }

    def __init__(self):

        self.model = self.build_model()

        self.preprocessor = None
        self.explainer = None
        self.feature_names = None
        self.cv_metrics = None

    @classmethod
    def build_model(cls):

        return xgb.XGBClassifier(**cls.XGB_PARAMS)

    def clean_data(self, df):

        df = df.copy()

        # The imputer must learn the median only from the training data.
        if "TotalCharges" in df.columns:

            df["TotalCharges"] = pd.to_numeric(
                df["TotalCharges"],
                errors="coerce"
            )

        return df

    # Feature Engineering (stateless and row-wise, so it cannot leak)
    def create_features(self, df):

        df = self.clean_data(df)

        # tenure == 0 (or a missing TotalCharges) falls back to MonthlyCharges
        # instead of a NaN that the imputer would fill with a median.
        tenure = df["tenure"].replace(0, np.nan)

        df["AvgMonthlySpend"] = (
            df["TotalCharges"] / tenure
        ).fillna(df["MonthlyCharges"])

        df["TotalServices"] = (
            df[self.SERVICE_COLUMNS] == "Yes"
        ).sum(axis=1)

        return df

    def prepare_raw_features(self, df):

        df = self.create_features(df)

        feature_columns = self.MODEL_FEATURES + self.ENGINEERED_FEATURES

        return df[feature_columns].copy()

    def build_preprocessor(self):

        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median"))
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                        # Yes/No columns become one column instead of two
                        # perfectly collinear ones (cleaner SHAP attribution).
                        drop="if_binary"
                    )
                )
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, self.NUMERIC_FEATURES),
                ("categorical", categorical_pipeline, self.CATEGORICAL_FEATURES)
            ],
            remainder="drop"
        )

        preprocessor.set_output(transform="pandas")

        return preprocessor

    @staticmethod
    def _apply_preprocessor(preprocessor, raw_features):

        return preprocessor.transform(raw_features).astype(float)

    def _fit_pipeline(self, raw_features, y):

        preprocessor = self.build_preprocessor()
        processed = preprocessor.fit_transform(raw_features).astype(float)

        model = self.build_model()
        model.fit(processed, y)

        return preprocessor, model

    def _require_fitted(self):

        if (
            self.model is None
            or self.preprocessor is None
            or self.explainer is None
        ):

            raise RuntimeError("Model is not loaded. Train or load the model first.")
            
    def transform(self, df):

        if self.preprocessor is None:

            raise RuntimeError("Preprocessor has not been fitted. Train or load the model first.")

        raw = self.prepare_raw_features(df)

        return self._apply_preprocessor(self.preprocessor, raw)

    def predict_proba(self, df):
        
        self._require_fitted()

        return self.model.predict_proba(self.transform(df))[:, 1]

    def split_data(self, X, y):

        return train_test_split(
            X,
            y,
            test_size=self.TEST_SIZE,
            random_state=self.SEED,
            stratify=y
        )

    # Cross-validation on the training data only (preprocessor re-fitted per fold)
    def cross_validate_metrics(self, raw_features, y):

        y = np.asarray(y, dtype=int)

        folds = StratifiedKFold(n_splits=self.CV_FOLDS, shuffle=True, random_state=self.SEED)

        roc_scores = []
        pr_scores = []

        for train_idx, valid_idx in folds.split(raw_features, y):

            preprocessor, model = self._fit_pipeline(
                raw_features.iloc[train_idx],
                y[train_idx]
            )

            proba = model.predict_proba(
                self._apply_preprocessor(
                    preprocessor,
                    raw_features.iloc[valid_idx]
                )
            )[:, 1]

            roc_scores.append(roc_auc_score(y[valid_idx], proba))
            pr_scores.append(average_precision_score(y[valid_idx], proba))

        return {
            "folds": self.CV_FOLDS,
            "roc_auc_mean": float(np.mean(roc_scores)),
            "roc_auc_std": float(np.std(roc_scores)),
            "pr_auc_mean": float(np.mean(pr_scores)),
            "pr_auc_std": float(np.std(pr_scores)),
        }

    def train(self, X, y):

        # Split before fitting any preprocessing to prevent data leakage.
        X_train, X_test, y_train, y_test = self.split_data(X, y)

        X_train_eng = self.prepare_raw_features(X_train)
        y_train_arr = np.asarray(y_train, dtype=int)

        self.cv_metrics = self.cross_validate_metrics(X_train_eng, y_train_arr)

        self.preprocessor, self.model = self._fit_pipeline(X_train_eng, y_train_arr)

        self.feature_names = self.preprocessor.get_feature_names_out().tolist()

        self.explainer = shap.TreeExplainer(self.model)

        return X_test, y_test

    def evaluate(self, X_test, y_test):

        self._require_fitted()

        y_true = np.asarray(y_test, dtype=int)
        y_prob = self.predict_proba(X_test)

        roc_auc = float(roc_auc_score(y_true, y_prob))
        pr_auc = float(average_precision_score(y_true, y_prob))
        cv = self.cv_metrics
        
        thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.75]

        for threshold in thresholds:

            y_threshold = (y_prob >= threshold).astype(int)

            precision = precision_score(y_true, y_threshold, zero_division=0)
            recall = recall_score(y_true, y_threshold, zero_division=0)

        # Default classification threshold, for reference.
        y_pred = (y_prob >= 0.50).astype(int)

        cm = confusion_matrix(y_true, y_pred)

        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))

        return {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "cv": cv,
            "precision": precision,
            "recall": recall,
            "confusion_matrix": cm.tolist()
        }

    def group_shap(self, shap_row):
        """Sum SHAP values of one-hot columns back to their original feature.
        Values are in log-odds space, not probability points.
        """
        grouped = {}

        for name, value in zip(self.feature_names, shap_row):

            name = name.split("__", 1)[-1]

            base = next(
                (
                    column
                    for column in self.CATEGORICAL_FEATURES
                    if name.startswith(column + "_")
                ),
                name
            )

            grouped[base] = grouped.get(base, 0.0) + float(value)

        return sorted(
            grouped.items(),
            key=lambda item: abs(item[1]),
            reverse=True
        )

    @staticmethod
    def _display_value(value):

        if hasattr(value, "item"):
            value = value.item()

        if value is None or (not isinstance(value, str) and pd.isna(value)):
            return None

        if isinstance(value, float):
            return round(value, 2)

        return value

    def predict_with_explanation(self, input_df):

        self._require_fitted()

        input_df = input_df.copy()

        # customer_value is a business input. It is not used by the ML model.
        customer_value = (
            input_df["customer_value"].iloc[0]
            if "customer_value" in input_df.columns
            else "Medium"
        )

        model_input = input_df.drop(
            columns=["customer_value"],
            errors="ignore"
        )

        raw = self.prepare_raw_features(model_input)

        X_processed = self._apply_preprocessor(self.preprocessor, raw)

        churn_probability = float(
            self.model.predict_proba(X_processed)[0][1]
        )

        shap_output = self.explainer(X_processed)

        shap_values = shap_output.values

        if len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]

        top_factors = []

        for feature, impact in self.group_shap(shap_values[0])[:3]:

            top_factors.append(
                {
                    "feature": self.FEATURE_LABELS.get(feature, feature),
                    "value": self._display_value(raw.iloc[0][feature]),
                    "impact_direction": (
                        "Increases Churn"
                        if impact > 0
                        else "Decreases Churn"
                    ),
                    # log-odds contribution, not probability points
                    "shap_value": round(float(impact), 3)
                }
            )

        return {
            "churn_probability": round(churn_probability, 3),
            "risk_level": self.risk_level(churn_probability),
            "customer_value": customer_value,
            "recommended_action": self.determine_action(
                churn_probability,
                customer_value
            ),
            "top_churn_factors": top_factors
        }

    @classmethod
    def determine_action(cls, churn_probability, customer_value):

        if churn_probability >= cls.HIGH_RISK_THRESHOLD:

            if customer_value == "High":
                return "Send 10% Discount Offer"

            return "Send Support Message"

        elif churn_probability >= cls.MEDIUM_RISK_THRESHOLD:

            return "Send Personalized Usage Tips"

        return "No action"

    @classmethod
    def risk_level(cls, churn_probability):

        if churn_probability >= cls.HIGH_RISK_THRESHOLD:
            return "High"

        if churn_probability >= cls.MEDIUM_RISK_THRESHOLD:
            return "Medium"

        return "Low"

    @staticmethod
    def _library_versions():

        return {
            "xgboost": xgb.__version__,
            "scikit-learn": sklearn.__version__,
            "shap": shap.__version__,
        }

    def save_model(self, path="churn_model.pkl"):

        if self.model is None or self.preprocessor is None:

            raise RuntimeError("Cannot save an untrained model.")

        model_data = {
            "model": self.model,
            "preprocessor": self.preprocessor,
            "feature_names": self.feature_names,
            "cv_metrics": self.cv_metrics,
            "versions": self._library_versions(),
        }

        joblib.dump(model_data, path)

        logging.info(f"Model saved: {path}")

    def load_model(self, path="churn_model.pkl"):

        if not os.path.exists(path):

            raise FileNotFoundError(f"Trained model not found: {path}")

        model_data = joblib.load(path)

        self.model = model_data["model"]
        self.preprocessor = model_data["preprocessor"]
        self.feature_names = model_data["feature_names"]
        self.cv_metrics = model_data.get("cv_metrics")

        current_versions = self._library_versions()

        for library, saved_version in model_data.get("versions", {}).items():

            if current_versions.get(library) != saved_version:

                warnings.warn(
                    f"{library} {saved_version} was used for training, "
                    f"but {current_versions.get(library)} is installed."
                )

        self.explainer = shap.TreeExplainer(self.model)

        logging.info(f"Model loaded: {path}")