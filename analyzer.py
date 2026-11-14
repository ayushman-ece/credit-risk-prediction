
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


FEATURES = [
    "annual_income",
    "credit_score",
    "loan_amount",
    "employment_years",
    "existing_loans"
]


def load_loan_data(file_path):
    df = pd.read_csv(file_path)

    required = ["applicant_id"] + FEATURES + ["loan_approved"]

    if not all(col in df.columns for col in required):
        raise ValueError("Dataset is missing required columns.")

    if df.empty or df[required].isnull().any().any():
        raise ValueError("Dataset is empty or contains missing values.")

    if df["applicant_id"].duplicated().any():
        raise ValueError("Duplicate applicant IDs found.")

    numeric_cols = required
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="raise")

    if not np.isfinite(df[numeric_cols].to_numpy()).all():
        raise ValueError("Dataset contains invalid numeric values.")

    if (df[FEATURES] < 0).any().any():
        raise ValueError("Financial and employment values cannot be negative.")

    if not df["loan_approved"].isin([0, 1]).all():
        raise ValueError("loan_approved must contain only 0 or 1.")

    if df["loan_approved"].nunique() != 2:
        raise ValueError("Dataset must contain both approval classes.")

    print(f"Loaded {len(df)} loan applications.")
    return df


def analyze_loan_data(df):
    print("\n========== DATA ANALYSIS ==========")

    print(f"Total applications: {len(df)}")
    print(f"Average annual income: {df['annual_income'].mean():.2f}")
    print(f"Average credit score: {df['credit_score'].mean():.2f}")

    print("\nLoan approval distribution:")
    print(df["loan_approved"].value_counts().rename(
        index={0: "Not Approved", 1: "Approved"}
    ))


def prepare_data(df):
    X = df[FEATURES]
    y = df["loan_approved"]

    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    print("\nRandom Forest model trained successfully.")
    return model


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    print("\n========== MODEL EVALUATION ==========")
    print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, predictions, zero_division=0):.4f}")
    print(f"F1 Score:  {f1_score(y_test, predictions, zero_division=0):.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test, predictions,
        labels=[0, 1],
        target_names=["Not Approved", "Approved"],
        zero_division=0
    ))

    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

    plt.figure(figsize=(6, 5))
    plt.imshow(matrix)
    plt.title("Loan Approval Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.xticks([0, 1], ["Not Approved", "Approved"])
    plt.yticks([0, 1], ["Not Approved", "Approved"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, matrix[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()

    print("\nConfusion matrix saved.")

    return predictions


def predict_loan(model, income, credit_score, loan_amount,
                 employment_years, existing_loans):

    applicant = pd.DataFrame([{
        "annual_income": income,
        "credit_score": credit_score,
        "loan_amount": loan_amount,
        "employment_years": employment_years,
        "existing_loans": existing_loans
    }])

    if not np.isfinite(applicant.to_numpy()).all():
        raise ValueError("Input values must be finite numbers.")

    if (applicant < 0).any().any():
        raise ValueError("Input values cannot be negative.")

    prediction = model.predict(applicant)[0]
    probabilities = model.predict_proba(applicant)[0]
    approval_probability = probabilities[
        list(model.classes_).index(1)
    ]

    print("\n========== NEW APPLICATION ==========")
    print("Predicted class:",
          "Approved" if prediction == 1 else "Not Approved")
    print(f"Model approval probability: {approval_probability:.2%}")

    return prediction, approval_probability


def export_results(X_test, y_test, predictions):
    results = X_test.copy()
    results["actual_status"] = y_test.to_numpy()
    results["predicted_status"] = predictions
    results.to_csv("loan_predictions.csv", index=False)

    print("\nPrediction results exported to loan_predictions.csv")
