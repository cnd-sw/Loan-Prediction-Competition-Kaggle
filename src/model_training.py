import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

from src.data_preprocessing import load_data, preprocess_data
from src.model_evaluation import evaluate_model, cross_val_performance

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def build_model():
    base_models = [
        ("gb", GradientBoostingClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4
        )),
        ("xgb", XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            n_jobs=-1
        )),
        ("lgb", LGBMClassifier(
            n_estimators=600,
            learning_rate=0.03,
            max_depth=-1,
            num_leaves=40,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1
        ))
    ]

    final_estimator = LogisticRegression(max_iter=500)

    model = StackingClassifier(
        estimators=base_models,
        final_estimator=final_estimator,
        stack_method="predict_proba",
        n_jobs=-1
    )
    return model


def save_model(model, path="models/trained_model.pkl"):
    joblib.dump(model, path)


def main():
    train_df, test_df = load_data("data/train.csv", "data/test.csv")
    train_df, test_df = preprocess_data(train_df, test_df)

    X = train_df.drop("loan_status", axis=1)
    y = train_df["loan_status"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = build_model()
    model.fit(X_train, y_train)

    evaluate_model(model, X_train, y_train, X_val, y_val)
    cross_val_performance(model, X, y)

    y_proba = model.predict_proba(X_val)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()

    y_pred = model.predict(X_val)
    cm = confusion_matrix(y_val, y_pred)

    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha='center', va='center')

    plt.tight_layout()
    plt.show()

    save_model(model)


if __name__ == "__main__":
    main()