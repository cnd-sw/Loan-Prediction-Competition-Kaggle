import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import matplotlib.pyplot as plt

from src.data_preprocessing import load_data, preprocess_data
from src.model_evaluation import evaluate_model, cross_val_performance
from sklearn.metrics import roc_curve, auc


def train_model(X_train, y_train):

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def save_model(model, path="models/trained_model.pkl"):

    joblib.dump(model, path)
    print(f"Model saved to {path}")


def main():
    # load and preprocess data
    train_path = "data/train.csv"
    test_path = "data/test.csv"

    train_df, test_df = load_data(train_path, test_path)
    train_df, test_df = preprocess_data(train_df, test_df)

    # separate features and target
    X = train_df.drop('loan_status', axis=1)
    y = train_df['loan_status']

    # split into train/test
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # train model
    model = train_model(X_train, y_train)

    # evaluate
    evaluate_model(model, X_train, y_train, X_val, y_val)

    # cross-validation
    cross_val_performance(model, X, y)

    

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_val, model.predict_proba(X_val)[:, 1])
    roc_auc = auc(fpr, tpr)

    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # save model
    save_model(model)


if __name__ == "__main__":
    main()