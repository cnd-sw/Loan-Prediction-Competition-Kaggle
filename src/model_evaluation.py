import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score


def evaluate_model(model, X_train, y_train, X_test, y_test):

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    print("train Metrics")
    print("accuracy:", round(accuracy_score(y_train, y_pred_train), 4))
    print("roc auc:", round(roc_auc_score(y_train, y_pred_train), 4))
    print()

    print("test Metrics")
    print("accuracy:", round(accuracy_score(y_test, y_pred_test), 4))
    print("roc auc:", round(roc_auc_score(y_test, y_pred_test), 4))
    print()

    print("confusion Matrix (Test)")
    print(confusion_matrix(y_test, y_pred_test))
    print()

    print("classification Report (Test)")
    print(classification_report(y_test, y_pred_test))


def cross_val_performance(model, X, y, cv=5):

    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"cross-validation Accuracy ({cv}-fold): {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    return scores


if __name__ == "__main__":
    print("this module provides model evaluation utilities. run from model_training.py.")