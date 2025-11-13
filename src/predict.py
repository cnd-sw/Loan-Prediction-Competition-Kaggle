import pandas as pd
import joblib
from src.data_preprocessing import load_data, preprocess_data


def load_model(path="models/trained_model.pkl"):
    model = joblib.load(path)
    return model


def make_predictions(model, test_df):

    preds = model.predict(test_df)
    return preds


def save_predictions(predictions, output_path="data/predictions.csv"):

    df = pd.DataFrame(predictions, columns=["loan_status_predicted"])
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


def main():
    # load and preprocess data
    train_path = "data/train.csv"
    test_path = "data/test.csv"

    train_df, test_df = load_data(train_path, test_path)
    train_df, test_df = preprocess_data(train_df, test_df)

    # load trained model
    model = load_model("models/trained_model.pkl")

    # ensure the test dataset has no target column
    if "loan_status" in test_df.columns:
        test_df = test_df.drop("loan_status", axis=1)

    # make predictions
    predictions = make_predictions(model, test_df)

    # save to file
    save_predictions(predictions)


if __name__ == "__main__":
    main()