from src.model_training import main as train_model_main
from src.predict import main as predict_main

if __name__ == "__main__":
    print("Starting Loan Prediction Pipeline")
    train_model_main()
    predict_main()
    print("Pipeline Complete")