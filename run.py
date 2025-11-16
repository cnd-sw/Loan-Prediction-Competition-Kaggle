#for just training the model
# from src.model_training import main as train_model

# if __name__ == "__main__":
#     train_model()


#for training and predicting in one run
from src.model_training import main as train_model_main
from src.predict import main as predict_main

if __name__ == "__main__":
    print("Starting Loan Prediction Pipeline")
    train_model_main()
    predict_main()
    print("Pipeline Complete")
    