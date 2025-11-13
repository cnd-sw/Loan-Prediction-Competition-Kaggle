# Loan Prediction Competition (Kaggle)

This project replicates and extends a Kaggle Loan Prediction challenge.  
It builds an end-to-end pipeline — from data preprocessing to model training, evaluation, and prediction — using Python and scikit-learn.

---

## Project Structure

Loan Prediction Kaggle/
├── data/
│   ├── .gitkeep                  # placeholder (real datasets not tracked)
│   ├── train.csv                 # training data (local only)
│   └── test.csv                  # test data (local only)
│
├── models/
│   ├── .gitkeep
│   └── trained_model.pkl         # saved model (auto-generated)
│
├── src/
│   ├── data_preprocessing.py     # handles missing values, encoding, cleaning
│   ├── model_training.py         # trains model and plots ROC-AUC
│   ├── model_evaluation.py       # metrics, cross-validation
│   └── predict.py                # loads model, runs inference
│
├── run.py                        # executes the full pipeline
├── requirements.txt              # all dependencies
└── README.md                     # project documentation

---

## Features
- Clean, modular pipeline (preprocessing → training → evaluation → prediction)
- Random Forest classifier baseline
- ROC-AUC visualization
- Cross-validation for model robustness
- Easy model persistence with `joblib`
- `.gitignore` configured to exclude sensitive and large files (datasets, models)

---

## Tech Stack
- Python 3.10+
- pandas, numpy
- scikit-learn
- matplotlib
- joblib

---

## ⚙️ Setup & Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/cnd-sw/Loan-Prediction-Competition-Kaggle.git
   cd Loan-Prediction-Competition-Kaggle

	2.	Create and activate a virtual environment

python3 -m venv venv
source venv/bin/activate


	3.	Install dependencies

pip install -r requirements.txt


	4.	Add your dataset
Place your train.csv and test.csv inside the data/ folder.
	5.	Run the entire pipeline

python run.py

Outputs:
	•	Metrics (accuracy, ROC-AUC, confusion matrix)
	•	Saved model in models/trained_model.pkl
	•	ROC-AUC plot
	•	Predictions saved to data/predictions.csv

⸻

Example Output


Accuracy: 1.0
ROC AUC: 1.0

Accuracy: 0.9495
ROC AUC: 0.847


⸻

Next Steps
	•	Experiment with other models (XGBoost, LightGBM, CatBoost)
	•	Perform hyperparameter tuning using GridSearchCV
	•	Add feature importance visualization
	•	Integrate model explainability (SHAP / LIME)

⸻

Note

Data files are intentionally excluded from version control for privacy and size reasons.
Use your own dataset under data/.

⸻


