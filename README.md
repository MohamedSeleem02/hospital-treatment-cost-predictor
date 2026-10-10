# Hospital Treatment Cost Predictor

A machine learning project that predicts the **treatment cost (USD)** of a hospital patient from admission details, with a small **Streamlit app** for entering a patient and getting a predicted cost.

> **Type:** Academic project (NTI Machine Learning course), done individually by Mohamed Ahmed Farag AbdAlAal.

## Problem
Given a patient's age, gender, department, diagnosis, severity level, length of stay, wait time, insurance type, readmission flag and discharge outcome, predict `Treatment_Cost_USD`.

## Data
`Hospital_Operations_Dataset.csv`: 100,000 rows x 15 columns, including missing values and invalid values (for example ages above 100).

## What I did
- **Cleaning:** turned out-of-range values (age, length of stay, wait time) into nulls using fixed bounds.
- **Split first:** 80% train / 20% test (`random_state=42`), so no information from the test set leaked into preprocessing.
- **Preprocessing (fitted on the training set only):** median / most-frequent imputation, ordinal encoding for severity, label encoding for gender, one-hot encoding for the other categories, standard scaling. Result: 30 features.
- **Models trained:** Linear Regression, Decision Tree, SVR (RBF, trained on a 10,000-row sample) and Random Forest.
- **Final model:** Random Forest Regressor (100 trees, max depth 10), saved with all preprocessing objects in `model_file.joblib`.

## Results (Random Forest)
| Set | MAE (USD) | R2 |
|---|---|---|
| Training (80,000 rows) | 1,310.12 | 0.8981 |
| **Unseen test (20,000 rows)** | **1,426.32** | **0.8626** |

Training R2 of the other models (for reference, training data only): Linear Regression 0.6859, Decision Tree 0.8868, SVR 0.9734 (on its 10,000-row sample).

## Files
| File | Purpose |
|---|---|
| `train_PR.ipynb` | Cleaning, EDA, preprocessing, training, saving the model |
| `test_PR.ipynb` | Loads the saved model and evaluates it on the test set |
| `GUI.ipynb` | Runs the Streamlit app in Google Colab |
| `app.py` | Streamlit app |
| `model_file.joblib` | Saved model + preprocessing objects |
| `requirements.txt` | Python dependencies |
| `run_app.bat` | Starts the app on Windows |

## Run the app locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
The model was saved with **scikit-learn 1.6.1**; use the same version to load it.

## Tools
Python, pandas, NumPy, scikit-learn, Matplotlib, seaborn, Streamlit, joblib.
