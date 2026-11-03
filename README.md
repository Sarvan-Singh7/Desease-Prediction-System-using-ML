

Readme · MD
# Multi-Disease Prediction from Symptoms
 
![Python](https://img.shields.io/badge/Python-3.12-blue)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-yellow)
![Status](https://img.shields.io/badge/Status-Educational%20project-lightgrey)
 
A machine learning project that predicts one of **41 diseases** from the symptoms a user reports. It is built step by step in a Jupyter notebook that follows the full ML lifecycle, and it includes a data-leakage investigation that explains why a naive version of this project shows a fake **100% accuracy**.
 
> **Disclaimer:** This is a student project for learning. It is **not** a medical tool and must not be used for diagnosis or treatment decisions. Always consult a qualified doctor.
 
---
 
## Why this project is different
 
Most tutorials on this dataset report near-perfect accuracy. We found out why, and it is not because the model is that good:
 
| Finding | Number |
|---|---|
| Rows in `Training.csv` | 4,920 |
| **Distinct** rows | **304** (about 94% are exact copies) |
| Test rows copied from training in a random 80/20 split | 984 of 984 |
| Rows of `Testing.csv` already inside `Training.csv` | 41 of 41 |
| Real symptom columns | **131** (the file has a second, all-zero `fluid_overload` column) |
 
So any model that memorizes the training rows scores 100%. This project fixes the problem by **removing duplicates before splitting**, and proves with a check that train and test share zero rows.
 
---
 
## Features
 
- Full ML lifecycle in one notebook: collection, cleaning, EDA, feature engineering, training, evaluation, deployment
- Bug investigation that reproduces and explains the 100% accuracy
- Six classifiers compared: Logistic Regression, KNN, Decision Tree, Random Forest, Naive Bayes, SVM
- Honest evaluation: stratified hold-out, 5-fold cross-validation, macro precision, recall and F1, confusion matrix
- **Robustness test** that hides 25% and 50% of a patient's symptoms
- Prediction function that returns the **top 3 diseases** with confidence, a description, precautions and a severity score
- Interactive symptom picker (ipywidgets) for live demos
---
 
## Dataset
 
Source: [Disease Prediction Using Machine Learning](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning) on Kaggle.
 
| File | Purpose |
|---|---|
| `Data/Training.csv` | Main modelling data (binary symptoms + `prognosis`) |
| `Data/dataset.csv` | Raw symptom-name version, used for inspection |
| `Data/Testing.csv` | Used only to demonstrate the leakage (all rows are already in training) |
| `MasterData/symptom_Description.csv` | Disease descriptions |
| `MasterData/symptom_precaution.csv` | Four precautions per disease |
| `MasterData/Symptom_severity.csv` | Severity weight (1 to 7) per symptom |
 
---
 
## Project structure
 
```
AIML-Project/
├── Data/
│   ├── dataset.csv
│   ├── Testing.csv
│   └── Training.csv
├── MasterData/
│   ├── symptom_Description.csv
│   ├── symptom_precaution.csv
│   └── Symptom_severity.csv
├── ML.ipynb              # the full project
├── requirements.txt
├── .gitignore
└── README.md
```
 
---
 
## Pipeline
 
```
Raw data -> Cleaning -> De-duplication -> Stratified split (243 / 61)
        -> Train 6 models -> Evaluate (hold-out, 5-fold CV, symptom dropout)
        -> Select model -> Top-3 prediction + description, precautions, severity
```
 
**Cleaning steps:** assign headers to header-less files, remove stray spaces in names (for example `dischromic _patches`), remove the empty duplicate `fluid_overload` column, and drop duplicate rows.
 
---
 
## Results
 
Evaluated on the 304 distinct records (243 train, 61 test, stratified, seed 42).
 
| Model | Hold-out accuracy | 5-fold CV accuracy |
|---|---|---|
| Logistic Regression | 100% | 100% |
| KNN (k=5) | 100% | 100% |
| Random Forest | 100% | 100% |
| SVM (linear) | 100% | 100% |
| Naive Bayes | 96.7% | 97.7% |
| Decision Tree | **68.9%** | **62.8%** |
 
The Decision Tree shows classic **overfitting**: 100% on training data, far lower on unseen data.
 
**Robustness (accuracy when part of the symptoms is hidden):**
 
| Model | Clean | 25% hidden | 50% hidden |
|---|---|---|---|
| Logistic Regression | 100.0% | 97.2% | 82.3% |
| Random Forest | 100.0% | 97.0% | 80.2% |
| KNN | 100.0% | 94.9% | 66.1% |
| SVM | 100.0% | 94.8% | 63.7% |
| Naive Bayes | 96.7% | 90.4% | 69.2% |
| Decision Tree | 68.9% | 55.4% | 37.5% |
 
Logistic Regression degrades most gracefully, so it is selected as the final model.
 
> **Important:** several models reach 100% on complete input because the dataset looks synthetic: each disease has only 5 to 10 fixed symptom patterns. These scores do **not** reflect real clinical accuracy.
 
---
 
## Getting started
 
**1. Clone the repository**
 
```bash
git clone <your-repository-url>
cd AIML-Project
```
 
**2. Install dependencies**
 
```bash
pip install -r requirements.txt
```
 
**3. Run the notebook**
 
```bash
jupyter notebook ML.ipynb
```
 
Run the cells from top to bottom (**Kernel, then Restart & Run All**). The notebook reads the CSVs using relative paths (`Data/` and `MasterData/`), so open it from the project root folder.
 
---
 
## Example
 
```python
predict_disease(['itching', 'skin_rash', 'nodal_skin_eruptions'])
```
 
```
Symptoms       : itching, skin_rash, nodal_skin_eruptions
Severity score : 8
-------------------------------------------------------
1 . Fungal infection - 53.1 %
2 . Drug Reaction - 7.8 %
3 . Acne - 3.7 %
-------------------------------------------------------
About       : <short description of the disease>
Precautions :
   - <precaution 1>
   - ...
Note: this is a student project, not medical advice.
```
 
Vague input (for example `headache, vomiting, nausea`) gives low confidence on purpose, which is why the function returns a ranked list rather than a single label.
 
---
 
## Limitations
 
- The dataset appears synthetic, so high scores reflect its simple structure, not real patients.
- Only 5 to 10 distinct patterns per disease, so the hold-out set is small (61 rows). Cross-validation is the more reliable estimate.
- No age, sex, symptom duration or medical history.
- The robustness test models missing symptoms only, not wrongly reported ones.
- Probabilities are not calibrated and should be read as scores.
- No validation on external or real clinical data.
## Future work
 
- Conversational chatbot front end that asks follow-up questions
- Streamlit web app
- Explainability with SHAP
- Validation on real clinical data and probability calibration
---
 
## Tech stack
 
Python 3.12, pandas, NumPy, Matplotlib, scikit-learn, ipywidgets, Jupyter
 
## Authors
 
- Sarvan Singh
- Arsh Wadhwa
- Vedant Jain
- Vivek Sharma
Department of Computer Science & Engineering, Chitkara University Institute of Engineering and Technology, Chitkara University, Punjab, India.