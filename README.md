# Kidney Function Prediction (CKD Stages)

Predicts **Chronic Kidney Disease (CKD) stages (1–5)** using machine learning from clinical data.

## Models
- Random Forest
- Gradient Boosting
- SVM  
(+ ensemble voting)

## Features
- GFR → CKD stage conversion
- Leakage-safe preprocessing
- Confusion matrices & model comparison
- Ensemble prediction

## Run
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python main.py
