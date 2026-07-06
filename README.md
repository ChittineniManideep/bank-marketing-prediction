# Bank Marketing Campaign Prediction Using Supervised Machine Learning

Predicting whether a bank client will subscribe to a term deposit after a
direct marketing campaign, using supervised classification models.

## Problem Statement

Banks contact customers through marketing campaigns to promote financial
products such as term deposits. Contacting every customer is expensive and
time-consuming, so this project builds a model to predict which customers
are most likely to subscribe — helping the bank target campaigns more
efficiently.

## Dataset

- **Source:** [UCI Machine Learning Repository — Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- 45,211 records, 17 variables (16 predictors + target `y`)
- Collected from direct marketing campaigns of a Portuguese banking institution

> The raw CSV (`bank-full.csv`) is not included in this repo — download it
> from the UCI link above and place it in the project root before running
> the notebook.

## Approach

1. **Exploratory Data Analysis** — shape, dtypes, summary statistics, target
   distribution, outlier checks
2. **Data Preparation**
   - Checked for missing values and duplicates
   - Handled `"unknown"` categories in categorical features
   - One-hot encoded categorical variables
   - 80/20 train-test split (stratified)
   - Feature scaling with `StandardScaler`
   - 5-fold Stratified Cross-Validation for model evaluation
3. **Modeling** — comparison of supervised classification algorithms:
   - Logistic Regression (baseline)
   - Random Forest
   - *(third algorithm — see notebook)*

## Repository Structure

```
├── bank_marketing_prediction.ipynb   # Main analysis notebook
├── requirements.txt                  # Python dependencies
├── .gitignore
└── README.md
```

## Setup

```bash
git clone https://github.com/<your-username>/bank-marketing-prediction.git
cd bank-marketing-prediction
pip install -r requirements.txt
```

Download `bank-full.csv` from the [UCI dataset page](https://archive.ics.uci.edu/dataset/222/bank+marketing)
and place it in the project root, then launch:

```bash
jupyter notebook bank_marketing_prediction.ipynb
```

## Tech Stack

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn
