# Ordinal Logistic Regression for Salary Prediction

**Author**: Hon Wa Ng\
**Date**: October 2024  

## Overview

This project implements Ordinal Logistic Regression for salary prediction based on the 2022 Kaggle Salary Dataset. The dataset has been cleaned and preprocessed to extract meaningful features, perform feature engineering, and apply machine learning techniques to classify salaries into ordered categories.

The repository includes data cleaning, exploratory analysis, feature engineering, model implementation, hyperparameter tuning, and evaluation.


## Objectives

- Preprocess and clean the Kaggle salary dataset.
- Perform exploratory feature analysis and selection.
- Implement Ordinal Logistic Regression to predict salary categories.
- Tune model hyperparameters using grid search.
- Evaluate the model using k-fold cross-validation and bias-variance trade-off analysis.

## Repository Structure
```bash
Ordinal-Logistic-Regression-Salary-Prediction/
│── data/                                # Dataset storage
│   ├── clean_kaggle_data_2022.csv        # Processed salary dataset
│
│── doc/                                 # Documentation files
│   ├── project_question.pdf              # Problem statement & questions
│   ├── project_report.pdf                # Final project report & analysis
│
│── src/                                 # Source code
│   ├── salary_prediction_pipeline.py     # Main script for preprocessing, training & evaluation
│
│── LICENSE                              # License file
│── requirements.txt                      # Dependencies for running the project

```

---

## Installation & Usage

### 1. Clone the Repository
```
git clone https://github.com/Edwardnhw/Ordinal-Logistic-Regression-Salary-Prediction.git
cd Ordinal-Logistic-Regression-Salary-Prediction


```

### 2. Install Dependencies
Ensure you have Python installed (>=3.7), then run:
```
pip install -r requirements.txt

```

---
## How to Run the Project
Execute the main pipeline script:

```
python src/salary_prediction_pipeline.py


```
The script will:

- Load and clean the dataset.
- Engineer relevant features.
- Train and evaluate an Ordinal Logistic Regression model.
- Perform model tuning and bias-variance trade-off analysis.


---
## Methods Used

1. Data Preprocessing
- Removed unnecessary columns.
- Imputed missing values.
- Encoded categorical variables.
2. Feature Engineering
- Derived new features (e.g., high education level, number of tools used).
- Applied Lasso regression for feature selection.
3. Model Implementation
- Ordinal Logistic Regression trained on salary categories.
- Evaluated using accuracy, precision, recall, and F1-score.
- Used k-fold cross-validation for performance assessment.
4. Model Tuning & Bias-Variance Trade-Off
- Applied grid search for hyperparameter optimization.
- Visualized trade-offs between bias and variance.


---

## Results & Analysis

- The final Ordinal Logistic Regression model achieved X% accuracy (replace with actual value).
- Feature selection using Lasso regression improved model interpretability.
- Bias-variance analysis revealed optimal regularization strength (C = X) for generalization.

For detailed results, refer to project_report.pdf in the doc/ folder.
---
## License
This project is licensed under the MIT License.



