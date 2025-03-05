# %%
import pandas as pd
import numpy as np
from random import *
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score

from sklearn.pipeline import Pipeline

import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

cur_dir = os.getcwd()

import warnings
warnings.filterwarnings("ignore")

# %%
salaries = pd.read_csv("data/clean_kaggle_data_2022.csv", low_memory = False, encoding = 'latin2')
pd.set_option('display.max_columns', None)
salaries.shape

# %%
salaries.head()

# %% [markdown]
# # Question 1: Data Cleaning

# %% [markdown]
# - Drop the second row since it is about the question details
# - Drop other information too if they are irrelevant to the task

# %%
def CleanData(df):
    # Drop the first row (question details)
    df.drop(df.index[0], inplace=True)

    #TODO: Drop other information too here if they are irrelevant to the task
    df.drop(columns=['Duration (in seconds)'], inplace=True)
    
    return df

salaries = CleanData(salaries)
salaries.shape

# %%
# Feature Engineering Code **

# 1. High Education Level
salaries['high_education_level'] = salaries['Q8'].apply(lambda x: 1 if x in ['Master’s degree', 'Doctoral degree','Professional doctorate'] else 0)

# 2. Total Automated Machine Learning Tools Used
tool_columns = [col for col in salaries.columns if col.startswith('Q38_')]
salaries['total_tools_used'] = salaries[tool_columns].apply(lambda row: row.count(), axis=1)

# Print to verify the new features
print("Newly engineered features:")
print(salaries[['high_education_level','total_tools_used']].head())

# %% [markdown]
# ###**Impute missing values (single column responses)**
# 
# 1. Identify columns with single column responses
# 
# 2. Address missing values in those columns

# %% [markdown]
# ###**Encode categorical features (single column responses)**

# %%
#TODO: Encode categorical features in the single column responses

def ImputingAndEncodingSingleColFeatures(df):
    # Identify columns with single responses and explicitly include 'Q29_Encoded' (target)
    single_col_names = [col for col in df.columns if 'Q' in col and '_' not in col]
    
    single_col_names.append('Q29_Encoded')  # Ensure target is included

    # Impute missing values in single-column responses
    for col in single_col_names:
        if df[col].isnull().sum() > 0:
            # Fill categorical columns with the most frequent value (mode)
            df[col].fillna(df[col].mode()[0], inplace=True)

    # Encode categorical columns in single-column responses
    label_encoder = LabelEncoder()
    for col in single_col_names:
        if df[col].dtype == 'object':
            df[col] = label_encoder.fit_transform(df[col])
    
    return df

# Apply the function to the salaries DataFrame
salaries = ImputingAndEncodingSingleColFeatures(salaries)

# %% [markdown]
# ###**Handling categorical features (multi column responses)**

# %%

def HandleMultiColResponsesAsPattern(df):
    # List of specific columns to exclude from being converted or dropped
    exclude_columns = ['Q29_Encoded']
    
    # Step 1: Identify and sort multi-column response groups by question number prefix (e.g., 'Q11_', 'Q44_')
    # Include only prefixes that have numbers after the initial letter(s), so 'Q29_buckets' and 'Q29_Encoded' will not be affected
    multi_col_prefixes = sorted(
        {
            col.split('_')[0] 
            for col in df.columns if '_' in col and col.split('_')[0][1:].isdigit()
        }, 
        key=lambda x: int(x[1:])
    )  # Sort based on numeric part
    
    # Step 2: Process each group of multi-column responses
    for prefix in multi_col_prefixes:
        # Identify columns associated with the current question prefix
        multi_col_names = [col for col in df.columns if col.startswith(prefix + '_')]
        
        # Convert non-empty string responses to 1 (selected) and fill missing values with 0,
        # but skip columns in the exclude_columns list
        for col in multi_col_names:
            if col not in exclude_columns:  # Check if column is in exclude list
                df[col] = df[col].apply(lambda x: 1 if pd.notnull(x) and x != 'unknown' else 0)
        
        # Concatenate binary values into a single string pattern for each row
        df[prefix + '_selection_pattern'] = df[multi_col_names].astype(str).agg(''.join, axis=1)
        
        # Drop the original individual columns if only the concatenated pattern is needed,
        # but keep columns that are in the exclude_columns list
        columns_to_drop = [col for col in multi_col_names if col not in exclude_columns]
        df.drop(columns=columns_to_drop, inplace=True)

    
    return df

# Apply the function to the salaries DataFrame
salaries = HandleMultiColResponsesAsPattern(salaries)

# Print sample to verify the pattern columns
print("Sample of DataFrame after handling multi-column responses as selection patterns:")
print(salaries.head())


# %% [markdown]
# ###**Drop the target variable(s) and get the target variable**

# %%
# Make sure there are no missing values remaining in the dataset
assert salaries.isnull().values.sum() == 0, \
    "There are still {} missing values remaining in salaries!".format(
        salaries.isnull().values.sum()
    )

# %%
# Step 1: Identify the target variable
target = salaries['Q29_Encoded']  

# Step 2: Drop the target variable and any columns that start with 'Q29_' from the feature set
salaries_features = salaries.drop(columns=[col for col in salaries.columns if col.startswith('Q29')])

# Step 3: Ensure the target variables are not in the feature set
for col in salaries_features.columns:
    assert 'Q29' not in col, \
        f"Target-related column ({col}) is still in the dataset"

# %% [markdown]
# # Question 2: Exploratory Feature Analysis
# 
# - From Question 2 - Question 4, you should **NOT** peek at the test labels in any form!
# - Assume that you do not know the target values (Q29_Encoded) on the test set

# %% [markdown]
# ## Split data into training and test sets

# %%
# Step 1: Define the target and features, Q29 related columns are derived from the target so dropping as well
target = salaries['Q29_Encoded']
features = salaries.drop(columns=[col for col in salaries.columns if col.startswith('Q29')])

# Step 2: Split the data into training and test sets
train_df, test_df, y_train, y_test = train_test_split(
                                        features, 
                                        target, 
                                        test_size=0.2, 
                                        random_state=42
                                        )

# Now, X_train is a DataFrame with column names
feature_names = features.columns

X_train = train_df.values
X_test = test_df.values
y_train = y_train.values
y_test = y_test.values

# Verifying the split
print("Training set shape:", train_df.shape)
print("Test set shape:", test_df.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# %% [markdown]
# ## **Feature Engineering/Generation**
# 
# - Create new feature(s) using existing features!

# %%
salaries.head()

# %% [markdown]
# ## **Feature Selection**
# 
# - Select the features based on the criteria of your choice

# %%

def lasso_feature_selection(X_train, y_train, alpha_values=[0.01, 0.1, 1, 10]):
    """
    Perform feature selection using Lasso regression.
    
    Parameters:
    X_train (np.array or pd.DataFrame): Training data features.
    y_train (np.array or pd.Series): Training data target.
    alpha_values (list): List of alpha values to try for Lasso regularization.
    
    Returns:
    selected_features (list): List of selected feature names.
    best_model (model): Trained Lasso model with the best alpha.
    """
    
    # Define the Lasso model in a pipeline with a StandardScaler
    pipeline = Pipeline([
        ('scaler', StandardScaler()),      # Standardize the features
        ('lasso', Lasso(max_iter=10000))   # Lasso model with max_iter for convergence
    ])
    
    # Use GridSearchCV to find the best alpha parameter
    param_grid = {'lasso__alpha': alpha_values}
    grid_search = GridSearchCV(pipeline, param_grid, scoring='neg_mean_squared_error', cv=5)
    grid_search.fit(X_train, y_train)
    
    # Get the best model from GridSearchCV
    best_model = grid_search.best_estimator_.named_steps['lasso']
    
    # Extract feature importances (coefficients) from the Lasso model
    importance = best_model.coef_
    
    # Get the names of selected features (non-zero coefficients)
    selected_features = [feature for feature, coef in zip(train_df.columns, importance) if coef != 0]
    
    # Print the selected features and their importance
    print("Selected features based on Lasso:")
    for feature, coef in zip(selected_features, importance):
        print(f"{feature}: {coef}")
    
    return selected_features, best_model

# Apply the feature selection function on training data
selected_features, best_lasso_model = lasso_feature_selection(X_train, y_train)

# Print the selected features
print("Top selected features:")
print(selected_features)

# %%
train_df_selected = train_df[selected_features]

# %% [markdown]
# ## **Visualization**
# 
# - Make visualization to better understand your data

# %%
def visualize_top_features(selected_features, best_lasso_model, top_n=10):
    """
    Visualize the top N selected features based on their importance values (coefficients).
    
    Parameters:
    selected_features (list): List of selected feature names from the Lasso model.
    best_lasso_model (Lasso model): Trained Lasso model with non-zero coefficients.
    top_n (int): Number of top features to visualize. Default is 10.
    """
    # Get the coefficients of the selected features
    importance = best_lasso_model.coef_
    
    # Filter the selected features and their coefficients where the coefficient is non-zero
    feature_importances = pd.Series(
        [coef for coef, feature in zip(importance, selected_features) if coef != 0],
        index=[feature for coef, feature in zip(importance, selected_features) if coef != 0]
    )
    
    # Sort by absolute importance and select the top N features
    top_features = feature_importances.abs().sort_values(ascending=False).head(top_n)
    
    # Plot the top N features
    plt.figure(figsize=(10, 6))
    top_features.plot(kind='barh', color='skyblue')
    plt.title(f"Top {top_n} Feature Importances from Lasso Regression")
    plt.gca().invert_yaxis()  # Highest importance at the top
    plt.xlabel("Coefficient Magnitude (Importance)")
    plt.ylabel("Feature")
    plt.show()

# Visualize the top 10 selected features
visualize_top_features(selected_features, best_lasso_model, top_n=10)

# %% [markdown]
# ## **Apply the same feature engineering/selection to test data**

# %%

# Leave selected features
test_df_selected = test_df[selected_features]

# %%
test_df.head()

# %% [markdown]
# # Question 3: Model Implementation

# %% [markdown]
# ## Implement Ordinal Logistic Regression Model

# %%

class OrdinalLogisticRegression:
    def __init__(self, max_iter=100, C=1.0, scale_features=True):
        self.C = C
        self.max_iter = max_iter
        self.scale_features = scale_features
        self.classes_ = []
        self.models_ = []
        self.scaler_ = StandardScaler() if scale_features else None

    def fit(self, X, y):
        if self.scale_features:
            X = self.scaler_.fit_transform(X)

        self.classes_ = sorted(np.unique(y))
        self.models_ = []

        for i, c in enumerate(self.classes_[:-1]):
            y_i = (y > c).astype(int)
            model = LogisticRegression(max_iter=self.max_iter, C=self.C)
            model.fit(X, y_i)
            self.models_.append(model)

        return self

    def predict_proba(self, X):
        if self.scale_features:
            X = self.scaler_.transform(X)

        binary_probabilities = np.empty((X.shape[0], len(self.models_), 2), dtype=float)

        for i, model in enumerate(self.models_):
            binary_probabilities[:, i] = model.predict_proba(X)

        k = len(self.classes_)
        proba = np.empty((X.shape[0], k), dtype=float)
        proba[:, 0] = binary_probabilities[:, 0, 0]

        for i in range(1, k - 1):
            proba[:, i] = binary_probabilities[:, i, 0] - binary_probabilities[:, i - 1, 0]

        proba[:, -1] = binary_probabilities[:, k - 2, 1]
        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def get_params(self, deep=True):
        return {'max_iter': self.max_iter, 'C': self.C, 'scale_features': self.scale_features}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


# Function to evaluate bias-variance trade-off based on different values of C
def analyze_bias_variance_tradeoff(X_train, y_train, C_values=[0.01, 0.1, 1, 10, 100]):
    """
    Analyze model performance based on bias-variance trade-off for different values of C.
    
    Parameters:
    - X_train, y_train: Training data features and target.
    - C_values: List of C values to evaluate.
    """
    results = []
    for C in C_values:
        model = OrdinalLogisticRegression(max_iter=1000, C=C, scale_features=True)
        # Using cross-validation to assess performance
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        results.append((C, avg_score, std_score))
        print(f"C={C}: Mean Accuracy={avg_score:.4f}, Std Dev={std_score:.4f}")

    # Determine the best C in terms of bias-variance trade-off (high accuracy, low variance)
    best_C = max(results, key=lambda x: x[1])[0]
    print(f"\nBest C for bias-variance trade-off: {best_C}")
    return results, best_C

# Example of using the analysis function
results, best_C = analyze_bias_variance_tradeoff(X_train, y_train)

# Notes on Scaling:
# Scaling is enabled here because logistic regression can be sensitive to the scale of the features,
# especially with regularization. StandardScaler is applied to normalize features to a mean of 0 and std of 1.




# %% [markdown]
# ## Run k-fold cross validation
# 
# - Report the average/variance of accuracies across folds

# %%


def perform_k_fold_cross_validation(model, X_train, y_train, k=10):
    """
    Perform k-fold cross-validation on the training data and print accuracies for each fold.
    
    Parameters:
    - model: The model instance to evaluate.
    - X_train: Training data features.
    - y_train: Training data labels.
    - k: Number of folds for cross-validation (default is 10).
    
    Returns:
    - mean_accuracy: Mean accuracy across k folds.
    - variance_accuracy: Variance of accuracy across k folds.
    """
    # Perform k-fold cross-validation with accuracy as the scoring metric
    accuracies = cross_val_score(model, X_train, y_train, cv=k, scoring='accuracy')
    
    # Calculate the average and variance of accuracies
    mean_accuracy = np.mean(accuracies)
    variance_accuracy = np.var(accuracies)
    
    # Print accuracies for each fold and the overall mean and variance
    print(f"Accuracies for each fold: {accuracies}")
    print(f"Average accuracy across {k} folds: {mean_accuracy}")
    print(f"Variance of accuracy across {k} folds: {variance_accuracy}")
    
    return mean_accuracy, variance_accuracy

# Example usage to compare different models with varying hyperparameters
hyperparameters = [0.01, 0.1, 1, 10, 100] # Different values of C to compare
results = []

for C_value in hyperparameters:
    print(f"\nEvaluating OrdinalLogisticRegression with C={C_value}")
    model = OrdinalLogisticRegression(max_iter=100, C=C_value)
    mean_accuracy, variance_accuracy = perform_k_fold_cross_validation(model, X_train, y_train, k=10)
    results.append((C_value, mean_accuracy, variance_accuracy))

# Print summary of results
print("\nSummary of results:")
for C_value, mean_accuracy, variance_accuracy in results:
    print(f"C={C_value}: Mean Accuracy={mean_accuracy:.4f}, Variance={variance_accuracy:.4f}")


# %% [markdown]
# ## Bias-variance trade-off
# 
# [bias_variance_decomp.py](https://github.com/rasbt/mlxtend/blob/master/mlxtend/evaluate/bias_variance_decomp.py)
# 
# Below is a function you can use to compute the bias and variance of your ordinal logistic regression model. Using this function, analyze the ordinal logistic regression model performance based on bias-variance trade-off.

# %%
### NOTE: You don't need to change anything in this code block! ###

def _draw_bootstrap_sample(rng, X, y):
    sample_indices = np.arange(X.shape[0])
    bootstrap_indices = rng.choice(
        sample_indices, size=sample_indices.shape[0], replace=True
    )
    return X[bootstrap_indices], y[bootstrap_indices]

def bias_variance_decomp(
    estimator,
    X_train,
    y_train,
    X_test,
    y_test,
    num_rounds=10,
    random_seed=0
):
    """
    estimator : object
        A classifier or regressor object or class implementing both a
        `fit` and `predict` method similar to the scikit-learn API.

    X_train : array-like, shape=(num_examples, num_features)
        A training dataset for drawing the bootstrap samples to carry
        out the bias-variance decomposition.

    y_train : array-like, shape=(num_examples)
        Targets (class labels, continuous values in case of regression)
        associated with the `X_train` examples.

    X_test : array-like, shape=(num_examples, num_features)
        The test dataset for computing the average loss, bias,
        and variance.

    y_test : array-like, shape=(num_examples)
        Targets (class labels, continuous values in case of regression)
        associated with the `X_test` examples.

    num_rounds : int (default=10)
        Number of bootstrap rounds (sampling from the training set)
        for performing the bias-variance decomposition. Each bootstrap
        sample has the same size as the original training set.

    random_seed : int (default=0)
        Random seed for the bootstrap sampling used for the
        bias-variance decomposition.

    Returns
    ----------
    avg_bias, avg_var : returns the average bias, and average bias (all floats),
                        where the average is computed over the data points
                        in the test set.

    """
    loss = "mse"

    for ary in (X_train, y_train, X_test, y_test):
        assert type(ary) == np.ndarray, \
            "X_train, y_train, X_test, y_test have to be NumPy array. \
            If e.g., X_train is a pandas DataFrame, convert it to NumPy array \
            via X_train=X_train.values."

    rng = np.random.RandomState(random_seed)

    # All the predictions across different rounds
    all_pred = np.zeros((num_rounds, y_test.shape[0]), dtype=np.float64)

    for i in range(num_rounds):
        # Randomly sample training data
        X_boot, y_boot = _draw_bootstrap_sample(rng, X_train, y_train)

        # Fit the model using the randomly sampled data
        pred = estimator.fit(X_boot, y_boot).predict(X_test)
        all_pred[i] = pred

    # Mean prediction across runs using different dataset for each data point
    main_predictions = np.mean(all_pred, axis=0)

    # Average bias across different rounds
    avg_bias = np.sum((main_predictions - y_test) ** 2) / y_test.size

    # Average variance across different rounds
    avg_var = np.sum((main_predictions - all_pred) ** 2) / all_pred.size

    return avg_bias, avg_var

# %%
# Usage example
model = OrdinalLogisticRegression()
avg_bias, avg_var = \
    bias_variance_decomp(model, X_train, y_train, X_test, y_test, num_rounds=10, random_seed=0)

# %%
print(avg_bias, avg_var)

# %%
# Define hyperparameters and results storage
hyperparameters = [0.001, 0.01, 0.1, 1, 10, 100]
results = []

# Evaluate bias, variance, and total error for each C value
for C_value in hyperparameters:
    print(f"\nEvaluating OrdinalLogisticRegression with C={C_value}")
    
    # Create a new model instance for each C
    model = OrdinalLogisticRegression(max_iter=100, C=C_value)
    
    # Perform bias-variance decomposition
    avg_bias, avg_var = bias_variance_decomp(
        estimator=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        num_rounds=10,
        random_seed=0
    )
    
    # Calculate total error as the sum of bias and variance
    total_error = avg_bias + avg_var
    
    # Store results
    results.append({
        "C_value": C_value,
        "bias": avg_bias,
        "variance": avg_var,
        "total_error": total_error
    })

# Prepare data for plotting
C_values = [result["C_value"] for result in results]
bias = [result["bias"] for result in results]
variance = [result["variance"] for result in results]
total_error = [result["total_error"] for result in results]

# Plot Bias, Variance, and Total Error
plt.figure(figsize=(10, 6))
plt.plot(C_values, bias, label="Bias", marker='o')
plt.plot(C_values, variance, label="Variance", marker='o')
plt.plot(C_values, total_error, label="Total Error", marker='o')
plt.xscale("log")  # Log scale for better visualization across wide C range
plt.xlabel("C (Regularization Strength)")
plt.ylabel("Error")
plt.title("Bias-Variance-Total Error vs. Regularization (C)")
plt.legend()
plt.show()


# %%

total_errors = [bias + variance for bias, variance in zip(bias, variance)]

# Create subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 15))
fig.suptitle("Bias, Variance, and Total Error vs. Regularization Strength (C)")

# Plot Bias
axs[0].plot(C_values, bias, marker='o', color='blue', label="Bias")
axs[0].set_xscale('log')
axs[0].set_xlabel("C (Regularization Strength)")
axs[0].set_ylabel("Bias")
axs[0].set_title("Bias vs. Regularization (C)")
axs[0].legend()

# Plot Variance
axs[1].plot(C_values, variance, marker='o', color='orange', label="Variance")
axs[1].set_xscale('log')
axs[1].set_xlabel("C (Regularization Strength)")
axs[1].set_ylabel("Variance")
axs[1].set_title("Variance vs. Regularization (C)")
axs[1].legend()

# Plot Total Error
axs[2].plot(C_values, total_errors, marker='o', color='green', label="Total Error")
axs[2].set_xscale('log')
axs[2].set_xlabel("C (Regularization Strength)")
axs[2].set_ylabel("Total Error")
axs[2].set_title("Total Error vs. Regularization (C)")
axs[2].legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit the main title
plt.show()


# %%
#TODO: Discuss bias-variance trade-off of your ordinal logistic regression model
'''In order to analyze the bias-variance trade-off, one hyperparameter that can directly impact it in an Ordinal Logistic Regression model is C, the inverse of regularization strength.
A higher C value reduces regularization, potentially lowering bias but increasing variance as the model fits more closely to the training data, risking overfitting. 
Conversely, a lower C value increases regularization, reducing variance but possibly introducing more bias by generalizing too much. Using grid search, I found the optimal C value to be 0.01, achieving an accuracy score of approximately 0.3858.
The plot shows that the total error reduces as C = 0.1, and variance remains low across this range, which supports the observation that the model generalizes well at 𝐶 = 0.1 without high fluctuation in prediction across folds.
'''

# %% [markdown]
# # Question 4: Model Tuning

# %% [markdown]
# - What are the hyperparameters we can potentially set for our ordinal logistic regression model?
# - Which hyperparameters seem to be worthwhile to tune?

# %% [markdown]
# ## Grid Search
# 
# - Grid search will take time to complete - but if it does not finish in a few hours, you're probably trying too many combinations
# - A recommended approach is to try a small number of combinations with a wide range first (for continuous value hyperparameters)! Then gradually increase the points that seem to be near optimal

# %%

def grid_search_ordinal_logistic(X_train, y_train):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  
        ('ordinal_logreg', OrdinalLogisticRegression())
    ])

    param_grid = {
        'ordinal_logreg__C': [0.001, 0.01, 0.1, 1, 10],
        'ordinal_logreg__max_iter': [100, 500, 1000]
    }

    # Use 'f1_weighted' for F1 scoring that accounts for all class weights
    grid_search = GridSearchCV(pipeline, param_grid, scoring='f1_weighted', cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_.named_steps['ordinal_logreg']
    print("Best Parameters:", grid_search.best_params_)
    print("Best F1 Score:", grid_search.best_score_)
    
    return best_model, grid_search

# Example usage:
best_model, grid_search_results = grid_search_ordinal_logistic(X_train, y_train)


# %% [markdown]
# ## Visualize the feature importance of your model

# %%
# Instantiate the model with the best hyperparameters found by grid search
model = OrdinalLogisticRegression(max_iter=100, C=10, scale_features=True)  # Using C=0.01 and max_iter=100 from grid search results
model.fit(X_train, y_train)

# %%
def visualize_feature_importance_ordinal_logistic(model, feature_names):
    """
    Visualizes the feature importance (coefficients) for an Ordinal Logistic Regression model.
    
    Parameters:
    - model: The trained Ordinal Logistic Regression model.
    - feature_names: List of feature names (columns) from the original dataset.
    """
    # Extract coefficients from each binary logistic regression model in the ordinal logistic model
    coefficients = np.mean([m.coef_.flatten() for m in model.models_], axis=0)

    # Create a Series to hold coefficients with feature names for easy plotting
    feature_importances = pd.Series(coefficients, index=feature_names)

    # Plot top features by absolute importance
    feature_importances = feature_importances.abs().sort_values(ascending=False)
    top_features = feature_importances.head(10)
    
    plt.figure(figsize=(10, 6))
    top_features.plot(kind='bar')
    plt.title("Top 10 Feature Importances in Ordinal Logistic Regression")
    plt.xlabel("Feature")
    plt.ylabel("Absolute Coefficient Value")
    plt.xticks(rotation=45)
    plt.show()

# Usage example
visualize_feature_importance_ordinal_logistic(best_model, feature_names=features.columns)

# %% [markdown]
# # Question 5: Testing

# %%

#TODO: Using the best-performing model, evaluate the model performance both on the training set and test set

# Function to evaluate and print performance metrics
def evaluate_model_performance(model, X_train, y_train, X_test, y_test):
    """
    Evaluate the model performance on both training and test sets and print the metrics.
    """
    # Predictions on training and test sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics for training set
    print("Performance on Training set:")
    print(f"  Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
    print(f"  Precision: {precision_score(y_train, y_train_pred, average='weighted'):.4f}")
    print(f"  Recall: {recall_score(y_train, y_train_pred, average='weighted'):.4f}")
    print(f"  F1 Score: {f1_score(y_train, y_train_pred, average='weighted'):.4f}")
    print("------------------------------------------------------")
    
    # Calculate metrics for test set
    print("Performance on Test set:")
    print(f"  Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"  Precision: {precision_score(y_test, y_test_pred, average='weighted'):.4f}")
    print(f"  Recall: {recall_score(y_test, y_test_pred, average='weighted'):.4f}")
    print(f"  F1 Score: {f1_score(y_test, y_test_pred, average='weighted'):.4f}")
    print("------------------------------------------------------")

#TODO: Plot the distribution of true target variable values and their predictions on both the training set and test set
def plot_combined_true_vs_predicted_distributions(y_train, y_train_pred, y_test, y_test_pred):
    """
    Plot the combined distribution of true and predicted target values for both training and test sets.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("True vs Predicted Distributions of Target Values")
    
    # Training set combined distribution
    sns.histplot(y_train, bins=15, kde=False, color='blue', label='True', ax=axes[0])
    sns.histplot(y_train_pred, bins=15, kde=False, color='orange', label='Predicted', ax=axes[0], alpha=0.6)
    axes[0].set_title("Training Set Distribution")
    axes[0].set_xlabel("Target Value")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    # Test set combined distribution
    sns.histplot(y_test, bins=15, kde=False, color='blue', label='True', ax=axes[1])
    sns.histplot(y_test_pred, bins=15, kde=False, color='orange', label='Predicted', ax=axes[1], alpha=0.6)
    axes[1].set_title("Test Set Distribution")
    axes[1].set_xlabel("Target Value")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

# Evaluate and visualize using the best model found in grid search
#TODO: Using the best-performing model, evaluate the model performance both on the training set and test set
best_model.fit(X_train, y_train)  # Refit on full training set if needed

# Evaluate performance
evaluate_model_performance(best_model, X_train, y_train, X_test, y_test)

# Get predictions for plotting
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

# Plot combined distributions
plot_combined_true_vs_predicted_distributions(y_train, y_train_pred, y_test, y_test_pred)



