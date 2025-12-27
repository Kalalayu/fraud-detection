import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Load data
df = pd.read_csv(r"C:\Users\Dell\Pictures\fraud-detection\data\processed\processed_fraud_data.csv")  

X = df.drop(columns=["class"])
y = df["class"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

def train_models(X_train, y_train):
    """
    Trains Logistic Regression and Random Forest classifiers with hyperparameter tuning
    and saves the best models as .pkl files.
    Returns the best estimators.
    """
    print("Training Logistic Regression...")
    lr = LogisticRegression(class_weight="balanced", max_iter=500)
    lr_param_grid = {"C": [0.01, 0.1, 1, 10]}

    lr_grid = GridSearchCV(
        lr,
        param_grid=lr_param_grid,
        scoring="roc_auc",
        cv=5
    )
    lr_grid.fit(X_train, y_train)

    print("Training Random Forest...")
    rf = RandomForestClassifier(random_state=42)
    rf_param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "class_weight": ["balanced"]
    }

    rf_grid = GridSearchCV(
        rf,
        param_grid=rf_param_grid,
        scoring="roc_auc",
        cv=5,
        # n_jobs=-1
    )
    rf_grid.fit(X_train, y_train)

    # Save the best models
    joblib.dump(lr_grid.best_estimator_, "models/logistic_regression.pkl")
    joblib.dump(rf_grid.best_estimator_, "models/random_forest.pkl")

    print("Training complete.")
    return lr_grid.best_estimator_, rf_grid.best_estimator_

# Run training when executed as a script
if __name__ == "__main__":
    print("Starting model training...")
    best_lr, best_rf = train_models(X_train, y_train)
