import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve,
)

# ==============================
# Load trained models
# ==============================
lr = joblib.load("models/logistic_regression.pkl")
rf = joblib.load("models/random_forest.pkl")

# ==============================
# Load or define your test data
# ==============================
# Make sure X_test and y_test are preprocessed exactly like in training
# Example: replace this with your actual test dataset loading
import pandas as pd
df = pd.read_csv(r"C:\Users\Dell\Pictures\fraud-detection\data\processed\processed_fraud_data.csv") 
X = df.drop(columns=["class"])
y = df["class"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
# ==============================
# Predictions and probabilities
# ==============================
y_pred_lr = lr.predict(X_test)
y_proba_lr = lr.predict_proba(X_test)[:, 1]

y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

# ==============================
# Compute ROC-AUC
# ==============================
auc_lr = roc_auc_score(y_test, y_proba_lr)
auc_rf = roc_auc_score(y_test, y_proba_rf)

print(f"Logistic Regression ROC-AUC: {auc_lr:.4f}")
print(f"Random Forest ROC-AUC:      {auc_rf:.4f}")

# ==============================
# Classification reports
# ==============================
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, y_pred_lr))

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

# ==============================
# Confusion matrices
# ==============================
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_lr)
plt.title("Logistic Regression Confusion Matrix")
plt.show()

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_rf)
plt.title("Random Forest Confusion Matrix")
plt.show()

# ==============================
# ROC Curve Comparison
# ==============================
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)

plt.figure(figsize=(7, 5))
plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {auc_lr:.3f})")
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {auc_rf:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(True)
plt.show()

