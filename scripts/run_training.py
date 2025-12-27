from src.utils import load_data
from src.models import train_models

if __name__ == "__main__":
    df = load_data("processed_fraud_data.csv")
    X, y = df.drop(columns=["class"]), df["class"]
    best_lr, best_rf = train_models(X, y)
    print("Training finished.")
