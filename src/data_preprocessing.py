import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path

def preprocess_data():
    BASE_DIR = Path(__file__).resolve().parent.parent
    data_path = BASE_DIR / "data" / "crop_data.csv"
    data = pd.read_csv(data_path)
    X = data.drop("Yield", axis=1)
    y = data["Yield"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data()
    print("✅ Data preprocessing completed successfully")
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))
