from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle

from data_preprocessing import preprocess_data

def train_model():
    X_train, X_test, y_train, y_test = preprocess_data()
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("✅ Model trained successfully")
    print("Mean Squared Error:", mse)
    print("R2 Score:", r2)
    with open("src/crop_yield_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("📁 Model saved as crop_yield_model.pkl")
if __name__ == "__main__":
    train_model()
