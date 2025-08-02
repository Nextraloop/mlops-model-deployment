import mlflow
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from src.data_preprocessing import load_and_preprocess
from sklearn.metrics import mean_squared_error

def train_and_log_models():
    X_train, X_test, y_train, y_test = load_and_preprocess('data/raw/housing.csv')

    best_rmse = float('inf')
    best_model = None

    for model_name, model in {"LinearRegression": LinearRegression(), "DecisionTree": DecisionTreeRegressor()}.items():
        with mlflow.start_run(run_name=model_name):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            rmse = mean_squared_error(y_test, preds, squared=False)
            mlflow.log_param("model", model_name)
            mlflow.log_metric("rmse", rmse)
            mlflow.sklearn.log_model(model, "model")
            print(f"Model: {model_name}, RMSE: {rmse}")
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model

    joblib.dump(best_model, "models/best_model.pkl")
    print("Best model saved to models/best_model.pkl")

if __name__ == "__main__":
    train_and_log_models()
