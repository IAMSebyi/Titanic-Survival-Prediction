import pandas as pd
from src.data.load_data import DataLoader
from src.data.preprocess import Preprocessor
from xgboost import XGBClassifier
import joblib  # For loading the saved model


class ModelPredictor:
    def __init__(self, model_path: str):
        self.model_path = model_path

    def load_model(self):
        return joblib.load(self.model_path)

    def predict(self, test_df: pd.DataFrame, threshold=0.5) -> pd.DataFrame:
        # Load the model
        model = self.load_model()

        # Separate PassengerId and features
        passenger_ids = test_df['PassengerId']
        X_test = test_df.drop(['PassengerId'], axis=1)

        # Get probabilities
        probabilities = model.predict_proba(X_test)[:, 1]

        # Make predictions using thresholding
        predictions = (probabilities >= threshold).astype(int)

        # Create a DataFrame with PassengerId and predictions
        result_df = pd.DataFrame({
            'PassengerId': passenger_ids,
            'Survived': predictions
        })

        return result_df

    def save_predictions(self, result_df: pd.DataFrame, output_path: str):
        result_df.to_csv(output_path, index=False)
