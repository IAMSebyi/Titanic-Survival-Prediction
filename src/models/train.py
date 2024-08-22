from xgboost import XGBClassifier
import joblib


class ModelTrainer:
    def __init__(self, model=None):
        self.model = model

    def train(self, X_train, y_train):
        if self.model is None:
            raise ValueError("Model must be defined before training.")
        self.model.fit(X_train, y_train)

    def save_model(self, path):
        import joblib
        joblib.dump(self.model, path)
