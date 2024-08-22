import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


class ModelEvaluator:
    def __init__(self, model):
        self.model = model

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.DataFrame, threshold=0.5):
        # Get probabilities
        y_prob = self.model.predict_proba(X_test)[:, 1]

        # Make predictions
        y_pred = (y_prob >= threshold).astype(int)

        # Evaluate predictions and print results
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))
