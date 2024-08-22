from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier
import time


class ModelSelector:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.best_model = None
        self.best_params = None

    def select_model(self, search_method='grid'):
        print("Started Model Selection")

        # Define the model
        xgb_model = XGBClassifier(random_state=42)

        # Define the parameter grid including regularization parameters
        param_grid = {
            'n_estimators': [500],
            # Reduce rounds to avoid overfitting, and focus on optimal tree construction
            'max_depth': [3, 4],  # Keep depth shallow to prevent overfitting
            'learning_rate': [0.01, 0.05],
            # Standard learning rates, with a smaller initial value to avoid missing minima
            'colsample_bytree': [0.7, 0.8],  # Subsampling features to reduce overfitting
            'subsample': [0.8, 0.9],  # Subsampling rows; higher values might work better with smaller datasets
            'min_child_weight': [1, 2],  # Lower values allow for more complex models
            'gamma': [0, 0.1],  # Small regularization on splits, might help with noise
            'reg_alpha': [0.0001, 0.001],  # L1 regularization; small values to gently enforce sparsity
            'reg_lambda': [1.1, 1.2]  # L2 regularization; moderate values to avoid large coefficients
        }

        # Select search method: GridSearch or RandomizedSearch
        if search_method == 'grid':
            search = GridSearchCV(xgb_model, param_grid, cv=10, scoring='accuracy', verbose=1, n_jobs=-1)
        elif search_method == 'random':
            search = RandomizedSearchCV(xgb_model, param_grid, cv=10, scoring='accuracy', n_iter=100, verbose=1,
                                        n_jobs=-1)
        else:
            raise ValueError("search_method should be either 'grid' or 'random'")

        # Timing the search process
        start_time = time.time()

        # Fit the search to find the best parameters
        search.fit(self.X_train, self.y_train)

        elapsed_time = time.time() - start_time
        print(f"Model selection completed in {elapsed_time:.2f} seconds")

        # Store the best model and parameters
        self.best_model = search.best_estimator_
        self.best_params = search.best_params_

        # Print best parameters
        print("Best parameters found: ", self.best_params)

        return self.best_model, self.best_params
