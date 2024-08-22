import warnings

import pandas as pd

from sklearn.model_selection import train_test_split

from src.data.load_data import DataLoader
from src.data.feature_engineer import FeatureEngineer
from src.data.preprocess import Preprocessor
from src.models.train import ModelTrainer
from src.models.selection import ModelSelector
from src.models.evaluate import ModelEvaluator
from src.models.predict import ModelPredictor

warnings.simplefilter(action='ignore', category=FutureWarning)


def main():
    # Define dataset path
    train_path = 'data/train.csv'
    test_path = 'data/test.csv'

    # Load data
    train_loader = DataLoader(train_path)
    test_loader = DataLoader(test_path)
    train_df = train_loader.load_data(test=False)
    test_df = test_loader.load_data(test=True)

    # Perform feature engineering on data
    feature_engineer = FeatureEngineer(drop_original=True)
    train_df = feature_engineer.engineer_features(train_df)
    test_df = feature_engineer.engineer_features(test_df)

    # Preprocess data
    preprocessor = Preprocessor()
    train_df = preprocessor.preprocess_data(train_df)
    test_df = preprocessor.preprocess_data(test_df)

    # Split the data into male and female subsets
    male_train_df = train_df[train_df['Sex'] == 1]
    female_train_df = train_df[train_df['Sex'] == 0]

    male_test_df = test_df[test_df['Sex'] == 1]
    female_test_df = test_df[test_df['Sex'] == 0]

    # Separate X and y for male training data
    X_male = male_train_df.drop(['Survived'], axis=1)
    y_male = male_train_df['Survived']

    # Separate X and y for female training data
    X_female = female_train_df.drop(['Survived'], axis=1)
    y_female = female_train_df['Survived']

    # Split training data for males
    X_male_train, X_male_eval, y_male_train, y_male_eval = train_test_split(
        X_male, y_male, test_size=0.1, random_state=42)

    # Split training data for females
    X_female_train, X_female_eval, y_female_train, y_female_eval = train_test_split(
        X_female, y_female, test_size=0.1, random_state=42)

    # Select the best model for males
    selector_male = ModelSelector(X_male_train, y_male_train)
    best_male_model, best_male_params = selector_male.select_model(search_method='grid')

    # Train the best male model
    trainer_male = ModelTrainer(best_male_model)
    trainer_male.train(X_male_train, y_male_train)

    # Save male model
    male_model_path = 'models/titanic-survival-prediction-model-male'
    trainer_male.save_model(male_model_path)

    # Evaluate male model
    evaluator_male = ModelEvaluator(trainer_male.model)
    evaluator_male.evaluate(X_male_eval, y_male_eval, threshold=0.7)

    # Select the best model for females
    selector_female = ModelSelector(X_female_train, y_female_train)
    best_female_model, best_female_params = selector_female.select_model(search_method='grid')

    # Train the best female model
    trainer_female = ModelTrainer(best_female_model)
    trainer_female.train(X_female_train, y_female_train)

    # Save female model
    female_model_path = 'models/titanic-survival-prediction-model-female'
    trainer_female.save_model(female_model_path)

    # Evaluate female model
    evaluator_female = ModelEvaluator(trainer_female.model)
    evaluator_female.evaluate(X_female_eval, y_female_eval, threshold=0.4)

    # Predict for male test data
    predictor_male = ModelPredictor(male_model_path)
    male_result_df = predictor_male.predict(male_test_df, threshold=0.7)

    # Predict for female test data
    predictor_female = ModelPredictor(female_model_path)
    female_result_df = predictor_female.predict(female_test_df, threshold=0.4)

    # Combine male and female predictions
    combined_results_df = pd.concat([male_result_df, female_result_df]).sort_values(by='PassengerId')

    # Save combined predictions to CSV
    output_path = 'submission.csv'
    predictor_male.save_predictions(combined_results_df, output_path)


if __name__ == '__main__':
    main()
