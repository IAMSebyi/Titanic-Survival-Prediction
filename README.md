# Titanic Survival Prediction

This project was developed as part of the [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) competition on [Kaggle](https://www.kaggle.com/). The objective is to predict the survival of passengers aboard the Titanic using machine learning techniques. The approach involves splitting the dataset by gender and training separate models for male and female passengers, leveraging the powerful XGBoost classifier. The project not only aims to maximize accuracy but also to minimize false negatives through careful model selection and tuning.
## Project Structure

```plaintext
├── data/                     # Directory for the dataset
│   ├── train.csv             # Training dataset
│   └── test.csv              # Test dataset
├── models/                   # Directory for saving trained models
├── src/                      # Source code for the project
│   ├── data/
│   │   ├── load_data.py      # Script to load data
│   │   ├── feature_engineer.py  # Feature engineering methods
│   │   └── preprocess.py     # Preprocessing methods
│   ├── models/
│   │   ├── train.py          # Training script
│   │   ├── selection.py      # Model selection script
│   │   ├── evaluate.py       # Evaluation script
│   │   └── predict.py        # Prediction script
├── main.py                   # Main script to run the entire pipeline
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## Installation

### 1. Clone the repository:

```bash
git clone https://github.com/your-username/titanic-survival-prediction.git
cd titanic-survival-prediction
```

### 2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install the required packages:

```bash
pip install -r requirements.txt
```

## Dataset

The project uses the famous [Titanic dataset](https://www.kaggle.com/competitions/titanic/data), which contains data about the passengers on the Titanic, including whether they survived or not. The dataset should be stored in the data/ directory.

- train.csv: Contains the training data including the target variable Survived.
- test.csv: Contains the test data which you need to predict survival on.

## Features

The following features are engineered, processed and used for prediction:

- Pclass: Ticket class	(1 = 1st, 2 = 2nd, 3 = 3rd)
- Sex: Binary representing gender
- Age: Age in years
- SibSp: The number of siblings / spouses aboard the Titanic
- ParCh: The number of parents / children aboard the Titanic
- Fare: Passenger fare
- Embarked: Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
- Family Size: Total number of family members onboard (SibSp + Parch + 1).
- Alone: Boolean indicating whether the passenger was alone.
- Title: Extracted title from the passenger's name.
- Priority: Whether the passenger has priority based on the 'Women and Children First' rule.
- AgeGroup: Binned age groups.
- FareGroup: Binned fare groups.

## Gender-Based Model Selection

The project applies separate models for male and female passengers:

- Male Model: A specific XGBoost model is trained and optimized for male passengers.
- Female Model: A different XGBoost model is trained and optimized for female passengers.

### Model Selection

The ModelSelector class uses Grid Search or Randomized Search to find the best hyperparameters for each model.

### Evaluation

The ModelEvaluator class evaluates the trained models using a custom probability threshold for prediction.

### Prediction

The ModelPredictor class predicts the survival of passengers in the test dataset, separately for male and female passengers.

## How to Run the Project

### 1. Prepare the dataset: 

Ensure that train.csv and test.csv are in the data/ directory.

### 2. Run the main script:

```
python main.py
```

### 3. Check the outputs:

- Trained models will be saved in the models/ directory.
- The predictions will be saved in a file named submission.csv in the project root directory.

## LICENSE

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
