import pandas as pd


class DataLoader:
    def __init__(self, filepath: str):
        self.filepath = filepath

    def load_data(self, test: bool = False) -> pd.DataFrame:
        if test is False:
            columns = ['Survived', 'Name', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
        else:
            columns = ['PassengerId', 'Name', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

        return pd.read_csv(self.filepath, usecols=columns)
