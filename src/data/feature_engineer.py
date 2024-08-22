import pandas as pd
import numpy as np


class FeatureEngineer:
    def __init__(self, drop_original=False):
        self.drop_original = drop_original

    def create_family_size(self, df: pd.DataFrame) -> pd.DataFrame:
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1  # Add 1 to include the passenger themselves
        return df

    def create_alone(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Alone'] = (df.FamilySize == 1)
        return df

    def create_title(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Title'] = df.Name.str.split(',').str[1].str.split('.').str[0].str.strip()

        # Drop name column
        df.drop('Name', axis=1, inplace=True)
        return df

    def create_priority(self, df: pd.DataFrame) -> pd.DataFrame:
        # Birkenhead Drill: 'Woman and Children First!'
        # Check for women and children (boys) using the sex and title columns
        df['Priority'] = ((df.Title == 'Master') | (df.Sex == 'female'))
        return df

    def bin_age(self, df: pd.DataFrame) -> pd.DataFrame:
        # Fill N/A age values with the median
        df['Age'].fillna(df['Age'].median(), inplace=True)

        bins = [0, 18, 30, 40, 50, 60, 120]
        labels = ['0-18', '18-30', '30-40', '40-50', '50-60', '60-120']
        df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)
        if self.drop_original:
            df.drop('Age', axis=1, inplace=True)
        return df

    def bin_fare(self, df: pd.DataFrame) -> pd.DataFrame:
        # Fill N/A fare values with the median
        df['Fare'].fillna(df['Fare'].median(), inplace=True)

        bins = [0, 25, 50, 100, 200, np.inf]
        labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
        df['FareGroup'] = pd.cut(df['Fare'], bins=bins, labels=labels)
        if self.drop_original:
            df.drop('Fare', axis=1, inplace=True)
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.create_family_size(df)
        df = self.create_alone(df)
        df = self.bin_age(df)
        df = self.bin_fare(df)
        df = self.create_title(df)
        df = self.create_priority(df)
        return df
