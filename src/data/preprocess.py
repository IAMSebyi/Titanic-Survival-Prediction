from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd

class Preprocessor:
    def __init__(self):
        pass

    def encode_sex(self, df: pd.DataFrame) -> pd.DataFrame:
        label_encoder = LabelEncoder()
        df['Sex'] = label_encoder.fit_transform(df['Sex'])
        return df

    def encode_embarked(self, df: pd.DataFrame) -> pd.DataFrame:
        df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
        return df

    def encode_bins(self, df: pd.DataFrame) -> pd.DataFrame:
        df = pd.get_dummies(df, columns=['AgeGroup', 'FareGroup'], drop_first=True)
        return df

    def encode_title(self, df: pd.DataFrame) -> pd.DataFrame:
        label_encoder = LabelEncoder()
        df['Title'] = label_encoder.fit_transform(df['Title'])
        return df

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Example preprocessing pipeline
        df = self.encode_sex(df)
        df = self.encode_embarked(df)
        df = self.encode_bins(df)
        df = self.encode_title(df)
        return df
