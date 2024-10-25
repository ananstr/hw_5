# %%
from abc import ABC, abstractmethod
import pandas as pd

class FeatureTransformer(ABC):
# Abstract base class for feature transformation
    @abstractmethod
    def transform(self, df):
        pass

class GenderBinaryTransformer(FeatureTransformer):
# Converts gender into binary (M/F)
    def transform(self, df):
        df['gender'] = df['gender'].map({'M': 1, 'F': 0})
        return df

class EthnicityOneHotEncoder(FeatureTransformer):
    # Performs one hot encoding on the ethnicity column
    def transform(self, df, column='ethnicity'):
        return pd.get_dummies(df, column, drop_first=True)


