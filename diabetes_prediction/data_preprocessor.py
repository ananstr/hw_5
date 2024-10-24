# %%
import pandas as pd

class NaNRemover:
    def __init__(self, columns_with_nan):
    # Constructor that initializes the NaNRemover class with a list of columns we want to treat
    # Takes as argumant a list of columns from which we want to remove rows with NaN values.
        self.columns_with_nan = columns_with_nan
    
    def remove_nan(self, df):
    # Removes rows with NaN values in specific columns
    # Arguments: DataFrame to clean
    # Returns: Cleaned DataFrame with rows removed
        return df.dropna(subset=self.columns_with_nan)

class NaNFiller:
    def __init__(self, columns_to_fill):
    # Constructor that initializes the NaNFiller class with columns to fill NaN values with the mean
    # Takes as argument a list of columns from which we want to fill NaN values with the mean
        self.columns_to_fill = columns_to_fill
    
    def fill_nan(self, df):
    # Fills NaN values in specific columns with the mean
    # Take as argument a DataFrame to fill NaN values in
    # Returns a DataFrame with NaN values filled with the mean
        for column in self.columns_to_fill:
            df[column].fillna(df[column].mean(), inplace=True)
        return df


