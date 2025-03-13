"""
AI Introduction Assignment - Practical Tasks

This assignment is designed to give you hands-on experience with some of the Python libraries commonly used in AI.
You will implement functions that demonstrate basic usage of these libraries.

Please make sure to install the necessary libraries before you start implementing the functions:
- numpy
- pandas
- scikit-learn

You can install them using pip:
pip install numpy pandas scikit-learn
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def perform_linear_regression(X, y):
    """
    Perform a simple linear regression.

    Parameters:
    X (numpy.ndarray): 2D array where each row represents a sample and each column represents a feature.
    y (numpy.ndarray): 1D array of targets associated with each sample in X.

    Returns:
    coefficients (numpy.ndarray): Coefficients of the linear regression model.
    intercept (float): Intercept of the linear regression model.
    """
    # Your code here
    pass

def calculate_statistics(data):
    """
    Calculate mean, median, and standard deviation of a dataset.

    Parameters:
    data (numpy.ndarray): 1D array containing numerical data.

    Returns:
    statistics (dict): A dictionary containing the mean, median, and standard deviation of the data,
                        with keys 'mean', 'median', and 'std'.
    """
    # Your code here
    pass

def preprocess_dataframe(df):
    """
    Preprocess a pandas dataframe: drop missing values, encode categorical variables (if any), and normalize numerical features.

    Parameters:
    df (pandas.DataFrame): DataFrame to preprocess.

    Returns:
    processed_df (pandas.DataFrame): The preprocessed DataFrame.
    """
    # Your code here
    pass

