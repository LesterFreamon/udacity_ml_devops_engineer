"""Feature engineering utils for churn modeling project."""
import logging
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(
    filename='logs/feature_engineering_utils.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def add_churn_column(df: pd.DataFrame) -> pd.DataFrame:
    '''
    add churn column to df

    input:
        df: pandas dataframe

    output:
        df: pandas dataframe with new column for churn
    '''
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    return df


def encoder_helper(df: pd.DataFrame, category_lst: List[str], response: str) -> pd.DataFrame:
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    for col in category_lst:
        logging.info('creating a new column for %s', col)
        df = df.join(
            df.groupby(col)[response].mean(), 
            on=col,
            rsuffix='_Mean'
            ).rename(columns={response+'_Mean': col+ '_' + response})
        logging.info('SUCCESS: Created a new column for %s: %s', col, col+ "_" + response)
    return df


def prepare_for_modeling(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    '''
    input:
        df: pandas dataframe
        feature_columns: list of feature columns
        target_column: string of target column
        test_size: float of test size
        random_state: int of random state

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    y = df[target_column]
    X = df[feature_columns]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    logging.info('SUCCESS: split data to train and test sets')
    return X_train, X_test, y_train, y_test
