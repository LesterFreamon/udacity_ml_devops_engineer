'''This script is used to train and evaluate a model to predict customer churn.

Author: Adam Arnon
Created: 03/22/2023
'''
import logging
import os
import tempfile

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from churn_library import (PredictionsData, encoder_helper, import_data, perform_eda,
                           perform_feature_engineering, setup_logger, train_models)

setup_logger('churn_script_logging_and_tests', 'logs/churn_script_logging_and_tests.log')

test_logger = logging.getLogger('churn_script_logging_and_tests')


def test_import() -> None:
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv", ["Dependent_count"])
        test_logger.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        test_logger.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        test_logger.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda() -> None:
    '''
    test perform eda function
    '''
    data = {
        'cat_column1': ['a', 'b', 'a', 'b', 'a'],
        'cat_column2': ['x', 'x', 'y', 'x', 'y'],
        'quant_column1': [1, 2, 3, 4, 5],
        'quant_column2': [6, 7, 8, 9, 10],
        'response': [0, 1, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    cat_columns = ['cat_column1', 'cat_column2']
    quant_columns = ['quant_column1', 'quant_column2']
    distribution_plot_columns = ['quant_column1']

    # Create a temporary directory to store the EDA images
    with tempfile.TemporaryDirectory() as eda_dir:
        # Set the global EDA_DIR variable to the temporary directory

        try:
            # Call the perform_eda function
            perform_eda(df, cat_columns, quant_columns, distribution_plot_columns, eda_dir)

            # Check if the images are created in the temporary directory
            for column in cat_columns:
                assert os.path.exists(os.path.join(eda_dir, f'{column}_hist.png'))
            for column in quant_columns:
                assert os.path.exists(os.path.join(eda_dir, f'{column}_bar_plot.png'))

            for column in distribution_plot_columns:
                assert os.path.exists(os.path.join(eda_dir, f'{column}_distribution_plot.png'))
            assert os.path.exists(os.path.join(eda_dir, 'heat_map.png'))
            test_logger.info("Testing perform_eda: SUCCESS")
        except FileNotFoundError as err:
            test_logger.error("Testing perform_eda: The file wasn't found")
            raise err


def test_encoder_helper() -> None:
    '''
    test encoder helper
    '''
    data = {
        'A': ['a', 'b', 'a', 'b', 'a'],
        'B': ['x', 'x', 'y', 'x', 'y'],
        'response': [0, 1, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    category_lst = ['A', 'B']
    response = 'response'

    # Call the encoder_helper function

    try:
        result_df = encoder_helper(df, category_lst, response)
        test_logger.info("Testing encoder_helper: SUCCESS")
    except FileNotFoundError as err:
        test_logger.error("Testing encoder_helper: The file wasn't found")
        raise err

    # Expected output
    expected_data = {
        'A': ['a', 'b', 'a', 'b', 'a'],
        'B': ['x', 'x', 'y', 'x', 'y'],
        'response': [0, 1, 1, 0, 1],
        'A_response': [2/3, 1/2, 2/3, 1/2, 2/3],
        'B_response': [1/3, 1/3, 1., 1/3, 1.]
    }
    expected_df = pd.DataFrame(expected_data)

    # Check if the output matches the expected output
    try:
        pd.testing.assert_frame_equal(result_df, expected_df, check_like=True)
    except AssertionError as err:
        test_logger.error(
            "Testing encoder_helper: The output doesn't match the expected output"
        )
        raise err


def test_perform_feature_engineering() -> None:
    '''
    test perform_feature_engineering
    '''
    data = {
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [6, 7, 8, 9, 10],
        'target': [0, 1, 0, 1, 0]
    }
    df = pd.DataFrame(data)
    feature_columns = ['feature1', 'feature2']
    target_column = 'target'
    test_size = 0.4
    random_state = 42

    # Call the perform_feature_engineering function
    try:
        x_train, x_test, y_train, y_test = perform_feature_engineering(
            df, feature_columns, target_column, test_size, random_state
        )
        test_logger.info("Testing perform_feature_engineering: SUCCESS")
    except FileNotFoundError as err:
        test_logger.error(
            "Testing perform_feature_engineering: The file wasn't found"
        )
        raise err

    # Check if the output shapes are correct
    try:
        assert x_train.shape == (3, 2)
        assert x_test.shape == (2, 2)
        assert y_train.shape == (3,)
        assert y_test.shape == (2,)
    except AssertionError as err:
        test_logger.error(
            "Testing perform_feature_engineering: The output shapes don't match the expected output"
        )
        raise err

    # Check if the output types are correct
    try:
        assert isinstance(x_train, pd.DataFrame)
        assert isinstance(x_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
    except AssertionError as err:
        test_logger.error(
            "Testing perform_feature_engineering: The output types don't match the expected output"
        )
        raise err


def test_train_models() -> None:
    '''
    test train_models
    '''
    X_train = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    })
    X_test = pd.DataFrame({
        'feature1': [7, 8],
        'feature2': [9, 10]
    })
    y_train = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_test = pd.Series([1, 0])

    # Create a temporary directory to store the models
    with tempfile.TemporaryDirectory() as model_dir:
        # Call the train_models function
        cv_rfc, lrc, predictions_data = train_models(
            X_train, X_test, y_train, y_test, model_dir)

        # Check if the output types are correct

        try:
            assert isinstance(cv_rfc, GridSearchCV)
            assert isinstance(lrc, LogisticRegression)
            assert isinstance(predictions_data, PredictionsData)
            test_logger.info("Testing train_models types: SUCCESS")
        except AssertionError as err:
            test_logger.error(
                "Testing train_models types: The output types don't match the expected output"
            )
            raise err

            # Check if the models are saved in the temporary directory
        try:
            assert os.path.exists(os.path.join(model_dir, 'rfc_model.pkl'))
            assert os.path.exists(os.path.join(
                model_dir, 'logistic_model.pkl'))
            test_logger.info("Testing train_models models: SUCCESS")
        except AssertionError as err:
            test_logger.error(
                "Testing train_models models: The models weren't saved in the temporary directory")
            raise err


if __name__ == "__main__":
    test_import()
    test_eda()
    test_encoder_helper()
    test_perform_feature_engineering()
    test_train_models()
