"""Test feature engineering module."""
import pandas as pd

from ..src.feature_engineering_utils import (
    add_churn_column,
    encoder_helper,
    prepare_for_modeling
    )


def test_encoder_helper():
    '''
	test encoder helper
	'''
    df = pd.DataFrame({'churn': [0, 1, 1], 'gender': ['m', 'm', 'f']})
    category_lst = ['gender']
    response = 'churn'
    result_df = encoder_helper(df, category_lst, response)
    expected_df = pd.DataFrame(
		{
		'churn': [0, 1, 1], 
		'gender': ['m', 'm', 'f'], 
		'gender_churn': [0.5, 0.5, 1.0]
		}
		)
    pd.testing.assert_frame_equal(result_df, expected_df)

def test_add_churn():
    """
    test add churn column
    """
    df = pd.DataFrame(
        {'Attrition_Flag': ['Existing Customer', 'Existing Customer', 'Attrited Customer']}
        )
    result_df = add_churn_column(df)
    expected_df = pd.DataFrame(
        {
        'Attrition_Flag': ['Existing Customer', 'Existing Customer', 'Attrited Customer'],
        'Churn': [0, 0, 1]
        }
        )
    pd.testing.assert_frame_equal(result_df, expected_df)


def test_prepare_for_modeling():
    '''test prepare for modeling'''
    df = pd.DataFrame(
        {
        'feature_1': [1.2, 2.3, 3.4, 4.5, 5.6],
        'feature_2': [0.5, 0.5, 1.0, 0.5, 1.0],
        'Churn': [0, 0, 1, 0, 1]
        }
        )
    feature_columns = ['feature_1', 'feature_2']
    target_column = 'Churn'
    X_train, X_test, y_train, y_test = prepare_for_modeling(df, feature_columns, target_column)
    assert X_train.shape == (4, 2)
    assert X_test.shape == (1, 2)
    assert y_train.shape == (4,)
    assert y_test.shape == (1,)
    assert X_train.columns.tolist() == ['feature_1', 'feature_2']
