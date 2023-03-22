'''This module contains functions for performing EDA and feature engineering on the churn data.'''


# import libraries
import logging
import os
from typing import List, Tuple
import joblib
import numpy as np

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, plot_roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split
import shap

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(CURRENT_DIR, 'images')
EDA_DIR = os.path.join(IMAGE_DIR, 'eda')
RESULTS_DIR = os.path.join(IMAGE_DIR, 'results')
MODEL_DIR = os.path.join(CURRENT_DIR, 'models')

logging.basicConfig(
    filename='logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


class PredictionsData:
    '''Class to hold the predictions data'''

    def __init__(
            self,
            y_train_preds_rf: pd.Series,
            y_test_preds_rf: pd.Series,
            y_train_preds_lr: pd.Series,
            y_test_preds_lr: pd.Series
    ) -> None:
        '''Constructor for PredictionsData class

        input:
            y_train_preds_rf: training predictions from random forest
            y_test_preds_rf: test predictions from random forest
            y_train_preds_lr: training predictions from logistic regression
            y_test_preds_lr: test predictions from logistic regression

        output:
            None
        '''
        self.y_train_preds_rf = y_train_preds_rf
        self.y_test_preds_rf = y_test_preds_rf
        self.y_train_preds_lr = y_train_preds_lr
        self.y_test_preds_lr = y_test_preds_lr

    def __str__(self) -> str:
        '''String representation of PredictionsData object'''
        return (
            f'PredictionsData(y_train_preds_rf={self.y_train_preds_rf}, '
            'y_test_preds_rf={self.y_test_preds_rf}, '
            'y_train_preds_lr={self.y_train_preds_lr}, y_test_preds_lr={self.y_test_preds_lr})'
        )

    def __repr__(self) -> str:
        '''String representation of PredictionsData object'''
        return (
            f'PredictionsData(y_train_preds_rf={self.y_train_preds_rf}, '
            'y_test_preds_rf={self.y_test_preds_rf}, '
            'y_train_preds_lr={self.y_train_preds_lr}, y_test_preds_lr={self.y_test_preds_lr})'
        )


def _get_models_predictions(
        cv_rfc: GridSearchCV,
        lrc: LogisticRegression,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame
) -> PredictionsData:
    '''
    helper function to get the predictions from the models
    input:
            cv_rf: GridSearchCV Random Forest Classifier
            lrc: Logistic Regression Classifier
            X_train: X training data
            X_test: X testing data

    output:
        predictions_data: PredictionsData object
    '''
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)
    predictions_data = PredictionsData(
        y_train_preds_rf,
        y_test_preds_rf,
        y_train_preds_lr,
        y_test_preds_lr
    )
    return predictions_data


def save_histogram(func: callable) -> callable:
    '''Decorator to save histogram to file

    input:
        func: function to decorate
    output:
        wrapper: decorated function
    '''
    def wrapper(df: pd.DataFrame, cat_column: str,
                file_name: str, *args, **kwargs) -> None:
        '''Wrapper function to save histogram to file

        input:
            df: pandas dataframe
            cat_column: categorical column
            file_name: name of file to save figure to
            *args: arguments to pass to func
            **kwargs: keyword arguments to pass to func

        output:
            None
        '''
        plt.figure(figsize=(20, 10))
        func(df, cat_column, *args, **kwargs)
        plt.savefig(file_name)
        plt.show()
        plt.close()
    return wrapper


@save_histogram
def plot_histogram(df: pd.DataFrame, cat_column: str, *args) -> None:
    """Plot histogram"""
    plt.title(f'{cat_column} Histogram', *args)
    df[cat_column].hist()


@save_histogram
def plot_normalized_bar(df: pd.DataFrame, cat_column: str, *args) -> None:
    """Plot normalized bar plot"""
    plt.title(f'{cat_column} Bar Plot')
    df[cat_column].value_counts(normalize=True).plot(kind='bar', *args)


@save_histogram
def plot_distribution(df: pd.DataFrame, cat_column: str, *args) -> None:
    """Plot distribution plot"""
    plt.title(f'{cat_column} Distribution Plot')
    sns.histplot(df[cat_column], stat='density', kde=True, *args)


def import_data(pth: str, to_drop_cols: List[str]) -> pd.DataFrame:
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth, index_col=0)
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda x: 1 if x == 'Existing Customer' else 0)
    df = df.drop(columns=to_drop_cols)
    return df


def perform_eda(
        df: pd.DataFrame,
        cat_columns: List[str],
        quant_columns: List[str],
        distribution_plot_columns: List[str],
        eda_dir: str = EDA_DIR
) -> None:
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe
            cat_columns: list of categorical columns
            quant_columns: list of quantitative columns
            distribution_plot_columns: list of columns to create distribution plots
            eda_dir: directory to save figures to

    output:
            None
    '''

    logging.info('Performing EDA')
    logging.info('Plotting histograms')
    for cat_column in cat_columns:
        try:
            logging.info('Plotting histogram for %s', cat_column)
            output_file_path = os.path.join(eda_dir, f'{cat_column}_hist.png')
            plot_histogram(df, cat_column, output_file_path)
        except KeyError as exception:
            logging.error('Error plotting histogram for %s', cat_column)
            logging.error(exception)

    logging.info('Plotting Normalized Bar Plots')
    for quant_column in quant_columns:
        try:
            logging.info('Plotting normalized bar plot for %s', quant_column)
            output_file_path = os.path.join(
                eda_dir, f'{quant_column}_bar_plot.png')
            plot_normalized_bar(df, quant_column, output_file_path)
        except KeyError as exception:
            logging.error('Error plotting bar plot for %s', quant_column)
            logging.error(exception)

    logging.info('Plotting Distribution Plots')
    for distribution_plot_column in distribution_plot_columns:
        try:
            logging.info('Plotting distribution plot for %s',
                         distribution_plot_column)
            file_name = f'{distribution_plot_column}_distribution_plot.png'
            output_file_path = os.path.join(eda_dir, file_name)
            plot_distribution(df, distribution_plot_column, output_file_path)
        except KeyError as exception:
            logging.error('Error plotting distribution plot for %s',
                          distribution_plot_column)
            logging.error(exception)

    logging.info('Plotting Heat Map')
    plt.figure(figsize=(20, 10))
    plt.title('Heat Map')
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(os.path.join(eda_dir, 'heat_map.png'))
    plt.show()
    plt.close()

    logging.info('Finished EDA')


def encoder_helper(
        df: pd.DataFrame, category_lst: List[str], response: str) -> pd.DataFrame:
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name

    output:
            df: pandas dataframe with new columns for
    '''
    for col in category_lst:
        logging.info('creating a new column for %s', col)
        df = df.join(
            df.groupby(col)[response].mean(),
            on=col,
            rsuffix='_Mean'
        ).rename(columns={response+'_Mean': col + '_' + response})
        logging.info('SUCCESS: Created a new column for %s: %s',
                     col, col + "_" + response)
    return df


def perform_feature_engineering(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
    test_size: float = 0.3,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    '''Performs feature engineering and returns X_train, X_test, y_train, y_test.
    input:
        df: pandas dataframe
        feature_columns: list of columns that contain features
        target_column: string of target column name
        test_size: float of test size [default = 0.3]
        random_state: int of random state [default = 42]

    output:
        x_train: X training data
        x_test: X testing data
        y_train: y training data
        y_test: y testing data
    '''
    features_df = df[feature_columns]
    target_ser = df[target_column]
    x_train, x_test, y_train, y_test = train_test_split(
        features_df, target_ser, test_size=test_size, random_state=random_state
    )
    return x_train, x_test, y_train, y_test


def classification_report_image(
        y_train: pd.Series,
        y_test: pd.Series,
        predictions_data: PredictionsData,
        results_dir: str = RESULTS_DIR
) -> None:
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
        y_train: training response values
        y_test:  test response values
        predictions_data: named tuple containing predictions data
        results_dir: string of results directory [default = RESULTS_DIR]

    output:
        None
    '''
    style_dict = {'fontdict': {'fontsize': 10}, 'fontproperties': 'monospace'}
    x_style = 0.01
    rf_train_results = classification_report(
        y_train, predictions_data.y_train_preds_rf)
    rf_test_results = classification_report(
        y_test, predictions_data.y_test_preds_rf)

    lr_train_results = classification_report(
        y_train, predictions_data.y_train_preds_lr)
    lr_test_results = classification_report(
        y_test, predictions_data.y_test_preds_lr)

    logging.info('calculating scores')

    plt.rc('figure', figsize=(5, 5))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    logging.info('random forest results')
    logging.info('train results')
    logging.info(rf_train_results)
    plt.text(x_style, 1.25, str('Random Forest Train'), **style_dict)
    # approach improved by OP -> monospace!
    plt.text(x_style, 0.05, str(rf_train_results), **style_dict)

    logging.info('test results')
    logging.info(rf_test_results)
    plt.text(x_style, 0.6, str('Random Forest Test'), **style_dict)
    # approach improved by OP -> monospace!
    plt.text(x_style, 0.7, str(rf_test_results), **style_dict)
    plt.axis('off')
    plt.savefig(os.path.join(results_dir, 'rfc_classification_report.png'))
    plt.show()

    plt.rc('figure', figsize=(5, 5))
    logging.info('logistic regression results')
    logging.info('train results')
    plt.text(x_style, 1.25, str('Logistic Regression Train'), **style_dict)
    # approach improved by OP -> monospace!
    plt.text(x_style, 0.05, str(lr_train_results), **style_dict)

    logging.info('test results')
    plt.text(x_style, 0.6, str('Logistic Regression Test'), **style_dict)
    # approach improved by OP -> monospace!
    plt.text(x_style, 0.7, str(lr_test_results), **style_dict)
    plt.axis('off')
    plt.savefig(os.path.join(results_dir, 'lrc_classification_report.png'))
    plt.show()


def feature_importance_plot(model: RandomForestClassifier,
                            X_test: pd.DataFrame, results_dir: str = RESULTS_DIR) -> None:
    '''
    creates and stores the feature importances the results directory
    input:
            model: model object containing feature_importances_
            X_test: pandas dataframe of X values
            results_dir: string of results directory [default = RESULTS_DIR]

    output:
             None
    '''
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.savefig(os.path.join(results_dir, 'shap_feature_importance.png'))
    plt.show()

    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_test.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_test.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_test.shape[1]), names, rotation=90)
    plt.savefig(
        os.path.join(
            results_dir,
            'random_forest_feature_importance.png'))
    plt.show()


def train_rfc_lrc(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        max_iter: int = 3000
) -> Tuple[GridSearchCV, LogisticRegression]:
    '''Train Random Forest Classifier and Logistic Regression Classifier and return the models.

        input:
            X_train: X training data
            y_train: y training data

        output:
            rfc: GridSearchCV Random Forest Classifier
            lrc: Logistic Regression Classifier
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=max_iter)
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)
    return cv_rfc, lrc


def create_performance_report(
        X_test: pd.DataFrame,
        y_test: pd.Series,
        rfc_model: RandomForestClassifier,
        lrc_model: LogisticRegression,
        results_dir: str = RESULTS_DIR
) -> None:
    '''
    creates and stores the performance report in results directory
    input:
                X_test: pandas dataframe of X test values
                y_test: pandas series of y test values
                rfc_model: Random Forest Classifier model
                lrc_model: Logistic Regression Classifier model
                results_dir: directory to store results
    output:
        None
    '''
    lrc_plot = plot_roc_curve(lrc_model, X_test, y_test)
    plt.figure(figsize=(16, 8))
    a_x = plt.gca()
    _ = plot_roc_curve(rfc_model, X_test, y_test, ax=a_x, alpha=0.8)
    lrc_plot.plot(ax=a_x, alpha=0.8)
    plt.savefig(os.path.join(results_dir, 'roc_curve.png'))


def train_models(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        model_dir: str = MODEL_DIR
) -> Tuple[GridSearchCV, LogisticRegression, PredictionsData]:
    '''
    train and log model scores
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
              model_dir: path to store the model
    output:
        cv_rfc: GridSearchCV Random Forest Classifier
        lrc: Logistic Regression Classifier
        predictions_data: named tuple containing predictions data
    '''
    cv_rfc, lrc = train_rfc_lrc(X_train, y_train)

    predictions_data = _get_models_predictions(cv_rfc, lrc, X_train, X_test)

    classification_report_image(y_train, y_test, predictions_data)

    joblib.dump(
        cv_rfc.best_estimator_, os.path.join(model_dir, 'rfc_model.pkl')
    )
    joblib.dump(lrc, os.path.join(model_dir, 'logistic_model.pkl'))

    feature_importance_plot(cv_rfc.best_estimator_, X_test)

    return cv_rfc, lrc, predictions_data


if __name__ == "__main__":
    data_df = import_data(r'./data/bank_data.csv', ['Attrition_Flag'])
    cat_cols = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    quant_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio'
    ]
    perform_eda(data_df, cat_cols, quant_cols, ['Total_Trans_Ct'])
    encoded_df = encoder_helper(data_df, cat_cols, 'Churn')
    feature_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
             'Income_Category_Churn', 'Card_Category_Churn']
    X_tr, X_tst, y_tr, y_tst = perform_feature_engineering(encoded_df, feature_cols, 'Churn')
    cv_rand_for_cls, log_reg_cls, pred_data = train_models(X_tr, X_tst, y_tr, y_tst)
    create_performance_report(X_tst, y_tst, cv_rand_for_cls, log_reg_cls)
