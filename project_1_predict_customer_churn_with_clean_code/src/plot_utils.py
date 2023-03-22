"""Plotting utils"""
import logging
import os
from typing import NamedTuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
import shap

sns.set()

logging.basicConfig(
    filename='logs/plot_utils.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

class ReportData(NamedTuple):
    """Report data"""
    y_train: pd.DataFrame
    y_test: pd.DataFrame
    y_train_preds_lr: np.array
    y_train_pred_rf: np.array
    y_test_preds_lr: np.array
    y_test_preds_rf: np.array

def save_histogram(func: callable) -> callable:
    """Decorator to save histogram to file"""
    def wrapper(df: pd.DataFrame, cat_column: str, file_name: str, *args, **kwargs) -> None:
        """Wrapper function to save histogram to file"""
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

def perform_eda(
        df: pd.DataFrame,
        cat_columns: List[str],
        normalized_bar_plot_columns: List[str],
        distribution_plot_columns: List[str],
        output_pth: str
        ) -> None:
    '''
    perform eda on df and save figures to images folder

    input:
        df: pandas dataframe
        cat_columns: list of categorical columns
        normalized_bar_plot_columns: list of columns to create normalized bar plots
        output_pth: path to save figures

    output:
        None
    '''
    logging.info('Performing EDA')
    try:
        for cat_column in cat_columns:
            output_file_path = os.path.join(output_pth, f'{cat_column}_hist.png')
            plot_histogram(df, cat_column, output_file_path)

        for normalized_bar_plot_column in normalized_bar_plot_columns:
            output_file_path = os.path.join(output_pth, f'{normalized_bar_plot_column}_bar.png')
            plot_normalized_bar(df, normalized_bar_plot_column, output_file_path)

        for distribution_plot_column in distribution_plot_columns:
            output_file_path = os.path.join(output_pth, f'{distribution_plot_column}_dist.png')
            plot_distribution(df, distribution_plot_column, output_file_path)

        plt.figure(figsize=(20,10))
        sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
        plt.savefig(os.path.join(output_pth, 'heatmap.png'))
        plt.show()
        plt.close()
        logging.info('SUCCESS: EDA complete')
    except KeyError as error:
        logging.error('ERROR: %s', str(error))

def plot_performance_curves(
        X_test: pd.DataFrame,
        y_test:pd.DataFrame,
        cv_rfc: GridSearchCV,
        lrc: LogisticRegression,
        output_pth: str
        ) -> None:
    """Plot performance curves
    
    Args:
        X_test (pd.DataFrame): X test data
        y_test (pd.DataFrame): y test data
        cv_rfc (GridSearchCV): GridSearchCV object
        lrc (LogisticRegression): LogisticRegression object
        output_pth (str): path to save figure
        
    Returns:
        None
    """
    plt.figure(figsize=(15, 8))
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    plt.savefig(os.path.join(output_pth, 'lrc_roc_curve.png'))
    plt.show()

    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    plot_roc_curve(cv_rfc.best_estimator_, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig(os.path.join(output_pth, 'rfc_roc_curve.png'))
    plt.show()


def plot_feature_importance(X_test: pd.DataFrame, cv_rfc: GridSearchCV, output_pth: str) -> None:
    """Plot feature importance
    
    Args:
        X_test (pd.DataFrame): X test data
        cv_rfc (GridSearchCV): GridSearchCV object
        output_pth (str): path to save figure

    Returns:
        None
    """
    explainer = shap.TreeExplainer(cv_rfc.best_estimator_)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.savefig(os.path.join(output_pth, 'shap_feature_importance.png'))
    plt.show()

    # Calculate feature importances
    importances = cv_rfc.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_test.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20,5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_test.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_test.shape[1]), names, rotation=90)
    plt.savefig(os.path.join(output_pth, 'random_forest_feature_importance.png'))
    plt.show()

def classification_report_image(report_data: ReportData, output_pth: str) -> None:
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
        report_data: ReportData object
        output_pth: path to save image

    output:
             None
    '''
    style_dict = {'fontdict': {'fontsize': 10}, 'fontproperties': 'monospace'}
    plt.rc('figure', figsize=(5, 5))
    #plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str('Random Forest Train'), **style_dict)
    plt.text(
        0.01,
        0.05,
        str(classification_report(report_data.y_test, report_data.y_test_preds_rf)),
        **style_dict
        )
    plt.text(0.01, 0.6, str('Random Forest Test'), **style_dict)
    plt.text(
        0.01,
        0.7,
        str(classification_report(report_data.y_train, report_data.y_train_preds_rf)),
        **style_dict
        )
    plt.axis('off')
    plt.savefig(os.path.join(output_pth, 'rfc_classification_report.png'))
    plt.show()

    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'), **style_dict)
    plt.text(
        0.01,
        0.05,
        str(classification_report(report_data.y_train, report_data.y_train_preds_lr)),
        **style_dict
        )
    plt.text(0.01, 0.6, str('Logistic Regression Test'), **style_dict)
    plt.text(
        0.01,
        0.7,
        str(classification_report(report_data.y_test, report_data.y_test_preds_lr)),
        **style_dict
        )
    plt.axis('off')
    plt.savefig(os.path.join(output_pth, 'lrc_classification_report.png'))
    plt.show()
