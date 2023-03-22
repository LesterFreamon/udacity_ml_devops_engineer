"""Train models and store model results"""
from typing import Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def train_models(
        x_train: pd.DataFrame,
        y_train: pd.DataFrame, 
        max_iter: int = 3000
        ) -> Tuple[GridSearchCV, LogisticRegression]:
    """Train models and return the best model for each

    Args:
        x_train (pd.DataFrame): training data
        y_train (pd.DataFrame): training labels
        max_iter (int, optional): max iterations for logistic regression. Defaults to 3000.

    Returns:
        Tuple[GridSearchCV, LogisticRegression]: trained models
    """
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=max_iter)
    param_grid = {
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt'],
    'max_depth' : [4,5,100],
    'criterion' :['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)

    lrc.fit(x_train, y_train)
    return cv_rfc, lrc
