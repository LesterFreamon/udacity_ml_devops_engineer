# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This is a machine learning project that aims to predict customer churn for a bank. The project involves building and evaluating several machine learning models, including logistic regression and random forest.

## Getting Started
* Create and activate a virtual environment
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
* Download the data from [Kaggle](https://www.kaggle.com/adammaus/predicting-churn-for-bank-customers) and place it in the data directory.


## Files and data description
Overview of the files and data present in the root directory. 
```
churn
├── data
│   └── bank_data.csv
├── logs
│   └── feature_engineering_utils.log
|   └── plot_utils.log
├── models
│   └── logistic_model.pkl
|   └── rfc_model.pkl
├── reports
|   └── eda
|   └── results
├── src
│   ├── data_utils.py
│   ├── feature_engineering_utils.py
│   ├── plot_utils.py
│   └── training.py
├── tests
│   ├── test_data_utils.py
│   └──test_feature_engineering.py
├── requirements.txt
├── churn_notebook.ipynb
├── .pylintrc
├── Guide.ipynb
└── README.md
```

## Running Files
* Run churn_notebook.ipynb.
```
jupyter notebook churn_notebook.ipynb
```
* Run all of the cells in the notebook to train the models and evaluate them.
* Look at the results in the reports directory.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details
