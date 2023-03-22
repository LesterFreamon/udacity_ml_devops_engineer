# Predict Customer Churn

Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
Many companies today are concerned with retaining their customers, since it is much more expensive to acquire a new customer than to retain an existing one. In this project, we attempt to build a machine learning model to predict customer churn. We will use data from a "bank" that contains relevant information about customers.


## Files and data description

### File Structure

```
project_1_churn
├── data
│   └── bank_data.csv
├── models
│   ├── logistic_model.pkl
│   └── rfc_model.pkl
├── images
│   ├── eda
│   └── results
├── logs
│   ├── churn_library.log
│   └── churn_script_logging_and_tests.log
├── .pylintrc
├── churn_library.py  #  main file that contains all functions
├── churn_notebook.ipynb
├── churn_script_logging_and_tests.py #  main file that runs the testing script
├── Guide.ipynb
├── README.md
├── requirements_py3.6.txt
└── requirements_py3.8.txt
```

### Data
#### Data File
* bank_data.csv: The data set contains 10,000 observations of 14 variables. Each row represents a customer, each column contains customer’s attributes described on the column Metadata.

#### Data Description

| Column Name | Description |
| --- | --- |
| CLIENTNUM | Client number. Unique identifier for the customer holding the account |
| Attrition_Flag | Whether the customer is Existing or Attrited |
| Customer_Age | Age of the customer |
| Gender | Gender of the customer |
| Dependent_count | Number of dependents |
| Education_Level | Education level of the customer |
| Marital_Status | Marital status of the customer |
| Income_Category | Annual income of the customer |
| Card_Category | Type of card |
| Months_on_book | Period of relationship with bank |
| Total_Relationship_Count | Total no. of products held by the customer |
| Months_Inactive_12_mon | No. of months inactive in the last 12 months |
| Contacts_Count_12_mon | No. of Contacts in the last 12 months |
| Credit_Limit | Credit Limit on the Credit Card |
| Total_Revolving_Bal | Total Revolving Balance on the Credit Card |
| Avg_Open_To_Buy | Open to Buy Credit Line (Average of last 12 months) |
| Total_Amt_Chng_Q4_Q1 | Change in Transaction Amount (Q4 over Q1) |
| Total_Trans_Amt | Total Transaction Amount (Last 12 months) |
| Total_Trans_Ct | Total Transaction Count (Last 12 months) |
| Total_Ct_Chng_Q4_Q1 | Change in Transaction Count (Q4 over Q1) |
| Avg_Utilization_Ratio | Average Card Utilization Ratio |


## Running Files

* Go to the project folder
* Create a virtual environment and run it
```bash
python3 -m venv .venv
activate .venv/bin/activate
```
* Install the requirements
```bash
pip install -r requirements_py3.8.txt
```
* Run the script
```bash
python churn_library.py
```
* The models will be saved in the models folder
* The EDA and results will be saved in the images folder
* The logs will be saved in the logs folder


## Running Tests
```bash
python churn_script_logging_and_tests.py
```