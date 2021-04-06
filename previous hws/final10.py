#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8


# In[1]:

# Standard libraries:

# In[2]:


import os
import logging


# Third-party libraries:

# In[3]:


import boto3
from catboost import CatBoostClassifier
import joblib
from joblib import dump
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import s3fs
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


# In[2]:

# Save location of notebook:

# In[4]:


notebook_dir = os.getcwd()
notebook_dir


# Path to repository on my machine:

# In[5]:


bankrupt_dir = "/Users/jeongjaehui/Documents/Stats_404/JEONG-JAEHEE"


# Change directories to the repository on your local machine:

# In[6]:


os.chdir(bankrupt_dir)


# In[7]:


logging.basicConfig(level=logging.INFO)
# Define one logger for current file, per
# https://www.loggly.com/blog/4-reasons-a-python-logging-library-is-much-better-than-putting-print-statements-everywhere/
LOGGER = logging.getLogger(__name__)


# Save variables to connent to AWS

# In[8]:


BUCKET_NAME = "stats404-project-jaehee"
KEY_NAME_DATA = "bankrupt_data.csv"
KEY_NAME_MODEL = "cat_model.joblib"
FILE_NAME = "s3://stats404-project-jaehee/bankrupt_data.csv"


# In[3]:

# In[9]:


def get_X(df):
    """
    Subtract output features to get X features
    :param df: dataframe
    :return: X columns( except the first column)
    """
    return df.iloc[:, 1:]


# In[10]:


def get_y(df):
    """
    Subtract "Bankrupt?" column to get y feature
    :param df: dataframe
    :return: y column( "Bankrupt")
    """
    return df["Bankrupt?"]


# efie a function to print confusion matrix

# In[11]:


def print_confusion_matrix(y_val, pred):
    """
    Generate a confusion matrix to check the result of a prediction
    :param y_val: Actual Y values
    :param pred: Predicted Y values
    :return: A confusion matrix
    """
    print(
        pd.DataFrame(
            confusion_matrix(y_val, pred),
            columns=["Predicted Nagative", "Predicted Positive"],
            index=["Actual Negative", "Actual Positive"],
        )
    )


# In[12]:


def cat_model_pred_prob_threshold_37(X_data):
    """
    Change a threshold to 0.37 to predict bankruptcy better
    :param X_data: Actual X values
    :return: 1 or 0 (if the probability is higher than 0.37, returns 1)
    """
    return np.where(cat_model.predict_proba(X_data)[:, 1] > 0.37, 1, 0)


# In[13]:


def print_bucket_name_objects():
    """
    Print bucket name and object list to check bucket name and objects.
    """
    for bucket in s3.buckets.all():
        if bucket.name == BUCKET_NAME:
            LOGGER.info(f"    {bucket.name}")
    for file in s3.Bucket(BUCKET_NAME).objects.all():
        LOGGER.info(f"    {file.key}")


def generate_precision_recall_chart(Xtest, ytest):
    """
    Generate precision recall chart to decide threshold.
    :param Xtest: Actual X values
    :param ytest: Actual y value
    :return: precision recall chart
    """
    probs = cat_model.predict_proba(Xtest)
    positive_probs = probs[:, 1]
    precision, recall, thresholds = precision_recall_curve(ytest, positive_probs)
    pr_auc = metrics.auc(recall, precision)
    plt.title("Precision-Recall vs Threshold Chart")
    plt.plot(thresholds, precision[:-1], "b--", label="Precision")
    plt.plot(thresholds, recall[:-1], "r--", label="Recall")
    plt.ylabel("Precision, Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="lower left")
    plt.ylim([0, 1])


# In[14]:


def upload_model_to_aws(s3_fs, cat_model):
    """
    Upload a model to AWS and print objects to upload a model
    :param s3_fs: s3fs.S3FileSystem(anon=False)
    :param cat_model: a prediction model
    """
    with s3_fs.open(f"{BUCKET_NAME}/{KEY_NAME_MODEL}", "wb") as file:
        joblib.dump(cat_model, file)
    for file in s3.Bucket(BUCKET_NAME).objects.all():
        LOGGER.info(f"    {file.key}")


def correlation_with_bankrupt_graph(importances, index, COLUMNS):
    """
    Generate a correlation graph to check correlations with target feature
    :param importances: correlations with target feautre
    :param index: index for the order of importances
    :param COLUMNS: column names
    """
    plt.figure(figsize=(15, 15))
    plt.title("correlation with Bankrupt")
    plt.barh(range(len(index)), importances[index], color="g", align="center")
    plt.yticks(range(len(index)), [COLUMNS[i] for i in index])
    plt.xlabel("Relative Importance")
    plt.show()


def save_variables_high_correlation_with_bank(importances, index, COLUMNS):
    """
    Save the input features that have higher correlations to select features
    :param importances: correlations with target feautre
    :param index: index for the order of importances
    :param COLUMNS: column names
    :return: high_corr_columns
    """
    high_corr_columns = []
    for i in range(0, len(index)):
        if np.abs(importances[i]) > 0.10:
            high_corr_columns.append(COLUMNS[i])
            print(COLUMNS[i])
    print(len(high_corr_columns))
    return high_corr_columns


def remove_dependent_variables(X, high_corr_columns):
    """
    Save independent features to remove dependent features
    :param X: X values
    :param high_corr_columns: features that have high correlation with target
    :return: selected_columns
    """
    corr = X.corr()
    high_corr_columns_bool = np.full((corr.shape[0],), True, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            if corr.iloc[i, j] >= 0.9:
                if high_corr_columns_bool[j]:
                    high_corr_columns_bool[j] = False
    selected_columns = X.columns[high_corr_columns_bool]

    print(len(selected_columns))

    return selected_columns


# In[15]:


def Kfold_split_5(X, y):
    """
    Save a training set and a test set to split a dataframe
    :param X: X features
    :param y: y features
    :return: Xtrain, Xtest, ytrain, ytest
    """
    skf = StratifiedKFold(n_splits=5, shuffle=False)
    for train_index, test_index in skf.split(X, y):
        print("TRAIN:", train_index, "TEST:", test_index)
        Xtrain, Xtest = X.iloc[train_index], X.iloc[test_index]
        ytrain, ytest = y.iloc[train_index], y.iloc[test_index]

    return (Xtrain, Xtest, ytrain, ytest)


# In[4]:

# In[16]:


if __name__ == "__main__":

    ### -----------------------------------------------------------------------
    ### --- Part 1: Connect to S3 Bucket on AWS
    ### -----------------------------------------------------------------------
    LOGGER.info("--- Part 1: Connect to S3 Bucket on AWS")

    # Connect to AWS
    s3 = boto3.resource("s3")

    # Sse AWS credentials to connect to file system, not as an anonymous user
    s3_fs = s3fs.S3FileSystem(anon=False)
    # Print bucket name and objects
    print_bucket_name_objects()

    ### -----------------------------------------------------------------------
    ### --- Part 2: Download CSV File from S3 Bucket
    ### -----------------------------------------------------------------------
    LOGGER.info("--- Part 2: Download CSV File from S3 Bucket")

    # Download the dataset in my S3 Bucket and save as df:
    LOGGER.info("    Download a dataset for bankruptcy")
    df = pd.read_csv(
        filepath_or_buffer=FILE_NAME, encoding="latin-1", nrows=6819, index_col=0
    )

    ### -----------------------------------------------------------------------
    ### --- Part 3: Check dataframe
    ### -----------------------------------------------------------------------
    LOGGER.info("--- Part 3: Check dataframe")

    # Check the shape of df
    print(df.shape)

    # Check if there are missing values
    print(df.info())

    ### -----------------------------------------------------------------------
    ### --- Part 4: Feature Engineering - Feature Selection
    ### -----------------------------------------------------------------------
    LOGGER.info("--- Part 4: Feature Engineering - Feature Selection")
    LOGGER.info("--- 4-1) Step1: Remove space from columns")

    # Remove spaces in the columns
    COLUMNS = df.columns.tolist()
    NO_SPACE_COLUMNS = [x.strip(" ") for x in COLUMNS]
    df.columns = NO_SPACE_COLUMNS

    # Split X features and y features to calculate correlations
    X = get_X(df)
    y = get_y(df)

    # Calculate correlations
    importances = X.apply(lambda x: x.corr(y))
    # Order by high correlations
    index = np.argsort(importances)
    print(importances[index])

    # Save columns to subtract later
    COLUMNS = df.columns[1:]

    # Generate a correlation graph to check
    correlation_with_bankrupt_graph(importances, index, COLUMNS)

    # Save the highly correlated features to get meaningful features
    high_corr_columns = save_variables_high_correlation_with_bank(
        importances, index, COLUMNS
    )
    LOGGER.info(
        "--- 4-1) Step2: Save the features that have a correlation higher with Bankrupt"
    )
    # Subtract important X features from all the X features
    X = df.loc[:, high_corr_columns]
    print(X.shape)

    LOGGER.info(
        "--- 4-1) Step3: Remove the features that have a correlation higher than 90% with other independent variables"
    )
    # Save the columns that are independent to remove dependent features
    selected_columns = remove_dependent_variables(X, remove_dependent_variables)

    # Now we only have 21 features
    LOGGER.info(
        "--- 4-1) Step4: Save all the selected variables to new_df and Bankrupt"
    )

    # Generate a new df with the important features
    # Save some features that I think it's important
    important_features = [
        "net income to total assets",
        "Cash flow to total assets",
        "tax Pre-net interest rate",
        "inventory and accounts receivable/net value",
    ]

    # Save y feature and all important X features
    new_df = pd.concat(
        [df["Bankrupt?"], df[important_features], X[selected_columns]], axis=1
    )
    print(new_df.columns)
    print(new_df.shape)

    # We have 25 features and target.

    ### -----------------------------------------------------------------------
    ### --- Part 5: Split the dataset
    ### -----------------------------------------------------------------------
    LOGGER.info("--- Part 5: Split the dataset")
    LOGGER.info("--- Split the dataset into training and test sets")

    # Split X features and y feature
    X = get_X(new_df)
    y = get_y(new_df)

    # Split the dataset to a training set and a test set
    Xtrain, Xtest, ytrain, ytest = Kfold_split_5(X, y)
    LOGGER.info("--- Export a training set and a test set as csv")

    # Export a training set and a test set
    Xtrain.to_csv(
        r"C:\Users\jeongjaehui\Documents\Stats_404\JEONG-JAEHEE\bankrupt_training_set.csv",
        index=False,
        header=True,
    )
    Xtest.to_csv(
        r"C:\Users\jeongjaehui\Documents\Stats_404\JEONG-JAEHEE\bankrupt_test_set.csv",
        index=False,
        header=True,
    )

    # Check the shape of df
    print(Xtrain.shape)
    print(Xtest.shape)

    LOGGER.info("--- Splite the data into training and valiation sets")
    # Split the training dataset to a training set and a validation set
    X_train, X_val, y_train, y_val = train_test_split(
        Xtrain, ytrain, test_size=0.1, random_state=1, stratify=ytrain, shuffle=True
    )

    ### -----------------------------------------------------------------------
    ### --- Part 6: Generates models
    ### -----------------------------------------------------------------------
    LOGGER.info("--- Part6: Part 6: Generates models")
    LOGGER.info("--- Generate a logistics regression model")

    # Generate a logistic regression model to predict bankrupt
    log_reg = LogisticRegression(class_weight="balanced", max_iter=10000)
    log_model = log_reg.fit(X_train, y_train)
    LOGGER.info("--- Print classification and confusion matrix with y_val")
    log_pred = log_model.predict(X_val)
    # Generate a classification report and a confusion matrix to check results
    print(classification_report(y_val, log_pred))
    print_confusion_matrix(y_val, log_pred)

    # check a classification & confusion matrix
    LOGGER.info("--- Print classification and confusion matrix with ytest")
    log_pred = log_model.predict(Xtest)
    print(classification_report(ytest, log_pred))
    print_confusion_matrix(ytest, log_pred)

    LOGGER.info("--- Generate a catboost classification model")

    # Define parameters for a catboost model
    params = {
        "iterations": 500,
        "loss_function": "Logloss",
        "depth": 6,
        "l2_leaf_reg": 1e-20,
        "eval_metric": "Accuracy",
        "leaf_estimation_iterations": 10,
        "use_best_model": True,
        "logging_level": "Silent",
        "random_seed": 42,
        "class_weights": (1, 30),
    }

    cat = CatBoostClassifier(**params)
    cat_model = cat.fit(X_train, y_train, eval_set=(X_val, y_val))

    LOGGER.info("--- Print classification and confusion matrix with y_val")
    cat_pred = cat_model.predict(X_val)
    print(classification_report(y_val, cat_pred))
    print_confusion_matrix(y_val, cat_pred)

    LOGGER.info("--- Print classification and confusion matrix with ytest")
    cat_pred = cat_model.predict(Xtest)
    print(classification_report(ytest, cat_pred))
    print_confusion_matrix(ytest, cat_pred)

    LOGGER.info(
        "--- Generate a precision recall curve to check if we can change the threshold to increase FalseNegative."
    )
    # Generate a precision recall curve to check recall by threshold
    generate_precision_recall_chart(Xtest, ytest)

    LOGGER.info("--- Change threshold and check the result")
    LOGGER.info("--- Print classification and confusion matrix with y_val")
    # Change the threshold to 0.37 to improve the prediction
    thre_37_preds_cat = cat_model_pred_prob_threshold_37(X_val)
    print(classification_report(y_val, thre_37_preds_cat))
    print_confusion_matrix(y_val, thre_37_preds_cat)
    LOGGER.info("--- Print classification and confusion matrix with ytest")
    thre_37_preds_cat = cat_model_pred_prob_threshold_37(Xtest)
    print(classification_report(ytest, thre_37_preds_cat))
    print_confusion_matrix(ytest, thre_37_preds_cat)

    ### -----------------------------------------------------------------------
    ### --- Part 7: Upload Model Object to S3 Bucket
    ### -----------------------------------------------------------------------
    LOGGER.info("--- Part 7: Upload Model Object to S3 Bucket")

    # Specify location and name of object to contain estimated model
    model_object_path = os.path.join(notebook_dir, "cat_model.joblib")
    # Save estimated model to specified location
    dump(cat_model, model_object_path)
    LOGGER.info("    Loading Catboost model object")
    cat_model = joblib.load("cat_model.joblib")

    # Specify name of file to be created on s3, to store this model object:
    upload_model_to_aws(s3_fs, cat_model)


# In[ ]:
