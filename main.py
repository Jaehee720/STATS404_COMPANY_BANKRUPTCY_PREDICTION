# Standard libraries
import logging

# Third-party libraries
import joblib
from clean_input.feature_engineering import *
import pandas as pd
import s3fs

# Define one logger for current file, per
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

if __name__ == "__main__":
    ### -----------------------------------------------------------------------
    ### --- Part 1: Get input
    ### -----------------------------------------------------------------------
    LOGGER.info("--- Part 1: Get input")

    input = {
        "net income": [738000],
        "Cash flow": [631000],
        "total assets": [1000000],
        "tax Pre-net interest rate": [0.797],
        "inventory and accounts receivable/net value": [0.41],
        "ROA(C) before interest and depreciation before interest": [0.419],
        "operating gross margin": [0.599],
        "tax rate (A)": [0.032],
        "per Net Share Value (B)": [0.160],
        "Persistent EPS in the Last Four Seasons": [0.189],
        "Operating Profit Per Share (Yuan)": [0.087],
        "debt ratio %": [0.187],
        "net worth/assets": [0.813],
        "borrowing dependency": [0.390],
        "working capital to total assets": [0.752],
        "cash / total assets": [0.048],
        "current liability to assets": [0.144],
        "working capital/equity": [0.726],
        "current liability/equity": [0.343],
        "Retained Earnings/Total assets": [0.904],
        "total expense /assets": [0.050],
        "equity to long-term liability": [0.131],
        "CFO to ASSETS": [0.556],
        "current liabilities to current assets": [0.060],
        "one if total liabilities exceeds total assets zero otherwise": [0.027],
        "Net income to stockholder's Equity": [0.826],
    }

    ### -----------------------------------------------------------------------
    ### --- Part 2: Load model from AWS
    ### -----------------------------------------------------------------------
    LOGGER.info("--- Part 2: Load model from AWS")

    # load model from AWS and save into log_model object
    s3_fs = s3fs.S3FileSystem(anon=True)
    MODEL_URL = "s3://stats404-project-jaehee/log_model.joblib"
    with s3_fs.open(MODEL_URL, "rb") as file:
        log_model = joblib.load(file)

    ### -----------------------------------------------------------------------
    ### --- Part 3: Feature engineering
    ### -----------------------------------------------------------------------
    LOGGER.info("--- Part 3: Feature engineering")

    # save input as df
    input_df = pd.DataFrame(input, index=[0])

    # feature engineering
    feature_engineering_input_df = feature_engineering(input_df)

    ### -----------------------------------------------------------------------
    ### --- Part 4: Input data into model
    ### -----------------------------------------------------------------------
    LOGGER.info("--- Part 4: Input data into model")

    # run model with the input and get prediction
    log_pred = log_model.predict(feature_engineering_input_df)

    # save probability
    log_pred_prob = log_model.predict_proba(feature_engineering_input_df)[:, 1]

    ### -----------------------------------------------------------------------
    ### --- Part 5: Get the result
    ### -----------------------------------------------------------------------
    LOGGER.info("--- Part 5: Get the result")
    LOGGER.info(f"--- The proability to go bankrupt is: {log_pred_prob}.")

    # print the result
    if log_pred == 1:
        print("This company may face bankruptcy. Please do not lend to this company.")
    if log_pred == 0:
        print("This company is stable. Please lend to this company.")
