#!/usr/bin/env python
# coding: utf-8

# Standard libraries
import os
import logging
from io import BytesIO

# Third-party libraries
import joblib
import pandas as pd
import requests

# Define one logger for current file, per
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

if __name__ == "__main__":
    ### -----------------------------------------------------------------------
    ### --- Part 1: Get input
    ### -----------------------------------------------------------------------
    LOGGER.info("--- Part 1: Get input")

    ### -----------------------------------------------------------------------
    ### The input must be between 0 and 1 (inclusive)
    ### -----------------------------------------------------------------------
'''
    input = {'net income to total assets': 0.738,
        'Cash flow to total assets' : 0.631,
       'tax Pre-net interest rate':  0.797,
       'inventory and accounts receivable/net value': 0.41,
       'ROA(C) before interest and depreciation before interest': 0.419,
       'operating gross margin': 0.599, 'tax rate (A)': 0.032,
       'per Net Share Value (B)': 0.160,
       'Persistent EPS in the Last Four Seasons': 0.189,
       'Operating Profit Per Share (Yuan)': 0.087, 'debt ratio %': 0.187,
       'net worth/assets': 0.813,
       'borrowing dependency': 0.390, 'working capital to total assets': 0.752,
       'cash / total assets': 0.048, 'current liability to assets': 0.144,
       'working capital/equity': 0.726, 'current liability/equity': 0.343,
       'Retained Earnings/Total assets': 0.904, 'total expense /assets': 0.050,
       'equity to long-term liability': 0.131, 'CFO to ASSETS': 0.556,
       'current liabilities to current assets': 0.060,
       'one if total liabilities exceeds total assets zero otherwise': 0.027,
       'Net income to stockholder\'s Equity': 0.826}
'''
    input = { 'net income to total assets': 0.8100831300180976,
     'Cash flow to total assets': 0.6503399574931823,
     'tax Pre-net interest rate': 0.7972097625333621,
     'inventory and accounts receivable/net value': 0.40227620612972925,
     'ROA(C) before interest and depreciation before interest': 0.5080692807159637,
     'operating gross margin': 0.6082573405235892,
     'tax rate (A)': 0.11777818468140971,
     'per Net Share Value (B)': 0.19166887253990908,
     'Persistent EPS in the Last Four Seasons': 0.23014621858970755,
     'Operating Profit Per Share (Yuan)': 0.10981541631354354,
     'debt ratio %': 0.11071438040147158,
     'net worth/assets': 0.8892856195985285,
     'borrowing dependency': 0.37412935472638476,
     'working capital to total assets': 0.8162069634997345,
     'cash / total assets': 0.12664023276893469,
     'current liability to assets': 0.0888870143456048,
     'working capital/equity': 0.7361304170361399,
     'current liability/equity': 0.33103098724531244,
     'Retained Earnings/Total assets': 0.9357492100831155,
     'total expense /assets': 0.028494877082716093,
     'equity to long-term liability': 0.11514899413627973,
     'CFO to ASSETS': 0.5946487243252321,
     'current liabilities to current assets': 0.030541663029235132,
     'one if total liabilities exceeds total assets zero otherwise': 0.00030307622367025305,
     "Net income to stockholder's Equity": 0.8408819416444057}

    ### -----------------------------------------------------------------------
    ### --- Part 2: Load model from AWS
    ### -----------------------------------------------------------------------
    LOGGER.info("--- Part 2: Load model from AWS")

    MODEL_URL="https://stats404-project-jaehee.s3-us-west-1.amazonaws.com/log_model.joblib"
    mymodel_response = requests.get(MODEL_URL)
    LOGGER.info(f"--- The response code for the model request is: {mymodel_response.status_code}")
    mymodel = BytesIO(mymodel_response.content)
    log_model = joblib.load(mymodel)

    ### -----------------------------------------------------------------------
    ### --- Part 3: Input data into model
    ### -----------------------------------------------------------------------
    LOGGER.info("--- Part 3: Input data into model")
    input_df = pd.DataFrame(input, index=[0])
    log_pred = log_model.predict(input_df)

    ### -----------------------------------------------------------------------
    ### --- Part 4: Get the result
    ### -----------------------------------------------------------------------
    print(log_pred)
    LOGGER.info("--- Part 4: Get the result")
    if log_pred == 1:
        print("This company may face bankruptcy. Please do not lend to this company.")
    if log_pred == 0:
        print("This company is stable. Please lend to this company.")