Business Use Case: Company bankruptcy prediction 

1. What business questions is:
    - Statement of Problem: 
         * Banks will lose money if they lend to businesses that will face bankruptcy in the future 
    - Client: 
        * Banks are based in Taiwan 
    - Key Business Question:
         * Are the companies that want to get loans from banks stable enough to be worth the risk? 


2. What the data set is:
    - The data were collected from the Taiwan Economic Journal for the years 1999 to 2009.
    - Data source:
        * Kaggle
        * https://www.kaggle.com/fedesoriano/company-bankruptcy-prediction
    - The number of observation: 6819
    - Features:
        * There are 96 features, 1 output feature, and 95 input features. All the input features are expressed as a rate. 
    - Bankruptcy rate:
        * 6599 companies are stable and 220 companies faced bankruptcy. The percentage of bankruptcy is 3.23%.


3. Business impact of work: 
    - (Conservatively) Suppose that:
        * Banks' loans in 2009 was $18,608,000,000 in Taiwan.
            - https://www.cbc.gov.tw/en/public/Attachment/062811551271.PDF
        * Ratio of corporate loans to total loans of Taiwan's domestic banks in 2013 was 44.65%
        * The total loans taken from all banks in Taiwan in 2009 by corporate entities: $18,608,000,000*44.65% = $8,308,472,000
            - https://www.statista.com/statistics/1079700/taiwan-corporate-loans-to-total-loans-of-domestic-banks/
        * Probability of bankruptcy is 3.23%.
        * -> Cost to Banks in Taiwan: $18,608,000,000(banks' loans) * 0.4465(percent of corporate loans) * 0.0323 (probability of bankruptcy): $268,363,646 annually
        * -> An average bank losses $268,363,646 annually/37 = $7,253,072/year
    - If we reduce the total amount of loans given to all unstable companies by 1%, we would be saving $2,245,533.


4. Methodology:  
    - What data processing steps you took and why, including how you handled missing data
        1. Check data info
            - There is no missing data. But, if there were, I would have checked if the company went bankrupt or not because there were only a few unstable companies. Therefore, if the company faced bankruptcy, I would have tried to keep the data. For example, I will perform feature selection without the observation that has a missing value. After feature selection, I will check if there is the column that has a missing value. If there is not, I will add the observation.
        2. Data cleaning
            - I deleted all the extra spaces on the columns.
        3. Feature selection
            - There are 95 input features, but I only have 6819 observations which is a relatively small number comparing to the number of features. Therefore, I tried to reduce the number of input features using the filter method to avoid overfitting and reduce the complexity of a model.
                1. Check the correlations between the target variable and the input variables. 
                2. Save the input features that have correlations higher than 0.1. (I saved 32 input features.)
                3. Check the dependence between input features.
                4. If a correlation is higher than 0.9, drop the features. (I dropped 9 input features.)
                5. Check all the columns that we have now and add some input features that you dropped if it seems very important to generate a model. (I added 4 features.)
            - I used 25 input features to generate a model.
    - Any assumptions/simplifications made
        1. If a correlation is low between the target feature and input feature, it has a minor effect on creating a model.
        2. If a feature is highly correlated to another feature, one feature can represent the other feature.
        3. We kept all the important input features that could affect the prediction.
    - What modeling approach implemented and why -- for your one final model
        - I tried to generate a model using many different classification methods such as catboost, xgboost, random forest, and logistic regression. Without feature selection, catboost classifier worked the best. But, this model was highly overfitted. Therefore, I performed feature selection to avoid overfitting. Logistic regression is a very simple classification method, but also extremely powerful when it comes to provided coefficients and p-values for each predictor variable. There could be better machine learning classification methods that can have higher prediction results. However, in my case, the model using logistic regression have the best prediction results. Even if, it did not have the best results, since these results can have a direct impact on the client's business, it was a lot more important to see the statistical significance for each variable and their coefficient. Furthermore, a model using logistic regression is more interpretable which means that it is easier for the clients to understand.

5. Results:
    1. What key findings and recommendations are
        - Using the 0.5 threshold is reasonable to determine if a business will go bankrupt or not because the recall value was high with the threshold. However, when I tried to make the threshold lower, even though the recall value did not increase that much, the precision decreased a lot which means banks will miss a lot of customers. Therefore, I recommend if the probability is lower than 0.5, banks can lend to the company.
    2. Give very concrete advice on how a business would use the model -- and model output to answer the business question
        - Before banks proceed with a loan program for companies, banks should ask the companies to provide their financial statement. Banks will input the data based on the financial statement into the model. If the probability to go bankrupt is equal to or higher than 0.5, the banks shouldn't lend to the companies.
    3. Additionally, give very specific advice for how business should act on model output from your example input
        - If the probability is lower than 50%, the companies are stable. Therefore, banks can proceed with a loan for the companies.
        - For a company with a high probability, see if banks will be able to sell the loan to a bigger bank.
            * For small banks, after they provide a loan to customers, sometimes they sell the loan to a bigger bank because they prefer having money now instead of collecting interest for 30 years.    
        - Monitor the metric to see if the model's promising
            * Company bankruptcy does not happen often, so we can monitor the metric monthly.


6. What potential next steps or further research topics are
    1. Use SMOTE classification with all 95 features
    2. Use PCA for feature selection
    3. Use feature importance using random forest for feature selection
    4. Use catboost classifier for feature selection, and then use PCA to avoid overfitting. After that, use catboost again to train a model with the PCA transformed data.


7. Design
    1. Input and output spec for my project    
    
    input = {
    "net income": [738000],\
    "Cash flow": [631000],\
    "total assets": [1000000],\
    "tax Pre-net interest rate": [0.797],\
    "inventory and accounts receivable/net value": [0.41],\
    "ROA(C) before interest and depreciation before interest": [0.419],\
    "operating gross margin": [0.599],\
    "tax rate (A)": [0.032],\
    "per Net Share Value (B)": [0.160],\
    "Persistent EPS in the Last Four Seasons": [0.189],\
    "Operating Profit Per Share (Yuan)": [0.087],\
    "debt ratio %": [0.187],\
    "net worth/assets": [0.813],\
    "borrowing dependency": [0.390],\
    "working capital to total assets": [0.752],\
    "cash / total assets": [0.048],\
    "current liability to assets": [0.144],\
    "working capital/equity": [0.726],\
    "current liability/equity": [0.343],\
    "Retained Earnings/Total assets": [0.904],\
    "total expense /assets": [0.050],\
    "equity to long-term liability": [0.131],\
    "CFO to ASSETS": [0.556],\
    "current liabilities to current assets": [0.060],\
    "one if total liabilities exceeds total assets zero otherwise": [0.027],\
    "Net income to stockholder's Equity": [0.826],\
    }
        
    output: The probability to go bankrupt is: [0.85276269]. This company may face bankruptcy. Please do not lend to this company.

    2. Architecture diagram

![alt tag](https://github.com/UCLA-Stats-404-W21/JEONG-JAEHEE/blob/feature/final/pictures/Architecture%20Diagram_final.png?raw=true)


8. Instructions for running code:
    - OS type and version: macOS Big Sur/Version 11.2.3
    - Anaconda and Python versions: conda 4.9.2/ python 3.9.2
    - Instructions to run the code: 
        - Step1: conda install --file requirements.txt
        - Step2: python main.py
    - Instructions to accept input: 
        - On part 1, there is a input dictionary in the main.py, python script, under if__name__ == "__main__".
        - Net come, cash flow and total assets need a positive real number.
        - Net come and cash flow should be smaller than total assets.
        - All the other input features must have a number between 0 to 1