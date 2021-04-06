Business Use Case: Company bankruptcy prediction 

1. Statement of Problem: 
    - Banks will lose money if they loan for the businesses that will face bankruptcy in the future. 

2. Client: 
    - Banks are in Taiwan

3. Key Business Question: 
    - Are the companies stable that want to get loans from banks?

4. Data Source(s):
    - https://www.kaggle.com/fedesoriano/company-bankruptcy-prediction

5. Business impact of work: 
    - Bankruptcies in Taiwan averaged 2760.21 Companies from 2000 until 2020 per month. 
    https://tradingeconomics.com/taiwan/bankruptcies#:~:text=Bankruptcies%20in%20Taiwan%20averaged%202760.21,Companies%20in%20February%20of%202018.

    - (Conservatively) Suppose that:
        * Banks' loans in 2009 was $18,608,000,000 in Taiwan.
        https://www.cbc.gov.tw/en/public/Attachment/062811551271.PDF
        * Ratio of corporate loans to total loans of Taiwan's domestic banks in 2013 was 44.65%
        https://www.statista.com/statistics/1079700/taiwan-corporate-loans-to-total-loans-of-domestic-banks/
        * Current Taiwan Semiconductor Probability Of Bankruptcy is 23%. Since, we have more industries other than semiconductor, we will assume that probability of bankruptcy is 18%.
        https://www.macroaxis.com/invest/ratio/TSM/Probability-Of-Bankruptcy
        * -> Cost to Banks in Taiwan: $18,608,000,000(banks' loans) * 0.4465(percent of corporate loans) * 0.23 (probability of bankruptcy): $1,910,948,560 annually
        * -> An average bank losses $33,233,888 annually/37 = $51,647,258/year
    -> If we can reduce to give a loan to unstable companies by 1%, we can reduce banks expenses by $516,472.6 /year.

6. How business will use (predicted) model to make decision(s):
    - Banks will use the model to check if companies are expected to face bankruptcy or not, if yes, then the banks will not give loans to those companies.

    - (Out of scope) Recommendation engine to recommend better spec for business

7. Metric : 
    - to be monitored to see if the model's promising
    
8. Methodology :
    (Write-up a summary of what you did and why in "Methodology" section of README, referencing 3+ cells, figures and/or tables)
    - Since we want to predict if companies will face bankruptcy or not, the outcome is a binary value. Therefore, I wanted to use a logistic regression model.
    
    
  ![Screenshot](log.png)



  - But, only 6% of the predictions for the class 1, bankrupt companies, are correctly classified. Therefore, I tried to improve the model using catboost. CatBoost is a machine learning algorithm that uses gradient boosting on decision trees. I'd like to use this method because I wanted to use decision trees. When I created a model using cat boost, the prediction increased significantly.
  
  
  ![Screenshot](catboost.png)
  
  
  - I checked which features are important. There were only 12 features that have importances out of 95 features.
  
  ![Screenshot](importance.png)
  
  
However, when I used a different random_state other than 1, the importance changed a lot. There were 36 important features. Therefore, I decided to keep all the features.