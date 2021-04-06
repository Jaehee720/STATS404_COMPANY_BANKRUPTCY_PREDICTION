Business Use Case: Predict purchases in the E-commerce industry

1. Statement of Problem: 
    - In the E-commerce industry, customers add products to cart, but they do not check out.

2. Client: 
    - E-commerce companies

3. Key Business Question: 
    - Are there any aspects of not purchased products that may be foreseen and encouraged people to buy the products? 

4. Data Source(s):
    - https://www.kaggle.com/mkechinov/ecommerce-behavior-data-from-multi-category-store

5. Business impact of work: 
    - There are more than 1.3 million companies that use e-commerce in the United States.
    - (Conservatively) Suppose that:
      * 38% of added products are purchased
      * Net income is $30/product
      * One e-commerce company has 100,000 users
      * 1% of total users adds to cart
    -> If we can increase the purchased rate by 1%, net income can be increased by $30,000/year

6. How business will use (predicted) model to make decision(s):
    - When emailing users to encourage them to purchase products, estimated probability of not purchasing products in the cart based on categories and event date
    - For products with high probability, see if recommending similar products can result in better outcomes
    - (Out of scope) Recommendation engine to recommend better spec for purchasing products

7. Metric : 
    - to be monitored to see if model's promising