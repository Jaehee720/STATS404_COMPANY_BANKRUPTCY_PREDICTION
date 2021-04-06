Business Use Case: Predict purchases in the E-commerce industry

1. Statement of Problem: 
    - Selling used car companies need to suggest the best price if the price is too cheap, they will not make the most profit. However, if the price is too expensive, coustomers will not buy a used car from their company and companies cannot turn theirs inventory quickly. It will eventually cause the used cars' actual value to drop.

2. Client: 
    - Selling used car companies such as CarMax, Carvana, and TrueCar

3. Key Business Question: 
    - Are there any aspects of evaluation used cars?

4. Data Source(s):
    - https://www.kaggle.com/avikasliwal/used-cars-price-prediction

5. Business impact of work: 
    - The used market in the U.S. is already estimated at 41 million units annually. https://www.cnbc.com/2020/10/15/used-car-boom-is-one-of-hottest-coronavirus-markets-for-consumers.html#:~:text=The%20used%20car%20market%20in,pandemic%20will%20continue%20to%20accelerate.
    - (Conservatively) Suppose that:
      * The average midsize sedan will loses $300 in value per month
      https://yourautoadvocate.com/guides/how-much-do-dealers-markup-used-cars/
      https://www.investopedia.com/articles/investing/091714/how-get-good-deal-used-car.asp
      https://www.tigerdroppings.com/rant/o-t-lounge/carmax---any-idea-on-how-much-they-mark-up-a-vehicle-after-they-take-it-in/57611344/
      * One location of a selling used car company store 1,000 vehicles.
      * 5% of not purchased used cars for 2 months
      * -> lose $360,000 annually only for one shop
    -> If we decrease the turn time to 1 month and not purchased cars by 1%, we reduce used car loses $36,000/year

6. How business will use (predicted) model to make decision(s):
    - When sell a used car, estimate the right price of the car
    - (Out of scope) Compare to other companies' suggested price and make the price a little bit cheaper
    - (Out of scope) Check the period of time that used cars have been in inventories, and use this variable as one of the predictor

7. Metric : 
    - to be monitored to see if model's promising