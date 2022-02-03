# King-County-Houses-Insights

Formulation and validation of hypothesis about housing market in King County, WA, through exploratory data analysis.


## Business Problem
House Rocket is a digital platform that uses technology to simplify the sale
and purchase of real estate in King County, Washington State.

The company has a simple business model that is basically to buy good houses
in great location at below-market prices and then resell them later at higher
prices to maximize profit and therefore maximize revenue.

The company has a dataset that contains the house sale prices for King County,
which includes homes sold between May 2014 and May 2015. A report with the best
business opportunities should be submitted that considers the following topics:
- Which houses should the company buy and at what price?
- Once the house owned by the company, what is the best time to sell them and 
what would be the sale price?
- Should the company do a renovation to raise the sale price? What would be the 
suggestions for changes? What is the price increase given for each refurbishment
option?

All the context about this project is completely fictitious, including company, CEO 
and business issues. 

This is a project provided by the DS community and can be access by this link https://sejaumdatascientist.com/os-5-projetos-de-data-science-que-fara-o-recrutador-olhar-para-voce/

## Business Assumptions
The assumptions about the business problem is as follow:
- The costs of maintaining the business are not considered when calculating the company's profit.
- Seasons interfere in houses price.
- Renovations can only be considered to add or remove rooms.
- House conditions are not altered by renovations.
- Costs for renovations are not considered when calculating the profit form the sale of houses.

## Solution Strategy
This project was developed based on the CRISP-DS (Cross-Industry Standard Process - Data Science, 
a.k.a. CRISP-DM) project management method, applied according to the steps bellow:
- **Data Description:** Use descriptive statistic to understand data and identify possible errors and data outside the 
scope of project.
- **Data Filtering:** Filter rows and delete columns that are not relevant or are not part of the project scope.
- **Data Transformation:** Transform attribute types, so they can be used in the project scope.
- **Feature Engineering:** Derive new attributes based on the original variables to better describe the sales that 
will be explored.
- **Exploratory Data Analysis:** Explore the data to generate a list of hypothesis that will be the basis for business 
insights and better understand the impact of insights on the answers to business questions.
- **Deploy Model to Production:** Publish a dashboard with the analysis and conclusions in a cloud environment so that 
other people or services can use the results to improve their business decisions.

## Main Data Insights
Considering the insight that brings the most value to the business, follow the result from data exploratory analysis.

**Hypothesis 01:** Houses with a water view are **30%** more expensive, on average.

**False:** Houses with a water view are **300%** more expensive, on average.

**Hypothesis 02**: Houses with a construction date less than 1955 are **50%** cheaper, in average.

**False:** The construction date has low influence on the price of houses.

**Hypothesis 03:** Houses without a basement have a lot size **50%** larger than those with a basement.

**False:** Houses without a basement have a lot size around **20%** smaller than thoses with a basement.

**Hypothesis 04:** The YoY (Year over Year) house price growth is **10%**.

**False:** The YoY houses price growth is less than **1%**.

**Hypothesis 05:** Houses with 3 bathrooms have a MoM (Month over Month) growth of **15%**.

**False:** Number of bathrooms don't have relationship with time.

### Hypothesis summary
| Hypothesis | Results | Relevance |
| ---------- | ------- | --------- |
|     H1     |  False  |    High   |
|     H2     |  False  |    Low    |
|     H3     |  False  |   Medium  |
|     H4     |  False  |    Low    |
|     H5     |  False  |    Low    |



More about the data exploratoty analysis can be check in 
https://king-county-houses-analytics.herokuapp.com/

## Machine Learning Model Applied
This project did not require a machine learning model.

## Machine Learning Model Performance
No machine learning performance was achieved.

## Business Results
- Validation of valuable insights for assist decision-making by business team.
- Model to selection bests properties to be bought by company depending on following conditions
choose by data orientation:
  - Houses there are under the median price from region.
  - Houses there are in good condition.
  - Houses that have water view.
- Model to determine a sell price aiming a 10-30% profit depending on following conditions:
  - 10% increase price if the properties is over the median price from region and season.
  - 30% increase price if the properties is under the median price from region and season.
- Model to select best refurbishment options and determine the increase price resulted by them. 
Options choose is:
  - Basement
  - Bathrooms
  - Bedrooms
  
## Conclusions

This project has the goal to generate insights to assist the decision-make by business team by a 
study and understanding of data about properties from King County Seattle. A additional study including
price per size lot should be considered. A multivariate analysis show that most correlations with price are
'sqft_living' and 'bathrroms' and shoulb considered in bought and refurbishment recommendations model.
Others conclusions can be made after the data study:
  - Properties that have pass for some renovate have 43% more value, on average.
  - There are much more houses that have never pass by a renovation than renovated houses (no/yes: 20699/914).

The dashboard can be consulted by the link bellow:
https://king-county-houses-analytics.herokuapp.com/


## Lessons Learned
1. Python data manipulation libraries: pandas, numpy.
2. Python data visualization libraries: seaborn, matplotlib, plotly.express.
3. Statistic libraries: scipy.stats.
4. Git/Github.
5. Crisp-DM methods.
6. Documentation format.
7. Data analysis fundamentals.
8. OOP fundamentals.
9. Hypothesis and insights.
10. Deploy model in Heroku
11. multipage functionality of Streamlit

## Next Steps
1. Improvements to first-cycle-notebook or nexts:
   1. Add conclusions on exploratory data analysis notebook.
   2. Add seção com todas os tipos variaveis categoricas (var_cat.unique()) e explciar cada uma;
   3. Add geopy.geocoders variables;
   4. Add pickes for each notebook section;
   5. Add data copy() for each notebook section;
   6. Review following Ds em Produçao format;
   7. Q&R section with CEO questions;
   8. Hypothesis without percentage variables, only text;
   9. Add hypothesis about seasons considering price and quantity of properties;
   10. Add recommendations sections (buy, sell and refurbishment);
   11. Considering all the attributes for multivariate analysis, with no data drop.
   12. Categorical attributes show technical mistakes, code should be review.

2. Apply dashboard improvements:
    - Add github link in navigation bar;
    - Upgrade to multipages for each section;
    - Add map for each recommendation;
      - Degign on ETl (POO)

3. Improvements in hypothesis:
   - Hypothesis map;
   - Hypothesis considering price by lot size;

4. Documentation inprovements:
   - Add images in readme.
   - Add dashboard link in conclusion.
   - [ ] checkbox test
   - [x] checked checkbox
   
