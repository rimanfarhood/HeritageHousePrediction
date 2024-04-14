# Heritage Housing Issue

## Business Case
As a good friend, you are requested by your friend, who has received an inheritance from a deceased great-grandfather located in Ames, Iowa, to  help in maximising the sales price for the inherited properties.

Although your friend has an excellent understanding of property prices in her own state and residential area, she fears that basing her estimates for property worth on her current knowledge might lead to inaccurate appraisals. What makes a house desirable and valuable where she comes from might not be the same in Ames, Iowa. She found a public dataset with house prices for Ames, Iowa, and will provide you with that.

### Business Requirements
* 1 - The client is interested in discovering how the house attributes correlate with the sale price. Therefore, the client expects data visualisations of the correlated variables against the sale price to show that.
* 2 - The client is interested in predicting the house sale price from her four inherited houses and any other house in Ames, Iowa.

---
## Dataset Content
* The dataset is sourced from [Kaggle](https://www.kaggle.com/codeinstitute/housing-prices-data). We then created a fictitious user story where predictive analytics can be applied in a real project in the workplace. 
* The dataset has almost 1.5 thousand rows and represents housing records from Ames, Iowa, indicating house profile (Floor Area, Basement, Garage, Kitchen, Lot, Porch, Wood Deck, Year Built) and its respective sale price for houses built between 1872 and 2010.

|Variable|Meaning|Units|
|:----|:----|:----|
|1stFlrSF|First Floor square feet|334 - 4692|
|2ndFlrSF|Second-floor square feet|0 - 2065|
|BedroomAbvGr|Bedrooms above grade (does NOT include basement bedrooms)|0 - 8|
|BsmtExposure|Refers to walkout or garden level walls|Gd: Good Exposure; Av: Average Exposure; Mn: Minimum Exposure; No: No Exposure; None: No Basement|
|BsmtFinType1|Rating of basement finished area|GLQ: Good Living Quarters; ALQ: Average Living Quarters; BLQ: Below Average Living Quarters; Rec: Average Rec Room; LwQ: Low Quality; Unf: Unfinshed; None: No Basement|
|BsmtFinSF1|Type 1 finished square feet|0 - 5644|
|BsmtUnfSF|Unfinished square feet of basement area|0 - 2336|
|TotalBsmtSF|Total square feet of basement area|0 - 6110|
|GarageArea|Size of garage in square feet|0 - 1418|
|GarageFinish|Interior finish of the garage|Fin: Finished; RFn: Rough Finished; Unf: Unfinished; None: No Garage|
|GarageYrBlt|Year garage was built|1900 - 2010|
|GrLivArea|Above grade (ground) living area square feet|334 - 5642|
|KitchenQual|Kitchen quality|Ex: Excellent; Gd: Good; TA: Typical/Average; Fa: Fair; Po: Poor|
|LotArea| Lot size in square feet|1300 - 215245|
|LotFrontage| Linear feet of street connected to property|21 - 313|
|MasVnrArea|Masonry veneer area in square feet|0 - 1600|
|EnclosedPorch|Enclosed porch area in square feet|0 - 286|
|OpenPorchSF|Open porch area in square feet|0 - 547|
|OverallCond|Rates the overall condition of the house|10: Very Excellent; 9: Excellent; 8: Very Good; 7: Good; 6: Above Average; 5: Average; 4: Below Average; 3: Fair; 2: Poor; 1: Very Poor|
|OverallQual|Rates the overall material and finish of the house|10: Very Excellent; 9: Excellent; 8: Very Good; 7: Good; 6: Above Average; 5: Average; 4: Below Average; 3: Fair; 2: Poor; 1: Very Poor|
|WoodDeckSF|Wood deck area in square feet|0 - 736|
|YearBuilt|Original construction date|1872 - 2010|
|YearRemodAdd|Remodel date (same as construction date if no remodelling or additions)|1950 - 2010|
|SalePrice|Sale Price|34900 - 755000|

---

## Hypothesis and how to validate?

**Hypothesis**

According to my hypothesis there is several factors that impact the sale price of a house. After some researching in the field of real estate and my prior knowledge I have come to this conclusion. 

External factors:
 - The location is the decisive factor of the sale price, and  accessibility to public transportation, grocery stores and school/preschool also impacts the value.

Since we're trying to predict the sale price of houses in Ames in Iowa the location is not included in our dataset and we can't validate that.

Beside external factors these factors that have the greatest impact on price:
 - The overall quality.
 - The size of the house, total square feet.
 - Number of bedrooms and bathrooms.
 - The condition of kitchen and bathroom, newly renovated has a significant impact on sale price. 
 - The Year it was built.

**Validation**
 - We can validate the hypothesis by conducting a Correlation  Study. 

## The rationale to map the business requirements to the Data Visualizations and ML tasks

**Business Requirement 1:** 

 - Inspect the House Records data and conduct a Correlation study both spearman and pearson. 

 - Visualize the correlation levels by Sale Price
   - Plot the highest correlated variables against Sale Price to summarize the insights. 


**Business Requirement 2:**

 - Deliver an ML model with the capability of predicting the sale price of the four inherited houses and any house in 
    Ames, Iowa with at least 75% accuracy.
 
 - Conventional Machine Learning to map the relationships between the features and the target. This will be Involving:

    - Applying a range of machine learning algorithms to capture the best model for our purpose.

    - Once that is done we will conduct extensive hyperparameter optimization to fine-tune the performance, ensuring optimal results.

## ML Business Case

**1. What are the business requirements?**

- The client is interested in discovering how house attributes correlate with sale prices. Therefore, the client expects data visualizations of the correlated variables against the sale price.

- The client is interested in predicting the house sale prices from her 4 inherited houses, and any other house in Ames, Iowa.

**2. Is there any business requirement that can be answered with conventional data analysis?**

- Yes, we can use conventional data analysis to investigate how house attributes are correlated with the sale prices.

**3. Does the client need a dashboard or an API endpoint?**

- The client needs a dashboard.

**4. What does the client consider as a successful project outcome?**

- A study showing the most relevant variables correlated to sale price.

- Also, a capability to predict the sale price for the 4 inherited houses, as well as any other house in Ames, Iowa.

**5. Can you break down the project into Epics and User Stories?**

- Information gathering and data collection.

- Data visualization, cleaning, and preparation.

- Model training, optimization and validation.

- Dashboard planning, designing, and development.

- Dashboard deployment and release.

Here you can find my:
- [Kanban board](https://github.com/rimanfarhood/HeritageHousePrediction/projects?query=is%3Aopen)
- [Epics](https://github.com/rimanfarhood/HeritageHousePrediction/milestones)

**6. Ethical or Privacy concerns?**

- No. The client found a public dataset.

**7. Does the data suggest a particular model?**

- The data suggests a regressor where the target is the sale price.

**8. What are the model's inputs and intended outputs?**

- The inputs are house attribute information and the output is the predicted sale price.


**9. What are the criteria for the performance goal of the predictions?**

- We agreed with the client an R2 score of at least 0.75 on the train set as well as on the test set.

**10. How will the client benefit?**

- The client will maximize the sales price for the inherited properties.


## Dashboard 

**Client Expectations:**

- A project summary page, displaying dataset summary and the requirements.
- A page listing the findings related to which features have the strongest correlation to the house sale price.
- A page 
   - Displaying the 4 houses' attributes and their respective predicted sale price.
   - Displaying a message informing the summed predicted price for all 4 inherited houses. 
   - Interactive input widgets that allows a user to provide real-time house data to predict the sale price.
- A page indicating hypothesis(es) and validation.
- A technical page displaying model performance. If a ML pipeline is deployed, Display the pipeline steps.

**Page 1 - Project Summary**

 - Summary of the dataset and the client's  requirements.

**Page 2 - Summary of Correlation Study**

 - Listing and Visualizing of the strongest correlated features to the Sale Price. 

**Page 3 - Housing Prediction Page**

 - The 4 houses' attributes and their respective predicted sale price as well as the summed predicted price for all 4 inherited houses. 

   - Interactive input widgets to provide with real-time house data to predict the sale price.
 

**Page 4 - Hypothesis and Validation Page**

   - The Hypothesis
   - And Validation 

**Page 5 - Model Performance Page**

   - ML Pipeline
   - Feature Importance Plot
   - Model Performance Metrics


## Fixed Bugs

* Problem install dependencies due to python version, to resolve issue I changed python version
* Had a issue with matplotlib plots not being displayed, resolved with matplotlib inline.
* Slug size to large, resolved by adding files to slugignore

## Unfixed Bugs

* 'None' displaying on the prediction page and I cant seem to understand where from.

![bug](https://github.com/rimanfarhood/HeritageHousePrediction/tree/main/media)

## Deployment
### Heroku

* The App live link is: [Heritage Housing](https://heritagehousing-1493ef292e3f.herokuapp.com/)
* Set the runtime.txt Python version to a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
* The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click the button Open App on the top of the page to access your App.
6. If the slug size is too large then add large files not required for the app to the .slugignore file.

## Main Data Analysis and Machine Learning Libraries

Most Frequently Used and Example of usage

   - Pandas
      - Reading files mostly
   - Seaborn
      - Mostly plots for example correlation plots.
   - Feature Engine 
      - Imputation methods
      - Ordinal Encoder
   - Matplotlib
      - Plots as well mostly
   - Scikit-learn
      - Pipeline
      - ML Algorithms

## Credits 

- Grammar check [Reverso](https://www.reverso.net/spell-checker/english-spelling-grammar/)
- Churnometer Walkthrough 
- Lms content
- Math videos 
   - [freecodecamp statistcs](https://youtu.be/xxpc-HPKN28?si=xgXopgHtDZF6TdDc)
   - [Crashcourse Regression](https://youtu.be/WWqE7YHR4Jc?si=39Ogjh28yHVWcYJg)
   - [Simplilearn Correlation](https://youtu.be/PEfQCv9nvSo?si=toS89kx4xyQ29pd4)
   - [Business Science ppscore](https://youtu.be/D5cGYBop8zw?si=wVrKQEYuR0A_gJ8L)


## Acknowledgements (optional)

I want to thank me for believing in me and not giving up, this was so hard.

