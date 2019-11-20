## Yuvoh - Data Scientist Challenge

The objective of this challenge is to predict price for Airbnb's listings in London. 

#### Data set characteristics
Number of observations : 83850

Number of attibute/features: 105

target : price

Missing data: 	1474446 (16.6%)

#### Feautures information (include target)

Numeric: 	33

Categorical: 	38

Boolean: 	8

Date 	3

URL 	5

#### Steps

1. Data exploratory analysis

Assess quality of the data, plot target distribution, missing data and relationship between target and fearures. Coding and furthe information can be found on [Data exploration Notebook](EDA.ipynb) 

2. Feature engineering

Missinga data imputation, encoding of categorical feautures and create new features. Coding and furthe information can be found on [Feature Engineering Notebook](Feature_Engineering.ipynb) 

3. Model training/testing

The number of features used in the modeling was 39. Boosting Gradient Regressor model was created and GridSearch was utilised the optimal hyper-parameters. Coding and furthe information can be found on [Modeling Notebook](Models.ipynb)


