# NYC Property Sales

### by: Aviv Farag, Joseph Logan, Abdulaziz Alquzi

---

## Table of Contents
 * [Abstract](#abstract)
 * [Python Packages](#python-packages)
 * [Functions](#functions)
 * [Setup and running the code](#setup-and-running-the-code)
 * [Acknowledgements](#acknowledgements)
 

---

## Abstract: 
We are data science consultants who are contracted by property management investors in New York City. Their company, supported by investors, wants to buy residential real estate in NYC at the cheapest price possible, renovate, then resell within a year. The renovation analysis is outside the scope of this project, but they want a baseline model that can predict the price of residential real-estate in order to :

Identify potential undervalued listed properties to buy
Predict market price when it’s time to sell in order to sell quickly while maximizing return on investment
Because the want to renovate and sell the properties quickly, they want less than 10 residential units, and properties less than 5 million each but are at least ten thousand.

---

## Python Packages:
1. pandas <br>
 `import pandas as pd`
 
1. numpy <br>
`import numpy as np`

1. matplotlib.pyplot <br>
`import matplotlib.pyplot as plt`

1. joblib <br>
`import joblib`

1. seaborn <br>
`import seaborn as sns`

1. scipy.stats.randint <br>
`from scipy.stats import randint`

1. sklearn:
	1. sklearn.metrics:
		  1. mean_squared_error
		  2. mean_absolute_error
		  3. r2_score
		  4. confusion_matrix

	1. sklearn.ensemble:
	    1. RandomForestRegressor
	    2. BaggingRegressor
	    
	1. sklearn.model_selection:
	    1. train_test_split
	    2. GridSearchCV
	    3. RandomizedSearchCV
	    4. cross_validate
	    5. KFold
      
	1. sklearn.preprocessing: 
	    1. StandardScaler 
	    2. OneHotEncoder 
	    3. RobustScaler
	 
	1. sklearn.linear_model: LinearRegression
	1. sklearn.model_selection: train_test_split
	1. sklearn.pipline: Pipline 
	1. sklearn.compose: ColumnTransformer 
	1. sklearn.decimposition: PCA 
	1. sklearn.dummy: DummyRegressor
	
  
 
  
	```
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.preprocessing import RobustScaler, StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from scipy.stats import randint
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    from sklearn.dummy import DummyRegressor
    from sklearn.tree import ExtraTreeRegressor
    from sklearn.model_selection import cross_validate, KFold
	```

---
## Functions
1. **random_SCV(pipe = [], 
               grid_param = [],
               n_iter = 10, 
               cv = 5, 
               scoring = 'neg_mean_squared_error', 
               rnd_state = 42, 
               file_name = "",
              training = [])** <br>
Running RandomizedSearchCV for an estimator "pipe" according to grid_param and the other parameters including a list of x_training and y_training (training). The results are saved in param_tuning folder in the file named: file_name.

1. **grid_SCV(pipe = [], grid_param = [], cv = 5, scoring = 'neg_mean_squared_error', file_name = "", training = [])** <br>
Similar to the first function, but this time it is GridSearchCV that runs on an estimator "pipe".

1. **wr_pkl_file(file_name = "",content = "", read = False)** <br>
Dealing with either reading or writing a pkl file that contains different machine learning pipelines with their corresponding results.

1. **print_results(labels = [], est = [], plt_num = 50, log = False, testing = [])** <br>
Predicting sales prices and printing results (R-Squared, MAE, and RMSE) for different estimators (est). 

1. **validation(models = [], estimators = [], training = [], cv = 5, train_score = False):** <br>
Performs cross validation for different models using their estimators and training set. 
---

## Setup and running the code:
Clone the repo using the following command in terminal:<br>
	`git clone https://github.com/avivfaraj/DSCI631-project.git`
	
After cloning the repo, open Final_project.ipynb and run each cell one at a time in the order that they are presented. You can run the whole notebook in a single step by clicking on the menu Cell -> Run All.<br>

The first two sections are packages and functions which are required for the code to run. Make sure to run those two sections before running the program. 

---

## Acknowledgements

Dataset was found at [Kaggle](https://www.kaggle.com/new-york-city/nyc-property-sales). <br>
The origin of the data in this dataset is NYC Department of Finance [Rolling Sales](https://www1.nyc.gov/site/finance/taxes/property-rolling-sales-data.page)

