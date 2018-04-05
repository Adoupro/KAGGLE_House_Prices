# =============================================================================
# KAGGLE House Prices: Advanced Regression Techniques
# =============================================================================

# =============================================================================
# Importation of librairies and datasets
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor

dataset = pd.read_csv('train.csv', index_col = 'Id')
kaggle = pd.read_csv('test.csv', index_col = 'Id')

# =============================================================================
# Functions
# =============================================================================

def data_cleaning(data):
    
#    After a data.info(), we can see :
#        - The numerics variable have Na values instead of zero.
#        - The categorical variables have NA values instead of "The house don't have this miscellaneous"
#        
#    Thus, we will replace all the Na with 0 for the numerics variables and 'Empty' for the categoricals variables.
   
    # Numerics variables preprocessing
    
    cols = data.dtypes
    mask = (cols == 'int64') | (cols == 'float64')
    cols = cols[mask].index.values
    
    for column in cols:
        data[column] = np.nan_to_num(data[column])
    
    # Categoricals variables preprocessing
    
    cols = data.dtypes
    mask = cols == 'object'
    cols = cols[mask].index.values
    
    for column in cols:
        encoder = LabelEncoder()
        data[column] = data[column].fillna('Empty')
        data[column] = encoder.fit_transform(data[column].astype(str))
    
    # Optimisation
    
    cols = []
    data = data.drop(cols, axis = 1)
    
    return data


def features_labels_split(data, y_col):
    X = data.drop(y_col, axis = 1)
    y = data[y_col]
    result = [X, y]
    return result


def best_features(data, y_col, limit):
    correlation = np.abs(data.corr())
    
    # In the purpose to localise not usefull features
    test = correlation
    test = test.sort_values(y_col, ascending = False)
    mask = correlation[y_col] >= limit
    test = test[mask]
    cols = test.index.values
    test = test[cols] >= limit
    
    correlation = correlation[y_col].sort_values(ascending = False)
    mask = correlation > limit
    correlation = correlation[mask]
    result = correlation.index.values[1:]
    return result


def baseline(name, model):
    model.fit(X_train, y_train)
    complexity_score = model.score(X_train, y_train)
    generalisation_score = model.score(X_test, y_test)
    print('%s\n\nComplexity : %s.\nGeneralisation : %s.\n\n' % (name, complexity_score, generalisation_score))


def top_parameters(name, model, params):
    print(name, '\n')
    
    grid = GridSearchCV(model, param_grid = params, cv = 3)
    grid.fit(X_train, y_train)
    
    parameters = grid.best_params_
    complexity_score = grid.best_score_
    
    grid = GridSearchCV(model, param_grid = params, cv = 3)
    grid.fit(X_train, y_train)
    
    generalisation_score = grid.best_score_
    print('For parameters : %s.' % parameters)
    print('Complexity : %s. \nGeneralisation : %s.\n\n' % (complexity_score, generalisation_score))
    

# =============================================================================
# Project
# =============================================================================

# Cleaning of the data
    
dataset = data_cleaning(dataset)


# Selection of the best features (Minimum 0.5 or -0.5 of correlation with the 'SalePrice' column)

X_cols = best_features(dataset, 'SalePrice', 0.5)


# Separation of X and y

X, y = features_labels_split(dataset, 'SalePrice')

X = X[X_cols]


# Split X_train, X_test, y_train, y_test 

X_train, X_test, y_train, y_test = train_test_split(X, y)

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# Optimisation 

names = ['Dummy Regression', 'Linear Regression']
models = [DummyRegressor(), LinearRegression()]

n_model = len(names)
for i in range(n_model):
    baseline(names[i], models[i])
    

names = ['Ridge Regression', 'Lasso Regression', 'Random Forest Regressor']
models = [Ridge(), Lasso(), RandomForestRegressor()]
ridge_alpha = [value for value in range(10,100,10)]
lasso_alpha = [value for value in range(500,1500,100)]
values = [value for value in range(4,12)]

params = [{'alpha' : ridge_alpha },
          {'alpha' : lasso_alpha },
          {'n_estimators' : values, 'max_features' : values, 'max_depth' : values}]

n_model = len(names)
for i in range(n_model):
    top_parameters(names[i], models[i], params[i])
    
    
# Prediction

kaggle = data_cleaning(kaggle)
kaggle = kaggle[X_cols]

model = RandomForestRegressor(max_depth = 10, max_features = 6, n_estimators = 9)
model.fit(X_train, y_train)

Id = kaggle.index.values
SalePrice = model.predict(kaggle)

prediction = pd.DataFrame({'Id' : Id, 'SalePrice' : SalePrice})
prediction = prediction.set_index('Id')
prediction.to_csv('result.csv')

print(prediction)