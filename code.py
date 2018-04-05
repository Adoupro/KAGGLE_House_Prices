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

# This function can preprocessing all of our dataset.

def data_cleaning(data):
    
#    After a data.info(), we can see :
#        - The numerics variable have Na values instead of zero.
#        - The categorical variables have NA values instead of "The house don't have this miscellaneous"
#        
#    Thus, we will replace all the Na with 0 for the numerics variables and 'Empty' for the categoricals variables.
   
    # Human classification of the variables

    dates = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold']
    
    binaries = ['CentralAir']
    
    ordinals = ['LotShape', 'LandContour', 'LandSlope', 'OverallQual',
               'OverallCond', 'ExterQual', 'ExterCond', 'BsmtQual',
               'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
               'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageFinish',
               'GarageQual', 'GarageCond', 'PoolQC', 'Fence']
    
    data['MoSold'] = pd.to_datetime(data['MoSold'], format="%m")
    data['MoSold'] = data['MoSold'].dt.strftime('%B')
    
    nominals = ['MSSubClass', 'MSZoning', 'Street', 'Alley',
               'Utilities', 'LotConfig', 'Neighborhood',
               'Condition1', 'Condition2', 'BldgType',
               'HouseStyle', 'RoofStyle', 'RoofMatl',
               'Exterior1st', 'Exterior2nd', 'MasVnrType',
               'Foundation', 'Heating', 'Electrical', 'Functional',
               'GarageType', 'PavedDrive', 'MiscFeature', 'MoSold',
               'SaleType', 'SaleCondition' ]
    
    numerics = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
                'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
                'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
                'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars',
                'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
                'ScreenPorch', 'PoolArea', 'MiscVal']

    # Dates variables preprocessing

    for column in dates:
        median = np.mean(data[column])
        data[column] = data[column].fillna(median)
        data[column] = data[column].astype('int')
        data[column] = pd.to_datetime(data[column], format="%Y")
        
    # Numerics variables preprocessing
    
    for column in numerics:
        data[column] = np.nan_to_num(data[column])
    
    # Ordinals variables preprocessing
    
    for column in ordinals + binaries:
        encoder = LabelEncoder()
        data[column] = data[column].fillna('Empty')
        data[column] = data[column].astype(str)
        data[column] = encoder.fit_transform(data[column])
        
    # nominals variables preprocessing
    
    for column in nominals:
        data[column] = data[column].fillna('Empty')
        data[column] = data[column].astype(str)
        dummies = data[column].str.get_dummies()
        data = pd.concat([data, dummies], axis = 1)
    
    # Reduction of dimentionality 
    
    return data


# This function split ours X (features) et y (label) in two different dataframe.
    
def features_labels_split(data, y_col):
    X = data.drop(y_col, axis = 1)
    y = data[y_col]
    result = [X, y]
    return result


# This function return a list of columns names. Only the columns names from X, which have a good correlation with our y column will be return.
    
def best_features(data, y_col, limit):
    correlation = np.abs(data.corr())
    correlation = correlation[y_col].sort_values(ascending = False)
    mask = correlation > limit
    correlation = correlation[mask]
    result = correlation.index.values[1:]
    return result


# This function return the accuraties scores for differents baselines models
    
def baseline(name, model):
    model.fit(X_train, y_train)
    complexity_score = model.score(X_train, y_train)
    generalisation_score = model.score(X_test, y_test)
    print('%s\n\nComplexity : %s.\nGeneralisation : %s.\n\n' % (name, complexity_score, generalisation_score))


# This function return the accuraties scores for differents models
    
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
kaggle = data_cleaning(kaggle)

dataset_cols = dataset.columns.tolist()
kaggle_cols = kaggle.columns.tolist()
columns = list(set(dataset_cols) & set(kaggle_cols))

kaggle = kaggle[columns]
Id_kaggle = kaggle.index.values


# Selection of the best features (Minimum 0.5 or -0.5 of correlation with the 'SalePrice' column)

best = best_features(dataset, 'SalePrice', 0.5)


# Separation of X and y

X, y = features_labels_split(dataset, 'SalePrice')

X = X[best]
kaggle = kaggle[best]

# Spliting of X_train, X_test, y_train, y_test 

X_train, X_test, y_train, y_test = train_test_split(X, y)


# Normalisation of our X dataframe

scaler = MinMaxScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
kaggle = scaler.transform(kaggle)


# Computation of our baselines models scores.

names = ['Dummy Regression', 'Linear Regression']
models = [DummyRegressor(), LinearRegression()]
n = len(names)

for i in range(n):
    baseline(names[i], models[i])


# Computation of our models scores.

names = ['Ridge Regression', 'Lasso Regression', 'Random Forest Regressor']
models = [Ridge(), Lasso(), RandomForestRegressor()]
ridge_alpha = [value for value in range(1,500,20)]
lasso_alpha = [value for value in range(10,200,10)]
values = [value for value in range(4,11)]
params = [{'alpha' : ridge_alpha },
          {'alpha' : lasso_alpha },
          {'n_estimators' : values, 'max_features' : values, 'max_depth' : values}]

n = len(names)
for i in range(n):
    top_parameters(names[i], models[i], params[i])
    
    
# Prediction of our kaggle dataset with our best model

model = RandomForestRegressor(max_depth = 9, max_features = 4, n_estimators = 9)
model.fit(X_train, y_train)

SalePrice = model.predict(kaggle)


# Saving of the results

prediction = pd.DataFrame({'Id' : Id_kaggle, 'SalePrice' : SalePrice})
prediction = prediction.set_index('Id')
prediction.to_csv('result.csv')