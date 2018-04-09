# =============================================================================
# KAGGLE House Prices: Advanced Regression Techniques
# =============================================================================


# =============================================================================
# Importation of librairies and datasets
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

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
    
    ordinals = ['LotShape', 'LandContour', 'LandSlope',
                'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 
                'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
               'HeatingQC', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageFinish',
               'GarageQual', 'GarageCond', 'PoolQC', 'Fence']
    
    # Columns 'OverallQual' and 'OverallCond' don't have to be processed
    
    data['MoSold'] = pd.to_datetime(data['MoSold'], format="%m")
    data['MoSold'] = data['MoSold'].dt.strftime('%B')
    
    nominals = ['MSSubClass', 'MSZoning', 'Street', 'Alley',
               'Utilities', 'LotConfig', 'Neighborhood',
               'Condition1', 'Condition2', 'BldgType',
               'HouseStyle', 'RoofStyle', 'RoofMatl',
               'Exterior1st', 'Exterior2nd', 'MasVnrType',
               'Foundation', 'Heating', 'Electrical',
               'GarageType', 'PavedDrive', 'MiscFeature', 'MoSold',
               'SaleType', 'SaleCondition' ]
    
    numerics = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
                'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
                'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
                'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars',
                'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
                'ScreenPorch', 'PoolArea', 'MiscVal']

    
    # Ordinales variables preprocessing
    
    encoding = {
    'LotShape' : {'Reg' : 0 , 'IR1' : 1 , 'IR2' : 2 , 'IR3' : 3},
    'LandContour' : {'Lvl' : 3 , 'Bnk' : 2 , 'Low' : 0 , 'HLS' : 1},
    'LandSlope' : {'Gtl' : 2 , 'Mod' : 1 , 'Sev' : 0},
    'ExterQual' : {'TA' : 3 , 'Gd' : 4 , 'Empty' : 0 , 'Fa' : 2 , 'Po' : 1, 'Ex' : 5},
    'ExterCond' : {'TA' : 3 , 'Gd' : 4 , 'Empty' : 0 , 'Fa' : 2 , 'Po' : 1, 'Ex' : 5},
    'BsmtQual' : {'TA' : 3 , 'Gd' : 4 , 'Empty' : 0 , 'Fa' : 2 , 'Po' : 1, 'Ex' : 5},
    'BsmtCond' : {'TA' : 3 , 'Gd' : 4 , 'Empty' : 0 , 'Fa' : 2 , 'Po' : 1, 'Ex' : 5},
    'BsmtExposure' : {'No' : 1 , 'Gd' : 4 , 'Mn' : 2 , 'Av' : 3 , 'Empty' : 0},
    'BsmtFinType1' : {'GLQ' : 6 , 'ALQ' : 5 , 'Unf' : 1 , 'Rec' : 3 , 'BLQ' : 4 , 'Empty' : 0 , 'LwQ' : 2},
    'BsmtFinType2' : {'GLQ' : 6 , 'ALQ' : 5 , 'Unf' : 1 , 'Rec' : 3 , 'BLQ' : 4 , 'Empty' : 0 , 'LwQ' : 2},
    'HeatingQC' : {'TA' : 3 , 'Gd' : 4 , 'Empty' : 0 , 'Fa' : 2 , 'Po' : 1, 'Ex' : 5},
    'KitchenQual' : {'TA' : 3 , 'Gd' : 4 , 'Empty' : 0 , 'Fa' : 2 , 'Po' : 1, 'Ex' : 5},
    'Functional' : {'Typ' : 7 , 'Min1' : 6 , 'Maj1' : 3 , 'Min2' : 5 , 'Mod' : 4 , 'Maj2' : 2 , 'Sev' : 1, 'Sal' : 0},
    'FireplaceQu' : {'TA' : 3 , 'Gd' : 4 , 'Empty' : 0 , 'Fa' : 2 , 'Po' : 1, 'Ex' : 5},
    'GarageFinish' : {'RFn' : 2 , 'Unf' : 1 , 'Fin' : 3 , 'Empty' : 0},
    'GarageQual' : {'TA' : 3 , 'Gd' : 4 , 'Empty' : 0 , 'Fa' : 2 , 'Po' : 1, 'Ex' : 5},
    'GarageCond' : {'TA' : 3 , 'Gd' : 4 , 'Empty' : 0 , 'Fa' : 2 , 'Po' : 1, 'Ex' : 5},
    'PoolQC' : {'TA' : 3 , 'Gd' : 4 , 'Empty' : 0 , 'Fa' : 2 , 'Po' : 1, 'Ex' : 5},
    'Fence' : {'Empty' : 0, 'MnPrv' : 3 , 'GdWo' : 2 , 'GdPrv' : 4 , 'MnWw' : 1}
    }
    
    for column in ordinals :
        data[column] = data[column].fillna('Empty')
        data[column] = data[column].replace(encoding[column])


    # Dates variables preprocessing
    
    now = datetime.datetime.now()
    
    for column in dates:
        median = np.mean(data[column])
        data[column] = data[column].fillna(median)
        data[column] = data[column].astype('int')
        data[column] = pd.to_datetime(data[column], format="%Y")
        data[column] = data[column] - now
        data[column] = data[column].dt.days
        data[column] = np.abs(data[column] / 365)
        data[column] = np.round(data[column])
        
        
    # Numerics variables preprocessing
    
    for column in numerics:
        data[column] = np.nan_to_num(data[column])
    
    # Binaries variables preprocessing
    
    for column in binaries:
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
    
    
# For our Data exploratory

# Repartition of na values

def nan_values(dataset):
    columns = dataset.columns.tolist()
    total = len(dataset)
    result = []
    
    for col in columns : 
        mask = pd.isnull(dataset[col])
        count = len(dataset[col].loc[mask])
        result.append([col, count/total])
        
    result = pd.DataFrame(result, columns = ['Column', 'NA Ratio'])
    
    plt.figure()
    ax = plt.subplot(111, title = 'NA Values repartition')
    ax.bar('Column', 'NA Ratio', data = result )
    ax.set_xticklabels(result['Column'], rotation='vertical')
    ax.set_ylabel('Ratio')
    plt.show()
    

#The three best feature component ['OverallQual' 'GrLivArea' 'ExterQual']
    
def outliers_plot(dataset, title):
    plt.figure()
    plt.suptitle(title, size = 20)
    
    feature = 'OverallQual'
    ax1 = plt.subplot(131, title = 'Quality of the house')
    ax1.boxplot(feature, 1, data = dataset)
    ax1.set_xticklabels([feature])
    ax1.set_ylabel('Rating')
    
    feature = 'ExterQual'
    ax2 = plt.subplot(132, title = 'Quality of the material on the exterior')
    ax2.boxplot(feature, 1, data = dataset)
    ax2.set_xticklabels([feature])
    ax2.set_ylabel('Rating')
    
    ax3 = plt.subplot(133, title = 'Ground living area')
    ax3.scatter('GrLivArea', 'SalePrice', data = dataset, alpha = 0.3)
    ax3.set_xlabel('GrLivArea')
    ax3.set_ylabel('SalePrice')
    
    plt.show()

# =============================================================================
# Project
# =============================================================================


# Exploratory analysis
    
    # Nan values
    
nan_values(dataset)

    # Data cleaning
    
dataset = data_cleaning(dataset)
kaggle = data_cleaning(kaggle)

dataset_cols = dataset.columns.tolist()
kaggle_cols = kaggle.columns.tolist()
columns = list(set(dataset_cols) & set(kaggle_cols))

kaggle = kaggle[columns]
Id_kaggle = kaggle.index.values



    # Outliers
outliers_plot(dataset, 'Exploratory Analysis (with outliers)')

outliers = []

mask = (dataset['GrLivArea'] > 4000) & (dataset['SalePrice'] < 200000)
outliers = outliers + dataset.index[mask].tolist()

mask = (dataset['OverallQual'] == 1)
outliers = outliers + dataset.index[mask].tolist()

    # Data cleaning

dataset = dataset.drop(outliers, axis = 0)

    # Checking
    
outliers_plot(dataset, 'Exploratory Analysis (without outliers)')
        

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

model = RandomForestRegressor(max_depth = 8, max_features = 6, n_estimators = 8)
model.fit(X_train, y_train)
SalePrice = model.predict(kaggle)


# Saving of the results

prediction = pd.DataFrame({'Id' : Id_kaggle, 'SalePrice' : SalePrice})
prediction = prediction.set_index('Id')
prediction.to_csv('result.csv')