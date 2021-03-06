# =============================================================================
# KAGGLE House Prices: Advanced Regression Techniques
# =============================================================================


# =============================================================================
# Importation of librairies and datasets
# =============================================================================

import pandas as pd
import numpy as np
import datetime

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

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
        data[column] = data[column].fillna(now.year)
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
        data[column] = data[column].astype(str)
        data[column] = encoder.fit_transform(data[column])
        
        
    # nominals variables preprocessing
    
    for column in nominals:
        data[column] = data[column].fillna('Empty')
        data[column] = data[column].astype(str)
        dummies = data[column].str.get_dummies()
        data = pd.concat([data, dummies], axis = 1)
    
    
    return data


# This function split ours X (features) et y (label) in two different dataframe.
    
def features_labels_split(data, y_col):
    X = data.drop(y_col, axis = 1)
    y = data[y_col]
    result = [X, y]
    return result


# This function return a list of columns names. Only the columns names from X, which have a good correlation with our y column will be return.
    
def best_features(data, y_col, limit):
    
    # List of the best columns
    
    columns = np.abs(data.corr())
    columns = columns[y_col].sort_values(ascending = False)
    mask = columns > limit
    columns = columns[mask].index.tolist()
    result = columns[1:]
    
    # Matrix of correlation

    matrix = data.corr()
    matrix = matrix[columns].loc[columns]
    
    matrix = sns.clustermap(matrix, row_cluster = False, col_cluster = False )
    plt.setp(matrix.ax_heatmap.get_yticklabels(), rotation = 'horizontal')
    plt.setp(matrix.ax_heatmap.get_xticklabels(), rotation = 'vertical')
    
    return result


# This function return the accuraties scores for differents baselines models
    
def score_computation(name, model):
    complexity_score = np.mean(cross_val_score(model, X_train, y_train, cv = 5))
    generalisation_score = np.mean(cross_val_score(model, X_test, y_test, cv = 5))
    print('%s\n\nComplexity : %s.\nGeneralisation : %s.\n\n' % (name, complexity_score, generalisation_score))


# This function return the accuraties scores for differents models
    
def top_parameters(name, model, params):
    print(name, '\n')
    
    grid = GridSearchCV(model, param_grid = params, cv = 5)
    grid.fit(X, y)
    
    parameters = grid.best_params_
    score = grid.best_score_
    
    print('For parameters : %s.' % parameters)
    print('Score : %s.\n\n' % score)
    
    
# For our Data exploratory

# Repartition of na values

def nan_values(dataset):
    columns = dataset.columns.tolist()
    total = len(dataset)
    result = []
    
    for col in columns : 
        mask = pd.isnull(dataset[col])
        count = len(dataset[col].loc[mask])
        if count > 0 :
            ratio = np.round(count/total, 2)
            result.append([col, ratio])
        
    result = pd.DataFrame(result, columns = ['Name', 'NA Ratio'])
    result = result.sort_values('NA Ratio', ascending = False).reset_index()
    result = result[['Name', 'NA Ratio']]
    
    plt.figure()
    ax = result.plot('Name', 'NA Ratio', kind = 'bar', color = 'red', title = 'NA repartition', legend = False)
    ax.set_xticklabels(result['Name'], fontsize = 13, rotation = 'vertical')
    ax.set_ylabel('Ratio')
    

# Annalysis of the three best feature component ['OverallQual' 'GrLivArea' 'ExterQual']
    
def outliers_plot(dataset, title, state = None):
    plt.figure()
    plt.suptitle(title, size = 20)
    
    ax1 = plt.subplot(221, title = 'Ground living area')
    if state == 'clean':
        ax1.scatter('GrLivArea', 'SalePrice', data = dataset, alpha = 0.3)
    else:
        ax1.scatter('GrLivArea', 'SalePrice', data = dataset, c = mask_GrLivArea, alpha = 0.3)
    ax1.set_xlabel('GrLivArea')
    ax1.set_ylabel('SalePrice')
    
    feature = 'OverallQual'
    ax2 = plt.subplot(222, title = 'Quality of the house')
    ax2.boxplot(feature, 1, data = dataset)
    ax2.set_xticklabels([feature])
    ax2.set_ylabel('Rating')
    
    ax3 = plt.subplot(223, title = 'Surface of the basement')
    if state == 'clean':
        ax3.scatter('TotalBsmtSF', 'SalePrice', data = dataset, alpha = 0.3)
    else:
        ax3.scatter('TotalBsmtSF', 'SalePrice', data = dataset, c = mask_TotalBsmtSF, alpha = 0.3)
    ax3.set_xlabel('TotalBsmtSF')
    ax3.set_ylabel('SalePrice')
    
    ax4 = plt.subplot(224, title = 'Car capacity')
    ax4.boxplot('GarageCars', data = dataset)
    ax4.set_xticklabels([feature])
    ax4.set_ylabel('Car capacity')
    

# =============================================================================
# [''KitchenQual' 'TotalBsmtSF'
# 'GarageCars' '1stFlrSF' 'GarageArea' 'BsmtQual' 'FullBath' 'GarageFinish'
# 'TotRmsAbvGrd' 'YearBuilt' 'FireplaceQu' 'YearRemodAdd']
# =============================================================================
    
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

outliers = []

mask_GrLivArea = (dataset['GrLivArea'] > 4000) & (dataset['SalePrice'] < 200000)
outliers = outliers + dataset.index[mask_GrLivArea].tolist()

mask_OverallQual = (dataset['OverallQual'] == 1)
outliers = outliers + dataset.index[mask_OverallQual].tolist()

mask_TotalBsmtSF = (dataset['TotalBsmtSF'] > 5000)
outliers = outliers + dataset.index[mask_TotalBsmtSF].tolist()

mask_GarageCars = (dataset['GarageCars'] == 4)
outliers = outliers + dataset.index[mask_GarageCars].tolist()
    
outliers_plot(dataset, 'Exploratory Analysis (with outliers)')


    # NA removing

dataset = dataset.drop(outliers, axis = 0)


    # Checking
    
outliers_plot(dataset, 'Exploratory Analysis (without outliers)', state = 'clean')
        

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
scaler.fit(X)

X = scaler.transform(X)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
kaggle = scaler.transform(kaggle)


# Research of ours best parameters.

names = ['Ridge Regression', 'Lasso Regression', 'Random Forest Regressor', 'Gradient Boosted Regressor']
models = [Ridge(), Lasso(), RandomForestRegressor(), GradientBoostingRegressor()]
ridge_alpha = [value for value in range(1,100,10)]
lasso_alpha = [value for value in range(10,100,10)]

values_RF = [value for value in range(4,11)]

n_estimators_GB = [value for value in range(1,5)]
max_depth_GB = [value for value in range(40,70,10)]
learning_rate = [0.1, 0.5, 1]

params = [{'alpha' : ridge_alpha },
          {'alpha' : lasso_alpha },
          {'n_estimators' : values_RF, 'max_features' : values_RF, 'max_depth' : values_RF},
          {'n_estimators' : n_estimators_GB, 'learning_rate' : learning_rate, 'max_depth' : max_depth_GB}]


print('RECHERCHE DES MEILLEURS PARAMÈTRES\n\n')
n = len(names)
for i in range(n):
    top_parameters(names[i], models[i], params[i])
    
    
# Computation of scores.

names = ['Dummy Regression', 'Linear Regression', 'Ridge Regression',
         'Lasso Regression', 'Random Forest Regressor', 'Gradient Boosted Regressor']
models = [DummyRegressor(), LinearRegression(),
          Ridge(alpha = 1), Lasso(alpha = 90),
          RandomForestRegressor(max_depth = 9, max_features = 6, n_estimators = 9),
          GradientBoostingRegressor(learning_rate = 0.1, max_depth = 3, n_estimators = 60)]

print('CALCUL DES SCORES\n\n')
n = len(names)
for i in range(n):
    score_computation(names[i], models[i])
    
    
# Prediction of our kaggle dataset with our best model

model = GradientBoostingRegressor(learning_rate = 0.1, max_depth = 3, n_estimators = 60)
model.fit(X, y)
SalePrice = model.predict(kaggle)


# Saving of the results

prediction = pd.DataFrame({'Id' : Id_kaggle, 'SalePrice' : SalePrice})
prediction = prediction.set_index('Id')
prediction.to_csv('result.csv')