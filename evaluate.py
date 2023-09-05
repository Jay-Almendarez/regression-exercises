import acquire
from prepare import zillow_scale
from wrangle import acquire, prep, wrangle_zillow
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from scipy import stats
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import plotly.express as px
from explore import plot_variable_pairs, plot_categorical_and_continuous_vars

df = wrangle_zillow()

train_and_validate, test = train_test_split(df, random_state=117)
train, validate = train_test_split(train_and_validate)

x = train['square_feet']
y = train['value']

train['yhat_baseline'] = y.mean()
ols_model = LinearRegression().fit(train[['square_feet']], y)
train['yhat'] = ols_model.predict(train[['square_feet']])
train['residual'] = train['yhat'] - y
train['residual_baseline'] = train['yhat_baseline'] - y


train['residual_baseline_sqr'] = train.residual_baseline**2
train['residual_sqr'] = train.residual**2
SSE_baseline = train.residual_baseline_sqr.sum()
SSE = train.residual_sqr.sum()
MSE = SSE/len(train)
MSE_baseline = SSE_baseline/len(train)
RMSE = MSE**.5
RMSE_baseline = MSE_baseline**.5
ESS = sum((train['yhat'] - y.mean())**2)
TSS = ESS + SSE

def plot_residuals(y,yhat):
    '''
    plot_residuals will take the argument of y(our dependent variable 'value'), as well as our 'yhat' column created and will create a residual plot
    '''
    plt.scatter(x, y)
    plt.plot(x, train['yhat'])
    plt.xlabel('x = square footage')
    plt.ylabel('y = value of house')
    plt.title('OLS linear model')
    return plt.show()

    
def regression_errors(y, yhat):
    '''
    regression_errors will take the argument of y(our dependent variable 'value'), as well as our 'yhat' column created and will return the calculations for sum of squared errors (SSE), explained sum of squares (ESS), total sum of squares (TSS), mean squared error (MSE), and root mean squared error (RMSE).
    '''
    return print(f'-------------------------------\n\n\
    Sum of Squared Errors:\n\n\
    {SSE}\n\n\
    -------------------------------\n\n\
    Explained Sum of Squares:\n\n\
    {ESS}\n\n\
    -------------------------------\n\n\
    Total Sum of Squares:\n\n\
    {TSS}\n\n\
    -------------------------------\n\n\
    Mean Squared Error:\n\n\
    {MSE}\n\n\
    -------------------------------\n\n\
    Root Mean Squared Error:\n\n\
    {RMSE}\n\n\
    -------------------------------\n\
    ')


def baseline_mean_errors(y, yhat):
    '''
    baseline_mean_errors will take the argument of y(our dependent variable 'value'), as well as our 'yhat' column created and will return the calculations for sum of squared errors (SSE), mean squared error (MSE), and root mean squared error (RMSE).
    '''
    return print(f'-------------------------------\n\n\
    Sum of Squared Errors Baseline:\n\n\
    SSE Baseline:\n{SSE_baseline}\n\n\
    -------------------------------\n\n\
    Mean Squared Error Baseline:\n\n\
    MSE Baseline:\n{MSE_baseline}\n\n\
    -------------------------------\n\n\
    Root Mean Squared Error Baseline:\n\n\
    RMSE Baseline:\n{RMSE_baseline}\n\n\
    -------------------------------\n\
    ')


def better_than_baseline(y, yhat):
    '''
    better_than_baseline will take the calculations of our model versus our bseline and return if our model is better or not.
    '''
    return SSE < SSE_baseline


