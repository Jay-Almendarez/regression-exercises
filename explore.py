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
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
import plotly.express as px


def plot_categorical_and_continuous_vars(cat_val, contin_val, train_and_sample_size):
    '''
    plot_categorical_and_continuous_vars will accept a categorical value, a continous value, and the dataframe we'd like to examine and will return a subplot of them using a strip plot, a box plot, and a violin plot.
    '''
    plt.figure(figsize=(20, 5))
    plt.suptitle(f'Discrete vs continuous visual relationships of {cat_val} and {contin_val}')
    plt.subplot(1, 3, 1)
    sns.stripplot(x=cat_val, y=contin_val, data=train_and_sample_size)
    plt.title(f'Strip plot of {cat_val} vs {contin_val}')
    plt.subplot(1, 3, 2)
    sns.boxplot(x=cat_val, y=contin_val, data=train_and_sample_size)
    plt.title(f'Box plot of {cat_val} vs {contin_val}')
    plt.subplot(1, 3, 3)
    sns.violinplot(x=cat_val, y=contin_val, data=train_and_sample_size)
    plt.title(f'Violin plot of {cat_val} vs {contin_val}')
    return plt.show()
    
    
def plot_variable_pairs(train):
    '''
    plot_variable_pairs will accept the dataframe we are working with and will plot a lm plot by the categorical values selected and map them against out continuous value 'value'
    '''
    columns_to_scale=['beds', 'baths', 'year_built', 'square_feet']
    for col in columns_to_scale:
        sns.lmplot(x=col, y='value', data=train.sample(1000), line_kws={'color': 'crimson'})
    return plt.show()