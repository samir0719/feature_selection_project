# %load q04_select_from_model/build.py
# Default imports
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')
def select_from_model(data):
    x=data.iloc[:,:-1]
    y=data.iloc[:,-1]
    np.random.seed(9)
    mod=RandomForestClassifier()
    z=SelectFromModel(mod)
    z.fit(x,y)
    a=x.columns.values
    b=a[z.get_support()].tolist()
# Your solution code here
    return b
