# %load q02_best_k_features/build.py
# Default imports

import pandas as pd
import numpy as np
data = pd.read_csv('data/house_prices_multivariate.csv')
#print(data.info())
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression


# Write your solution here:

def percentile_k_features(data,k=20):
    y=data['SalePrice']
    X=data.iloc[:,:-1]
    f_test, _ = f_regression(X, y)
    f_normalized=f_test/np.max(f_test)
    featureVar=X.columns.values

    topK=pd.DataFrame({'col1':f_normalized,'col2':featureVar})
    topK_sorted=topK.sort_values(by=['col1'],ascending=False)
    return topK_sorted.nlargest(7, 'col1')['col2'].tolist()
