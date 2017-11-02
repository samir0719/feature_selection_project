# %load q03_rf_rfe/build.py
# Default imports
import pandas as pd
import numpy as np
data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
def rf_rfe(data):
    model = RandomForestClassifier(n_jobs=-1)

    X, y = data.iloc[:,:-1], data.iloc[:,-1]
# create the RFE model and select 3 attributes
    rfe = RFE(model, n_features_to_select=17)
    rfe = rfe.fit(X, y)
    fl=X.columns.values
   # summarize the selection of the attributes
#print(fl[rfe.support_])
    ls=fl[rfe.ranking_==1].tolist()
    return ls
