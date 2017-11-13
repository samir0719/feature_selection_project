# %load q01_plot_corr/build.py
# Default imports
import pandas as pd
from matplotlib.pyplot import yticks, xticks, subplots, set_cmap
import matplotlib.pyplot as plt

data = pd.read_csv('data/house_prices_multivariate.csv')

def plot_corr(data,size=11):
    num_df=data._get_numeric_data().columns
    new_df=data[num_df]
    X=new_df.iloc[:,:-1]
    y=new_df.iloc[:,-1]
    a=new_df.corr()
    corr = new_df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns);
    plt.yticks(range(len(corr.columns)), corr.columns);
    #plt=subplots(figsize=size)
    plt.set_cmap('YlOrRd')
    plt.show()
