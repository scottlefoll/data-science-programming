#%%
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from matplotlib import pyplot as plt
%matplotlib inline
from altair import Chart, X, Y, Axis, SortField
from scipy import stats
alt.data_transformers.enable('json')
#%%


#%%
dwellings_ml = pd.read_csv("https://github.com/byuidatascience/data4dwellings/raw/master/data-raw/dwellings_ml/dwellings_ml.csv")
dwellings_hot = pd.read_csv("https://github.com/byuidatascience/data4dwellings/raw/master/data-raw/dwellings_neighborhoods_ml/dwellings_neighborhoods_ml.csv")  
#%%


#%%
dwellings_ml.head()
#%%


#%%
dwellings_hot.head()
#%%


#%%

#%%


#%%

#%%


#%%

#%%