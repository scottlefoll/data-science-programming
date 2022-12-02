# %%
# Load libraries
import pandas as pd 
import altair as alt
import seaborn as sns

# %%
# Load modules
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics

# from sklearn.naive_bayes import GaussianNB
# from sklearn.ensemble import GradientBoostingClassifier

# %%
# Load data
homes = pd.read_csv('https://github.com/byuidatascience/data4dwellings/raw/master/data-raw/dwellings_denver/dwellings_denver.csv')
dwellings_ml = pd.read_csv('https://github.com/byuidatascience/data4dwellings/raw/master/data-raw/dwellings_ml/dwellings_ml.csv')
neighborhood = pd.read_csv('https://github.com/byuidatascience/data4dwellings/raw/master/data-raw/dwellings_neighborhoods_ml/dwellings_neighborhoods_ml.csv')
alt.data_transformers.enable('json')


# %%
# Detials

# The state of Colorado has a large portion of their residential dwelling data that is 
# missing the year built and they would like you to build a predictive model that can 
# classify if a house is built pre 1980.  They would also like you to build a model that 
# predicts (regression) the actual age of each home.

# Grand Questions

# 1.Create 2-3 charts that evaluate potential relationships between the home variables and before1980.
# 2. Can you build a predictive model (before or after 1980) that has at least 90% accuracy for the state of Colorado to use?
# 3. Will you justify your classification model by detailing the most important variables in your model.
# 4. Can you describe your classification model using 2-3 model performance metrics?
# 5. Can you build a model that predicts the year the home was built?
# 6. Visualize your predicted ages compared to the homes actual age.

# %%
## Now predict before 1980 
X_pred = dwellings_ml.drop(dwellings_ml.filter(regex = 'before1980|yrbuilt').columns, axis = 1)

y_pred = dwellings_ml.filter(regex = "before1980")

X_train, X_test, y_train, y_test = train_test_split(X_pred, y_pred, test_size = .34, random_state = 76)  

y_test
# %%
X_train
# %%