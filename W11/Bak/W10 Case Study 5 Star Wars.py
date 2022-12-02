#%%
import pandas as pd
import altair as alt
import math
import numpy as np
import sklearn as sk
import seaborn as sns

# import scikit-plot as skp
# import urllib3
# import json
# import requests

from sklearn import metrics, datasets, tree
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from skfeature.function.similarity_based import fisher_score
from mlxtend.feature_selection import SequentialFeatureSelector as sfs, ExhaustiveFeatureSelector
from IPython.display import Markdown, display
from tabulate import tabulate
from matplotlib import pyplot as plt
%matplotlib inline
from altair import Chart, X, Y, Axis, SortField
from scipy import stats
alt.data_transformers.enable('json')
#%%

#%%
url = "https://github.com/fivethirtyeight/data/raw/master/star-wars-survey/StarWars.csv"
df = pd.read_csv(url, encoding = "ISO-8859-1")
df.head(5)
#%%

#%%
Markdown(df.head(5).to_markdown())
#%%



#%%
old_cols = list(df.columns)
print(old_cols)
#%%


#%%
df.rename(columns = {'RespondentID':'ID', 
                    'Have you seen any of the 6 films in the Star Wars franchise?':'seen_any',
                    'Do you consider yourself to be a fan of the Star Wars film franchise?':'fan',
                    'Which of the following Star Wars films have you seen? Please select all that apply.':'seen_1',
                    'Unnamed: 4':'seen_2',
                    'Unnamed: 5':'seen_3',
                    'Unnamed: 6':'seen_4',
                    'Unnamed: 7':'seen_5',
                    'Unnamed: 8':'seen_6',
                    'Please rank the Star Wars films in order of preference with 1 being your favorite film in the franchise and 6 being your least favorite film.':'rank_1',
                    'Unnamed: 10':'rank_2',
                    'Unnamed: 11':'rank_3',
                    'Unnamed: 12':'rank_4',
                    'Unnamed: 13':'rank_5',
                    'Unnamed: 14':'rank_6',
                    'Please state whether you view the following characters favorably, unfavorably, or are unfamiliar with him/her.':'favor_1',
                    'Unnamed: 16':'favor_2',
                    'Unnamed: 17':'favor_3',
                    'Unnamed: 18':'favor_4',
                    'Unnamed: 19':'favor_5',
                    'Unnamed: 20':'favor_6',
                    'Unnamed: 21':'favor_7',
                    'Unnamed: 22':'favor_8',
                    'Unnamed: 23':'favor_9',
                    'Unnamed: 24':'favor_10',
                    'Unnamed: 25':'favor_11',
                    'Unnamed: 26':'favor_12',
                    'Unnamed: 27':'favor_13',
                    'Unnamed: 28':'favor_14',
                    'Which character shot first?':'shot_first',
                    'Are you familiar with the Expanded Universe?':'familiar_expand',
                    'Do you consider yourself to be a fan of the Expanded Universe?æ':'fan_expand',
                    'Do you consider yourself to be a fan of the Star Trek franchise?':'fan_trek',
                    'Gender':'gender',
                    'Age':'age',
                    'Household Income':'income',
                    'Education':'education',
                    'Location (Census Region)':'location'
                    }, inplace = True)

Markdown(df.head(5).to_markdown())
#%%

#%%
df.describe()
#%%

#%%
df.info()
#%%


#%%
new_cols = list(df.columns)
print(new_cols)
#%%

#%%
col_changes = {'Old Column Name': 'New Column Name'}

for i in range (len(old_cols)):
    col_changes[old_cols[i]] = new_cols[i]
df_changes = pd.DataFrame.from_dict(col_changes, orient = 'index')

Markdown(df_changes.to_markdown())
    
#%%


#%%
# convert negative values to 0
num_cols = df._get_numeric_data()
num_cols[num_cols < 0 ] = 0
#%%



#%%
df.replace('', np.nan, inplace=True)
df.replace(0, np.nan, inplace=True)
df.replace("n/a", np.nan, inplace=True)
df.replace("N/A", np.nan, inplace=True)
df.replace("NA", np.nan, inplace=True)
df.replace("?", np.nan, inplace=True)
df.reset_index(drop=True, inplace=True)
#%%


#%%
#perform missing checks to clean data
def missing_checks(df, column ):
    out1 = df[column].isnull().sum()
    out1 = df[column].isnull().sum(axis = 0)
    out2 = df[column].describe()
    out3 = df[column].describe(exclude=np.number)
    print()
    print('Checking column' + column)
    print()
    print('Missing summary')
    print(out1)
    print()
    print("Numeric summaries")
    print(out2)
    print()
    print('Non Numeric summaries')
    print(out3)
#%%


#%%
missing_checks(df, 'ID')
missing_checks(df, 'seen_any')
missing_checks(df, 'fan')
missing_checks(df, 'seen_1')
missing_checks(df, 'seen_2')
missing_checks(df, 'seen_3')
missing_checks(df, 'seen_4')
missing_checks(df, 'seen_5')
missing_checks(df, 'seen_6')
missing_checks(df, 'rank_1')
missing_checks(df, 'rank_1')
missing_checks(df, 'rank_1')
missing_checks(df, 'rank_1')
missing_checks(df, 'rank_1')
missing_checks(df, 'rank_1')
missing_checks(df, 'favor_1')
missing_checks(df, 'favor_1')
missing_checks(df, 'favor_1')
missing_checks(df, 'favor_1')
missing_checks(df, 'favor_1')
missing_checks(df, 'favor_1')
missing_checks(df, 'favor_1')
missing_checks(df, 'favor_1')
missing_checks(df, 'favor_1')
missing_checks(df, 'favor_1')
missing_checks(df, 'favor_1')
missing_checks(df, 'favor_1')
missing_checks(df, 'favor_1')
missing_checks(df, 'favor_1')
missing_checks(df, 'shot_first')
missing_checks(df, 'familiar_expand')
missing_checks(df, 'fan_expand')
missing_checks(df, 'fan_trek')
missing_checks(df, 'gender')
missing_checks(df, 'age')
missing_checks(df, 'income')
missing_checks(df, 'education')
missing_checks(df, 'location')
#%%


#%%
#perform missing checks to clean data
def missing_checks(df, column ):
    out1 = df[column].isnull().sum()
    out1 = df[column].isnull().sum(axis = 0)
    out2 = df[column].describe()
    out3 = df[column].describe(exclude=np.number)
    print()
    print('Checking column' + column)
    print()
    print('Missing summary')
    print(out1)
    print()
    print("Numeric summaries")
    print(out2)
    print()
    print('Non Numeric summaries')
    print(out3)

#%%


#%%


#%%