#%%
import pandas as pd
import altair as alt
# import math
import numpy as np
# import sklearn as sk
# import seaborn as sns

# import scikit-plot as skp
# import urllib3
import json
# import requests

# from sklearn import metrics, datasets, tree
# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.naive_bayes import GaussianNB
# from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
# from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import f1_score
# from skfeature.function.similarity_based import fisher_score
# from mlxtend.feature_selection import SequentialFeatureSelector as sfs, ExhaustiveFeatureSelector
from IPython.display import Markdown, display
from tabulate import tabulate
# from matplotlib import pyplot as plt
%matplotlib inline
# from altair import Chart, X, Y, Axis, SortField
# from scipy import stats
alt.data_transformers.enable('json')
#%%

#%%
url = "https://github.com/fivethirtyeight/data/raw/master/star-wars-survey/StarWars.csv"
df = pd.read_csv(url, encoding = "ISO-8859-1")

dat_names = pd.read_csv(url, encoding = "ISO-8859-1", nrows = 1).melt()
# dat = pd.read_csv(url, encoding = "ISO-8859-1",skiprows =2, header = None )
#%%

# %%

# this is not complete.
(dat_names
   .replace('Unnamed: \d{1,2}', np.nan, regex=True)
   .replace('Response', "")
)

(dat_names
   .replace('Unnamed: \d{1,2}', np.nan, regex=True)
   .replace('Response', "")
   .assign(
      clean_variable = lambda x: x.variable.str.strip()
         .replace(
            'Which of the following Star Wars films have you seen? Please select all that apply.','seen'),
      clean_value = lambda x: x.value.str.strip()
   )
)

(dat_names
   .replace('Unnamed: \d{1,2}', np.nan, regex=True)
   .replace('Response', "")
   .assign(
      clean_variable = lambda x: x.variable.str.strip()
         .replace(
            'Which of the following Star Wars films have you seen? Please select all that apply.','seen'),
      clean_value = lambda x: x.value.str.strip())
   .fillna(method = 'ffill')
   .assign(
      column_name = lambda x: x.clean_variable.str.cat(x.clean_value, sep = "__"),
   )
)
#%%

#%%
dat_names.columns
dat.columns 
# %%

#%%
# Shorten the column names and clean them 
# up for easier use with pandas.
# we want to use this with the .replace() command that accepts a dictionary.
variables_replace = {
    'Which of the following Star Wars films have you seen\\? Please select all that apply\\.':'seen',
    'Please rank the Star Wars films in order of preference with 1 being your favorite film in the franchise and 6 being your least favorite film.':'rank',
    'Please state whether you view the following characters favorably, unfavorably, or are unfamiliar with him/her.':'view',
    'Do you consider yourself to be a fan of the Star Trek franchise\\?':'star_trek_fan',
    'Do you consider yourself to be a fan of the Expanded Universe\\?\x8cæ':'expanded_fan',
    'Are you familiar with the Expanded Universe\\?':'know_expanded',
    'Have you seen any of the 6 films in the Star Wars franchise\\?':'seen_any',
    'Do you consider yourself to be a fan of the Star Wars film franchise\\?':'star_wars_fans',
    'Which character shot first\\?':'shot_first',
    'Unnamed: \d{1,2}':np.nan,
    ' ':'_',
}

values_replace = {
    'Response':'',
    'Star Wars: Episode ':'',
    ' ':'_'
}
#%%

# %%
print(dat_names.value)
dat_names.value.str.strip().replace(values_replace, regex=True)
#%%

# %%
print(dat_names.variable)
dat_names.variable.str.strip().replace(variables_replace, regex=True)
#%%

# %%
dat_cols_use = (dat_names
    .assign(
        value_replace = lambda x:  x.value.str.strip().replace(values_replace, regex=True),
        variable_replace = lambda x: x.variable.str.strip().replace(variables_replace, regex=True)
    )
    .fillna(method = 'ffill')
    .fillna(value = "")
    .assign(column_names = lambda x: x.variable_replace.str.cat(x.value_replace, sep = "__").str.strip('__').str.lower())
    )
dat_cols_use
# %%

#%%
Markdown(df.head(20).to_markdown())
#%%


#%%
df.describe()
#%%

#%%
df.info()
#%%

#%%
old_cols = list(df.columns)
print(old_cols)
#%%


#%%
df.rename(columns = {'RespondentID':'ID', 
                    'Have you seen any of the 6 films in the Star Wars franchise?':'seen_any',
                    'Do you consider yourself to be a fan of the Star Wars film franchise?':'fan',
                    'Which of the following Star Wars films have you seen? Please select all that apply.':'seen_I',
                    'Unnamed: 4':'seen_II',
                    'Unnamed: 5':'seen_III',
                    'Unnamed: 6':'seen_IV',
                    'Unnamed: 7':'seen_V',
                    'Unnamed: 8':'seen_VI',
                    'Please rank the Star Wars films in order of preference with 1 being your favorite film in the franchise and 6 being your least favorite film.':'rank_I',
                    'Unnamed: 10':'rank_II',
                    'Unnamed: 11':'rank_III',
                    'Unnamed: 12':'rank_IV',
                    'Unnamed: 13':'rank_V',
                    'Unnamed: 14':'rank_VI',
                    'Please state whether you view the following characters favorably, unfavorably, or are unfamiliar with him/her.':'han',
                    'Unnamed: 16':'luke',
                    'Unnamed: 17':'leia',
                    'Unnamed: 18':'anakin',
                    'Unnamed: 19':'obiwan',
                    'Unnamed: 20':'emperor',
                    'Unnamed: 21':'darth',
                    'Unnamed: 22':'lando',
                    'Unnamed: 23':'boba',
                    'Unnamed: 24':'c3po',
                    'Unnamed: 25':'r2d2',
                    'Unnamed: 26':'jarjar',
                    'Unnamed: 27':'padme',
                    'Unnamed: 28':'yoda',
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

cols = ['seen_any', 'fan', 'seen_I', 'seen_II', 'seen_III', 'seen_IV', 'seen_V', 
        'seen_VI', 'rank_I', 'rank_II', 'rank_III', 'rank_IV', 'rank_V', 'rank_VI', 
        'han', 'luke', 'leia', 'anakin', 'obiwan', 'emperor', 'darth', 
        'lando', 'boba', 'c3po', 'r2d2', 'jarjar', 'padme', 
        'yoda', 'shot_first', 'familiar_expand', 'fan_expand', 'fan_trek', 
        'gender', 'age', 'income', 'education', 'location']

df.seen_I.value_counts()
#%%


#%%
Markdown(df.head(3).to_markdown())
#%%

#%%
new_cols = list(df.columns)
print(new_cols)
#%%


#%%
# convert negative values to 0
num_cols = df._get_numeric_data()
num_cols[num_cols < 0 ] = 0
#%%


#%%
#perform missing checks to clean data
def missing_checks(df, column ):
    out1 = df[column].isnull().sum()
    out1 = df[column].isnull().sum(axis = 0)
    out2 = df[column].describe()
    out3 = df[column].describe(exclude=np.number)
    print()
    print('*********Checking column ' + column+ ' **********')
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
missing_checks(df, 'seen_any')
missing_checks(df, 'fan')
missing_checks(df, 'seen_I')
missing_checks(df, 'seen_II')
missing_checks(df, 'seen_III')
missing_checks(df, 'seen_IV')
missing_checks(df, 'seen_V')
missing_checks(df, 'seen_VI')
missing_checks(df, 'rank_I')
missing_checks(df, 'rank_II')
missing_checks(df, 'rank_III')
missing_checks(df, 'rank_IV')
missing_checks(df, 'rank_V')
missing_checks(df, 'rank_VI')
missing_checks(df, 'han')
missing_checks(df, 'luke')
missing_checks(df, 'leia')
missing_checks(df, 'anakin')
missing_checks(df, 'obiwan')
missing_checks(df, 'emperor')
missing_checks(df, 'darth')
missing_checks(df, 'lando')
missing_checks(df, 'boba')
missing_checks(df, 'c3po')
missing_checks(df, 'r2d2')
missing_checks(df, 'jarjar')
missing_checks(df, 'padme')
missing_checks(df, 'yoda')
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
df = df.drop(index=0)
df.fillna(0, inplace = True)
df.replace({'Yes':1, 'No':0}, inplace = True)
df.replace({'Male':1, 'Female':2}, inplace = True)
df.replace({"I don't understand this question":0}, inplace = True)
df.replace({'Response':'0'}, inplace = True)
df.replace('', 0, inplace=True)
df.replace('nan', 0, inplace=True)
df.replace('NaN', 0, inplace=True)
df.replace("n/a", 0, inplace=True)
df.replace("N/A", 0, inplace=True)
df.replace("NA", 0, inplace=True)
df.replace("?", 0, inplace=True)

df.seen_I.loc[df['seen_I'] != 0] = 1
df.seen_II.loc[df['seen_II'] != 0] = 1
df.seen_III.loc[df['seen_III'] != 0] = 1
df.seen_IV.loc[df['seen_IV'] != 0] = 1
df.seen_V.loc[df['seen_V'] != 0] = 1
df.seen_VI.loc[df['seen_VI'] != 0] = 1

df.shot_first.loc[df['shot_first'] == 'Han'] = 1
df.shot_first.loc[df['shot_first'] == 'Greedo'] = 2

df.education.loc[df['education'] == 'Less than high school degree'] = 1
df.education.loc[df['education'] == 'High school degree'] = 2
df.education.loc[df['education'] == 'Some college or Associate degree'] = 3
df.education.loc[df['education'] == 'Bachelor degree'] = 4
df.education.loc[df['education'] == 'Graduate degree'] = 5

df.income.loc[df['income'] == '$0 - $24,999'] = 1
df.income.loc[df['income'] == '$25,000 - $49,999'] = 2
df.income.loc[df['income'] == '$50,000 - $99,999'] = 3
df.income.loc[df['income'] == '$100,000 - $149,999'] = 4
df.income.loc[df['income'] == '	$150,000+'] = 4
df.income.loc[df['income'] == '$150,000+'] = 4

df.age.loc[df['age'] == '18-29'] = 1
df.age.loc[df['age'] == '30-44'] = 2

df.location.loc[df['location'] == 'New England'] = 1
df.location.loc[df['location'] == 'Middle Atlantic'] = 2
df.location.loc[df['location'] == 'South Atlantic'] = 3
df.location.loc[df['location'] == 'East North Central'] = 4
df.location.loc[df['location'] == '	Mountain'] = 5
df.location.loc[df['location'] == 'Mountain'] = 5
df.location.loc[df['location'] == 'Pacific'] = 6
df.location.loc[df['location'] == 'West North Central'] = 7
df.location.loc[df['location'] == 'East South Central'] = 8
df.location.loc[df['location'] == 'West South Central'] = 9

df.replace("Very favorably", 4, inplace=True)
df.replace("Somewhat favorably", 3, inplace=True)
df.replace("Neither favorably nor unfavorably (neutral)", 2, inplace=True)
df.replace("	Unfamiliar (N/A)", 2, inplace=True)
df.replace("Unfamiliar (N/A)", 2, inplace=True)
df.replace("Somewhat unfavorably", 1, inplace=True)
df.replace("Very unfavorably", 0, inplace=True)

df['seen_total'] = df.seen_I + df.seen_II + df.seen_III + df.seen_IV + df.seen_V + df.seen_VI
df['seen_1'] = np.where(df['seen_total'] == 1, 1, 0)
df['seen_2'] = np.where(df['seen_total'] == 2, 1, 0)
df['seen_3'] = np.where(df['seen_total'] == 3, 1, 0)
df['seen_4'] = np.where(df['seen_total'] == 4, 1, 0)
df['seen_5'] = np.where(df['seen_total'] == 5, 1, 0)
df['seen_6'] = np.where(df['seen_total'] == 6, 1, 0)
df.drop(columns=['seen_total'], inplace=True)

df.reset_index(drop=True, inplace=True)
Markdown(df.head(5).to_markdown())
#%%


#%%
# one hot encoding
df = pd.get_dummies(data=df, columns=['rank_I'])
df = pd.get_dummies(data=df, columns=['rank_II'])
df = pd.get_dummies(data=df, columns=['rank_III'])
df = pd.get_dummies(data=df, columns=['rank_IV'])
df = pd.get_dummies(data=df, columns=['rank_V'])
df = pd.get_dummies(data=df, columns=['rank_VI'])

df = pd.get_dummies(data=df, columns=['han'])
df = pd.get_dummies(data=df, columns=['luke'])
df = pd.get_dummies(data=df, columns=['leia'])
df = pd.get_dummies(data=df, columns=['anakin'])
df = pd.get_dummies(data=df, columns=['obiwan'])
df = pd.get_dummies(data=df, columns=['emperor'])
df = pd.get_dummies(data=df, columns=['darth'])
df = pd.get_dummies(data=df, columns=['lando'])
df = pd.get_dummies(data=df, columns=['boba'])
df = pd.get_dummies(data=df, columns=['c3po'])
df = pd.get_dummies(data=df, columns=['r2d2'])
df = pd.get_dummies(data=df, columns=['jarjar'])
df = pd.get_dummies(data=df, columns=['padme'])
df = pd.get_dummies(data=df, columns=['yoda'])

df = pd.get_dummies(data=df, columns=['shot_first'])
df = pd.get_dummies(data=df, columns=['age'])
df = pd.get_dummies(data=df, columns=['education'])
df = pd.get_dummies(data=df, columns=['location'])

df.reset_index(drop=True, inplace=True)
Markdown(df.head(5).to_markdown())
#%%




'''
Project 5: The War with Star Wars

Survey data is notoriously difficult to munge. Even when the data is recorded cleanly the 
options for ‘write in questions’, ‘choose from multiple answers’, ‘pick all that are right’, 
and ‘multiple choice questions’ makes storing the data in a tidy format difficult.

In 2014, FiveThirtyEight surveyed over 1000 people to write the article titled, America’s 
Favorite ‘Star Wars’ Movies (And Least Favorite Characters). They have provided the data 
on GitHub.

For this project, your client would like to use the Star Wars survey data to figure out if 
they can predict an interviewing job candidate’s current income based on a few responses 
about Star Wars movies.


Deliverables

1.  A short summary that highlights key that describes the results describing insights 
from metrics of the project and the tools you used (Think “elevator pitch”).

2.  Answers to the grand questions. Each answer should include a written description of 
your results, code snippets, charts, and tables.
'''

#%%


#%%

## GRAND QUESTION 1

# Shorten the column names and clean them up for easier use with pandas. Provide a 
# table or list that exemplifies how you fixed the names.

#%%{python}
#| label: GQ1 Table
#| code-summary: table example
#| tbl-cap: "Not much of a table"
#| tbl-cap-location: top
# Include and execute your code here
col_changes = {'Old Column Name': 'New Column Name'}

for i in range (len(old_cols)):
    col_changes[old_cols[i]] = new_cols[i]
df_changes = pd.DataFrame.from_dict(col_changes, orient = 'index')
Markdown(df_changes.to_markdown()) 
#%%



## GRAND QUESTION 2

'''
Clean and format the data so that it can be used in a machine learning model. As 
you format the data, you should complete each item listed below. In your final 
report provide example(s) of the reformatted data with a short description of 
the changes made.

    1.  Filter the dataset to respondents that have seen at least one film.
    2.  Create a new column that converts the age ranges to a single number. Drop 
    the age range categorical column.
    3.  Create a new column that converts the education groupings to a single number. 
    Drop the school categorical column
    4.  Create a new column that converts the income ranges to a single number. Drop 
    the income range categorical column.
    5.  Create your target (also known as “y” or “label”) column based on the new 
    income range column.
    6.  One-hot encode all remaining categorical columns.'''


#%%
df.describe()
#%%

#%%
df.info()
#%%

#%%
df_filter = df[df['seen_any'] == 'Yes']
df['first_set'] = df['first_set'].replace(['18-29', 'Red'], ['Green', 'White'])
#%%

#%%
Markdown(df_filter.head(5).to_markdown())
#%%

#%%
df_filter.describe()
#%%

#%%
df_filter.info()
#%%


#%%{python}
#| label: GQ2 Table 1
#| code-summary: table example
#| tbl-cap: "Not much of a table"
#| tbl-cap-location: top
# Include and execute your code here
mydat = dat.head(1000)\
    .groupby('year')\
    .sum()\
    .reset_index()\
    .tail(10)\
    .filter(["year", "AK","AR"])

display(mydat)
#%%

#%%
{python}
#| label: GQ2 Chart 1
#| code-summary: plot example
#| fig-cap: "My useless chart"
#| fig-align: center
# Include and execute your code here
alt.Chart(dat.head(200))\
    .encode(x = "name", y = "AK")\
    .mark_bar()
#%%


## GRAND QUESTION 3

# Validate that the data provided on GitHub lines up with the article by recreating 2 
# of the visuals from the article.




#%%{python}
#| label: GQ3 Table 1
#| code-summary: table example
#| tbl-cap: "Not much of a table"
#| tbl-cap-location: top
# Include and execute your code here
mydat = dat.head(1000)\
    .groupby('year')\
    .sum()\
    .reset_index()\
    .tail(10)\
    .filter(["year", "AK","AR"])

display(mydat)
#%%

#%%
{python}
#| label: GQ3 Chart 1
#| code-summary: plot example
#| fig-cap: "My useless chart"
#| fig-align: center
# Include and execute your code here
alt.Chart(dat.head(200))\
    .encode(x = "name", y = "AK")\
    .mark_bar()
#%%

## GRAND QUESTION 4

# Build a machine learning model that predicts whether a person makes more than $50k. Describe 
# your model and report the accuracy.


#%%{python}
#| label: GQ4 Table 1
#| code-summary: table example
#| tbl-cap: "Not much of a table"
#| tbl-cap-location: top
# Include and execute your code here
mydat = dat.head(1000)\
    .groupby('year')\
    .sum()\
    .reset_index()\
    .tail(10)\
    .filter(["year", "AK","AR"])

display(mydat)
#%%

#%%
{python}
#| label: GQ4 Chart 1
#| code-summary: plot example
#| fig-cap: "My useless chart"
#| fig-align: center
# Include and execute your code here
alt.Chart(dat.head(200))\
    .encode(x = "name", y = "AK")\
    .mark_bar()
#%%