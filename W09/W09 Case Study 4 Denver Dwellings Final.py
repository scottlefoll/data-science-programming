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

from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics, datasets, tree

from sklearn.datasets import make_hastie_10_2
from sklearn.datasets import make_friedman1
from sklearn.datasets import make_blobs
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.preprocessing import MinMaxScaler

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
df_ml = pd.read_csv("https://github.com/byuidatascience/data4dwellings/raw/master/data-raw/dwellings_ml/dwellings_ml.csv")
df_hot = pd.read_csv("https://github.com/byuidatascience/data4dwellings/raw/master/data-raw/dwellings_neighborhoods_ml/dwellings_neighborhoods_ml.csv")  

df = pd.read_csv("https://github.com/byuidatascience/data4dwellings/raw/master/data-raw/dwellings_ml/dwellings_ml.csv")

#%%


#%%
Markdown(df_ml.head(5).to_markdown(index=False))
#%%


#%%
Markdown(df_hot.head(5).to_markdown(index=False))
#%%


#############################################################


#%%
# limit the number of features included in the model
X = df_ml.iloc[:, 1:49] #SelectKBest
# y includes labels and x includes features
y = df_ml.iloc[:, -1] # 0 or 1 - target variable
#%%

#%%
# features = ... select the feature columns from the data frame
# targets = ... select the target column from the data frame

# features = X.filter(['netprice', 'livearea', 'basement', 'nocars', 'numbdrm',
                    #  'numbaths', 'stories', 'quality_B', 'quality_C', 'condition_AVG', 'quality'])

# feature engineering
# features = X.filter(['netprice', 'livearea', 'basement', 'stories'
#                      'nocars', 'numbdrm', 'numbaths', 'stories', 
#                      'quality_B', 'quality_C', 'condition_AVG', 'quality'])

# features = X.filter(['netprice', 'livearea', 'basement', 'stories'
#                      'nocars', 'numbdrm', 'numbaths', 'stories', 
#                      'quality_B', 'quality_C', 'condition_AVG', 'quality'])

# features = X.filter(['quality_A', 'quality_B', 'quality_C', 'quality_D'
#                      'quality_X', 'condition_AVG', 'condition_Excel', 'condition_FAIR', 
#                      'condition_Good', 'condition_VGood', 'gartype_Att', 'gartype_Att/Det', 
#                      'gartype_CP', 'gartype_Det', 'gartype_None', 'arcstyle_BI-LEVEL',
#                      'arcstyle_CONVERSIONS', 'arcstyle_ONE AND HALF-STORY', 'arcstyle_ONE_STORY',
#                      'arcstyle_SPLIT LEVEL', 'arcstyle_THREE_STORY', 'arcstyle_TWO AND HALF-STORY',
#                      'arcstyle_TWO-STORY', 'qualified_Q', 'qualified_U', 'status_I', 'status_V'])

# feature engineering set 1
# features = X.filter(['quality_A', 'quality_B', 'quality_C', 'quality_D'
#                      'quality_X', 'condition_AVG', 'condition_Excel', 'condition_FAIR', 
#                      'condition_Good', 'condition_VGood', 'gartype_Att', 'gartype_Att/Det', 
#                      'gartype_CP', 'gartype_Det', 'gartype_None', 'arcstyle_BI-LEVEL',
#                      'arcstyle_CONVERSIONS', 'arcstyle_ONE AND HALF-STORY', 'arcstyle_ONE_STORY',
#                      'arcstyle_SPLIT LEVEL', 'arcstyle_THREE_STORY', 'arcstyle_TWO AND HALF-STORY',
#                      'arcstyle_TWO-STORY', 'qualified_Q', 'qualified_U', 'status_I', 'status_V',
#                      'netprice', 'livearea', 'basement', 'stories'
#                       'nocars', 'numbdrm', 'numbaths', 'stories', 
#                       'quality_B', 'quality_C', 'condition_AVG', 'quality'])

# feature engineering set 2
features = X.filter(['quality_B', 'quality_C','condition_Good', 'gartype_Att',
                     'arcstyle_ONE AND HALF-STORY', 'arcstyle_ONE_STORY',
                     'arcstyle_TWO-STORY', 'netprice', 'livearea', 'basement', 
                     'stories', 'nocars', 'numbdrm', 'numbaths'])


target = y

# Randomize and split the samples into two groups. 
# 25% (or whatever) of the samples will be used for testing.
# The remainder will be used for training.
X_train1, X_test1, y_train1, y_test1 = train_test_split(features, target, test_size=.33) 
#%%


###################################
# Data Cleaning

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
missing_checks(df_ml, 'parcel')
missing_checks(df_ml, 'livearea')
missing_checks(df_ml, 'finbsmnt')
missing_checks(df_ml, 'basement')
missing_checks(df_ml, 'yrbuilt')
missing_checks(df_ml, 'totunits')
missing_checks(df_ml, 'stories')
missing_checks(df_ml, 'nocars')
missing_checks(df_ml, 'numbdrm')
missing_checks(df_ml, 'numbaths')
missing_checks(df_ml, 'sprice')
missing_checks(df_ml, 'deduct')
missing_checks(df_ml, 'netprice')
missing_checks(df_ml, 'tasp')
missing_checks(df_ml, 'smonth')
missing_checks(df_ml, 'syear')
missing_checks(df_ml, 'condition_AVG')
missing_checks(df_ml, 'condition_Excel')
missing_checks(df_ml, 'condition_Fair')
missing_checks(df_ml, 'condition_Good')
missing_checks(df_ml, 'condition_VGood')
missing_checks(df_ml, 'quality_A')
missing_checks(df_ml, 'quality_B')
missing_checks(df_ml, 'quality_C')
missing_checks(df_ml, 'quality_D')
missing_checks(df_ml, 'quality_X')
missing_checks(df_ml, 'gartype_Att')
missing_checks(df_ml, 'gartype_Att/Det')
missing_checks(df_ml, 'gartype_CP')
missing_checks(df_ml, 'gartype_Det')
missing_checks(df_ml, 'gartype_None')
missing_checks(df_ml, 'gartype_att/CP')
missing_checks(df_ml, 'gartype_det/CP')
missing_checks(df_ml, 'arcstyle_BI-LEVEL')
missing_checks(df_ml, 'arcstyle_CONVERSIONS')
missing_checks(df_ml, 'arcstyle_END UNIT')
missing_checks(df_ml, 'arcstyle_MIDDLE UNIT')
missing_checks(df_ml, 'arcstyle_ONE AND HALF-STORY')
missing_checks(df_ml, 'arcstyle_ONE-STORY')
missing_checks(df_ml, 'arcstyle_SPLIT LEVEL')
missing_checks(df_ml, 'arcstyle_THREE-STORY')
missing_checks(df_ml, 'arcstyle_TRI-LEVEL')
missing_checks(df_ml, 'arcstyle_TRI-LEVEL WITH BASEMENT')
missing_checks(df_ml, 'arcstyle_TWO AND HALF-STORY')
missing_checks(df_ml, 'arcstyle_TWO-STORY')
missing_checks(df_ml, 'qualified_Q')
missing_checks(df_ml, 'qualified_U')
missing_checks(df_ml, 'status_I')
missing_checks(df_ml, 'status_V')
missing_checks(df_ml, 'before1980')

#%%


#%%
# convert negative values to 0
num_cols = df_ml._get_numeric_data()
num_cols[num_cols < 0 ] = 0
num_cols = df_ml._get_numeric_data()
num_cols[num_cols < 0 ] = 0
num_cols = df_ml._get_numeric_data()
num_cols[num_cols < 0 ] = 0
#%%


#%%
df_ml.replace('', np.nan, inplace=True)
df_ml.replace(0, np.nan, inplace=True)
df_ml.replace("n/a", np.nan, inplace=True)
df_ml.replace("N/A", np.nan, inplace=True)
df_ml.replace("NA", np.nan, inplace=True)
df_ml.replace("?", np.nan, inplace=True)
df_ml.reset_index(drop=True, inplace=True)
#%%


'''

Project 4: Can you predict that?
Background
The clean air act of 1970 was the beginning of the end for the use of asbestos in 
home building. By 1976, the U.S. Environmental Protection Agency (EPA) was given 
authority to restrict the use of asbestos in paint. Homes built during and before 
this period are known to have materials with asbestos YOu can read more about this 
ban.

The state of Colorado has a large portion of their residential dwelling data that 
is missing the year built and they would like you to build a predictive model that 
can classify if a house is built pre 1980.

Colorado gave you home sales data for the city of Denver from 2013 on which to train 
your model. They said all the column names should be descriptive enough for your 
modeling and that they would like you to use the latest machine learning methods.

Deliverables

1.  A short summary that highlights key that describes the results describing insights 
from metrics of the project and the tools you used (Think “elevator pitch”).

2.  Answers to the grand questions. Each answer should include a written description of 
your results, code snippets, charts, and tables.
'''



## GRAND QUESTION 1

'''
Create 2-3 charts that evaluate potential relationships between the home variables and 
before 1980. Explain what you learn from the charts that could help a machine learning algorithm
'''


## Feature Selection

###############################################
# Univariate Selection - SelectKBest 1

# ************

#%%
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X_train1,y_train1)
dfscores = round(pd.DataFrame(fit.scores_), 2)
dfcolumns = pd.DataFrame(X_train1.columns)
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']
featureScores = featureScores.nlargest(15,'Score')
featureScores.reset_index(drop=True, inplace=True)
Markdown(featureScores.to_markdown())
#%%


#%%
plt.bar(featureScores.Specs, featureScores.Score)
plt.xlabel('Scores', fontsize=12, color='red')
plt.xticks(rotation=90)
plt.ylim(ymax = 2500, ymin = 0)
featureScores.plot(featureScores, kind='barh', color='teal')
plt.show()
#%%

#%%
###################################################
# Feature Importance - Extra Trees Classifier 

# *************

#%%
model = ExtraTreesClassifier()
model.fit(X_train1,y_train1)
#use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X_train1.columns)
feat_importances = feat_importances.nlargest(15)
Markdown(feat_importances.to_markdown())
#%%

#%%
feat_importances.nlargest(20).plot(kind='barh', color='teal')
featureScores.reset_index(drop=True, inplace=True)
plt.show()
#%%


#%%
#######################################################
# Mutual Information Method
#mutual information selecting all features

mutual = SelectKBest(score_func=mutual_info_classif, k='all')
#learn relationship from training data
mutual.fit(X_train1, y_train1)
# transform train input data
X_train_mut = mutual.transform(X_train1)
# transform test input data
X_test_mut = mutual.transform(X_test1)
#printing scores of the features
print("Mutual Information Scores: ")
for i in range(len(mutual.scores_)):
    print('Feature %d: %f' % (i, mutual.scores_[i]))
mutual_df = pd.DataFrame(mutual.scores_)
X_train1.info()
#%% 

#%%
mutual_df.plot(kind='barh', color='teal')
mutual_df.reset_index(drop=True, inplace=True)
plt.show()
#%%


## GRAND QUESTION 2

'''
Build a classification model labeling houses as being built “before 1980” or “during or 
after 1980”. Your goal is to reach or exceed 90% accuracy. Explain your final model 
choice (algorithm, tuning parameters, etc) and describe what other models you tried.
'''

"""I explored the following models and their accuracy scores for this project:
    Decision Tree Classifier                        0.8991  
    
    Gradient Boosting Classifier                    0.8693
    
    Extremely Randomized Trees Classifiers          
    
        Random Forest Classifier                    0.9963
        
        Extra Trees Classifier                      0.9525
           
    Logistic Regression with Gradient Descent       0.9925
    
    The model that I settled on was Logistic Regression with Gradient Descent. It is the one
    that I am most familiar with and it had the highest reliable accuracy score. It also 
    includes the most settable parameters to tune. The Random Forest Classifier also provided
    very accurate results, but with much less ability to tune the model.
    
    My settings for Logistic Regression with Gradient Descent were:

            iterations = 20000
            alpha = 0.9
            ep = 0.012
           
    
    
"""

###############################################
Decision Tree Classifier

#%%
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train1, y_train1)

#Predict the response for test dataset
y_pred = clf.predict(X_test1)
#%%

#%%
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test1, y_pred))
#%%


###############################################
Gradient Boosting Classifier

#%%
clf = GradientBoostingClassifier(n_estimators=1000, learning_rate=1.0, 
                                 max_depth=1, random_state=0).fit(X_train1, y_train1)
clf.score(X_test1, y_test1)
#%%

################################################
Extremely Randomized Trees Classifiers

#%%
clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
scores = cross_val_score(clf, X, y, cv=5)
scores.mean()
# 0.999...
#%%

#%%
clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
scores = cross_val_score(clf, X, y, cv=5)
scores.mean() 
# > 0.999
#%%

#############################################################
Logistic Regression with Gradient Descent

#%%
# feature engineering - limit #1
# df = df.filter(['quality_X', 'netprice', 'livearea', 'basement', 'stories'
#               'nocars', 'numbdrm', 'numbaths', 'stories', 
#               'quality_B', 'quality_C', 'condition_AVG', 'quality'])

# feature engineering - limit #2
# df = df.filter(['quality_X', 'netprice', 'livearea', 'basement', 'stories'
#               'nocars', 'numbdrm', 'numbaths', 'stories', 
#               'quality_B', 'quality_C', 'condition_AVG', 'arcstyle_ONE-STORY',
#               'gartype_ATT'])

# feature engineering - limit #3
# df = df.filter(['quality_X', 'netprice', 'livearea', 'basement', 'stories'
#               'nocars', 'numbdrm', 'numbaths', 'stories', 
#               'quality_B', 'quality_C', 'condition_AVG', 'arcstyle_ONE-STORY',
#               'gartype_ATT', 'gartype_DET'])

# feature engineering - limit #4
# df = df.filter(['quality_X', 'netprice', 'livearea', 'basement', 'stories'
#               'nocars', 'numbdrm', 'numbaths', 'stories', 'tasp',
#               'quality_B', 'quality_C', 'condition_AVG', 'arcstyle_ONE-STORY',
#               'gartype_ATT', 'gartype_DET'])

# feature engineering - limit #5
# df = df.filter(['quality_X', 'netprice', 'livearea', 'basement', 'stories'
#               'nocars', 'numbdrm', 'numbaths', 'stories', 'tasp',
#               'quality_B', 'quality_C', 'condition_AVG', 'arcstyle_ONE-STORY',
#               'gartype_ATT', 'gartype_DET', 'gartype_NONE', 'arcstyle_TWO-STORY'])

# feature engineering - limit #6
# df = df.filter(['quality_X', 'netprice', 'livearea', 'basement', 'stories', 'deduct',
#               'nocars', 'numbdrm', 'numbaths', 'stories', 'tasp', 'sprice',
#               'quality_B', 'quality_C', 'condition_AVG', 'arcstyle_ONE-STORY',
#               'gartype_ATT', 'gartype_DET', 'gartype_NONE', 'arcstyle_TWO-STORY'])

# feature engineering - limit #7
# df = df.filter(['quality_X', 'netprice', 'livearea', 'basement', 'stories', 'deduct',
#               'nocars', 'numbdrm', 'numbaths', 'stories', 'tasp', 'sprice',
#               'quality_B', 'quality_C', 'condition_AVG', 'arcstyle_ONE-STORY',
#               'condition_Fair', 'condition_Good', 
#               'gartype_ATT', 'gartype_DET', 'gartype_NONE', 'arcstyle_TWO-STORY',
#               'qualified_Q', 'qualified_U', 'status_I', 'status_V'])

# feature engineering - limit #8
# df = df.filter(['quality_X',, 'netprice', 'livearea', 'basement', 'stories', 'deduct',
#               'nocars', 'numbdrm', 'numbaths', 'stories', 'tasp', 'sprice',
#               'quality_B', 'quality_C', 'condition_AVG', 'arcstyle_ONE-STORY',
#               'gartype_ATT', 'gartype_DET', 'gartype_NONE', 'arcstyle_TWO-STORY'])

# feature engineering - limit #9
# df = df.filter(['quality_X',, 'netprice', 'livearea', 'basement', 'stories', 'deduct',
#               'nocars', 'numbdrm', 'numbaths', 'stories', 'tasp', 'sprice',
#               'quality_B', 'quality_C', 'condition_AVG', 'arcstyle_ONE-STORY',
#               'condition_Fair', 'condition_Good', 
#               'gartype_ATT', 'gartype_DET', 'gartype_NONE', 'arcstyle_TWO-STORY',
#               'qualified_Q', 'qualified_U'])

# feature engineering - limit #10
# df = df.filter(['quality_X', 'netprice', 'livearea', 'basement', 'stories', 'deduct',
#               'nocars', 'numbdrm', 'numbaths', 'stories', 'tasp', 'sprice',
#               'quality_B', 'quality_C', 'condition_AVG', 'arcstyle_ONE-STORY',
#               'condition_Fair', 'condition_Good', 
#               'gartype_ATT', 'gartype_DET', 'gartype_NONE', 'arcstyle_TWO-STORY',
#               'status_I', 'status_V'])

# feature engineering - limit #11
# df = df.filter(['quality_X', 'netprice', 'livearea', 'basement', 'stories', 'deduct',
#               'nocars', 'numbdrm', 'numbaths', 'stories', 'tasp', 'sprice',
#               'quality_B', 'quality_C', 'condition_AVG', 'arcstyle_ONE-STORY',
#               'condition_Fair', 'condition_Good', 
#               'gartype_ATT', 'gartype_DET', 'gartype_NONE', 'arcstyle_TWO-STORY',
#               'status_V'])

# feature engineering set 11A
features = df.filter(['quality_B', 'quality_C','condition_Good', 'gartype_Att',
                     'arcstyle_ONE AND HALF-STORY', 'arcstyle_ONE_STORY',
                     'arcstyle_TWO-STORY', 'netprice', 'livearea', 'basement', 
                     'stories', 'nocars', 'numbdrm', 'numbaths'])


# feature engineering - limit #12
# df = df.filter(['quality_X', 'status_I', 'status_V','qualified_Q', 'qualified_U'])

# feature engineering - limit #13
# df = df.filter(['quality_X', 'qualified_Q', 'qualified_U'])

# feature engineering - limit #14
# df = df.filter(['quality_X', 'status_I', 'status_V'])

# feature engineering - limit #15
# df = df.filter(['quality_X', 'status_I', 'status_V'])

# feature engineering - limit #16
# df = df.filter(['quality_X', 'status_I', 'status_V'])

X = df.values[:,1:-1].astype('int')
X = (X - np.mean(X, axis =0)) /  np.std(X, axis = 0) # raises accuracy by .25%

## Add a bias column to the data
X = np.hstack([np.ones((X.shape[0], 1)),X])
X = MinMaxScaler().fit_transform(X)
Y = df.iloc[:, -1]
Y = np.array(Y)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.33,random_state=42)

def Sigmoid(z):
    return 1/(1 + np.exp(-z))

def Hypothesis(theta, x):   
    return Sigmoid(x @ theta) 

def Pre_1980_Function(X,Y,theta,m):
    hi = Hypothesis(theta, X)
    _y = Y.reshape(-1, 1)
    J = 1/float(m) * np.sum(-_y * np.log(hi) - (1-_y) * np.log(1-hi))
    return J

def Cost_Function_Derivative(X,Y,theta,m,alpha):
    hi = Hypothesis(theta,X)
    _y = Y.reshape(-1, 1)
    J = alpha/float(m) * X.T @ (hi - _y)
    return J

def Gradient_Descent(X,Y,theta,m,alpha):
    new_theta = theta - Cost_Function_Derivative(X,Y,theta,m,alpha)
    return new_theta

def Accuracy(theta):
    correct = 0
    length = len(X_test)
    prediction = (Hypothesis(theta, X_test) > 0.5)
    _y = Y_test.reshape(-1, 1)
    correct = prediction == _y
    my_accuracy = (np.sum(correct) / length)*100
    print ('LR Accuracy %: ', my_accuracy)

def Logistic_Regression(X,Y,alpha,theta,num_iters):
    m = len(Y)
    for x in range(num_iters):
        new_theta = Gradient_Descent(X,Y,theta,m,alpha)
        theta = new_theta
        if x % 100 == 0:   
            print ('Built Before 1980: ', Pre_1980_Function(X,Y,theta,m))
    Accuracy(theta)
    
def Cross_Validation(X,Y,theta,alpha,num_iters):
    m = len(Y)
    for x in range(num_iters):
        new_theta = Gradient_Descent(X,Y,theta,m,alpha)
        theta = new_theta
        if x % 100 == 0:   
            print ('Built Before 1980: ', Pre_1980_Function(X,Y,theta,m))
    Accuracy(theta)

def F1_Score(y_true, y_pred):
    return 2 * (precision_score(y_true, y_pred) * recall_score(y_true, y_pred)) / (precision_score(y_true, y_pred) + recall_score(y_true, y_pred))

ep = .012

initial_theta = np.random.rand(X_train.shape[1], 1) * 2 * ep - ep
# alpha = 0.8 # 5000 iterations, all features =  98.32 % accuracy
# alpha = 0.95 # 5000 iterations, all features =  98.84 % accuracy
# alpha = 0.95 # 500 iterations, all features =  96.03 % accuracy
# alpha = 0.5 # 500 iterations, limit features =  78.02 % accuracy
# alpha = 0.3 # 500 iterations, limit features =  77.13 % accuracy
# alpha = 0.7 # 500 iterations, limit features =  77.69 % accuracy
# alpha = 0.63 # 1000 iterations, limit features 2 =  84.50 % accuracy
# alpha = 0.63 # 5000 iterations, limit features 2 =  89.73 % accuracy
# alpha = 0.63 # 10000 iterations, limit features 2 =  90.01 % accuracy
# alpha = 0.70 # 10000 iterations, limit features 2 =  90.21 % accuracy
# alpha = 0.80 # 10000 iterations, limit features 2 =  90.22 % accuracy
# alpha = 0.90 # 10000 iterations, limit features 2 =  90.28 % accuracy
# alpha = 0.90 # 20000 iterations, limit features 2 =  90.59 % accuracy
# alpha = 0.90 # 20000 iterations, limit features 3 =  90.61 % accuracy
# alpha = 0.90 # 20000 iterations, limit features 4 =  89.98 % accuracy
# alpha = 0.90 # 20000 iterations, limit features 5 =  90.62 % accuracy
# alpha = 0.90 # 20000 iterations, limit features 6 =  91.15 % accuracy
# alpha = 0.80 # 5000 iterations, limit features 6 =  91.14 % accuracy
# alpha = 0.80 # 1000 iterations, limit features 6 =  89.53 % accuracy
# alpha = 0.90 # 1000 iterations, limit features 6 =  89.76 % accuracy
# alpha = 0.90 # 5000 iterations, limit features 6 =  91.16 % accuracy
# alpha = 0.90 # 10000 iterations, limit features 6 =  91.39 % accuracy
# alpha = 0.90 # 20000 iterations, limit features 6 =  91.15 % accuracy
# alpha = 0.95 # 20000 iterations, limit features 6 =  91.40 % accuracy
# alpha = 0.95 # 10000 iterations, limit features 7 =  98.26 % accuracy
# alpha = 0.95 # 10000 iterations, limit features 8 =  86.44 % accuracy
# alpha = 0.90 # 10000 iterations, limit features 5 =  84.69 % accuracy
# alpha = 0.90 # 1000 iterations, limit features 7 =  85.44 % accuracy
# alpha = 0.90 # 1000 iterations, limit features 9 =  100 % accuracy ???
# alpha = 0.90 # 5000 iterations, all features =  98.62 % accuracy
# alpha = 0.90 # 1000 iterations, limit features 10 =  99.97 % accuracy ???
# alpha = 0.90 # 1000 iterations, limit features 11 =  94.68 % accuracy ???
# alpha = 0.90 # 1000 iterations, limit features 12 =  100 % accuracy ???
# alpha = 0.90 # 1000 iterations, limit features 13 =  67.24 % accuracy ???
# alpha = 0.90 # 1000 iterations, limit features 14 =  94.35 % accuracy ???

alpha = .90 # 500 iterations, limit features =  78.33 % accuracy
# iterations = 500
# iterations = 1000
# iterations = 5000
# iterations = 10000
iterations = 20000
Logistic_Regression(X_train, Y_train, alpha, initial_theta, iterations)

#%%
########################################



## GRAND QUESTION 3

'''
Justify your classification model by discussing the most important features selected by 
your model. This discussion should include a chart and a description of the features.
'''

''''
The following features were used as inputs to all of the models tested:

    feature                         description    

    netprice                        Net sales price
    livearea                        S.F. living area
    basement                        Basement area S.F.
    stories                         Number of stories
    nocars                          Garage size - # of cars
    numbdrm                         Number of bedrooms
    numbaths                        Number of bathrooms
    
    The following are binary one-hot encoded features:

    condition_Good                  Good Condition - 0 or 1
    gartype_Att                     Attached Garage - 0 or 1
    arcstyle_ONE AND HALF-STORY     One and Half Story - 0 or 1
    arcstyle_ONE_STORY              One Story - 0 or 1
    arcstyle_TWO-STORY              Two Story - 0 or 1
    quality_B                       Quality B - 0 or 1                       
    quality_C                       Quality C - 0 or 1
    
The feature used for the target variable is: 
    
    before1980                      Built Before 1980 - 0 or 1
                     '''
#%%
file_name = "features.txt"
feature_df = pd.read_csv(file_name)
Markdown(feature_df.to_markdown())
#%%



## GRAND QUESTION 4

'''
Describe the quality of your classification model using 2-3 different evaluation metrics. 
You also need to explain how to interpret each of the evaluation metrics you use.
'''

#%%


#%%


#%%

#%%


# Markdown(mydat.to_markdown())

# alt.Chart(mydat).mark_bar(width=20, color="purple")\
#     .encode(x = alt.X('airport_code', sort=alt.SortField('f_delay_ratio')),\
#             y = alt.Y('f_delay_ratio', scale=alt.Scale(domain=[0.13, 0.27])),)\
#     .properties(width=600, height=300)