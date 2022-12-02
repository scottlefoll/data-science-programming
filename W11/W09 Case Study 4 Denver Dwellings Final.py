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
# dwellings_ml = pd.read_csv("https://github.com/byuidatascience/data4dwellings/raw/master/data-raw/dwellings_ml/dwellings_ml.csv")
# dwellings_hot = pd.read_csv("https://github.com/byuidatascience/data4dwellings/raw/master/data-raw/dwellings_neighborhoods_ml/dwellings_neighborhoods_ml.csv")  

# limit the number of features included in the model
# X = dwellings_ml.iloc[:,1:24]  #SelectKBest
# X1 = dwellings_ml.iloc[:,21:49] #SelectKBest
# X = dwellings_ml.iloc[:, 1:49] #SelectKBest
# X = dwellings_ml.iloc[:, 1:14] # Extra Trees Classifier

# y includes our labels and x includes our features
# y = dwellings_ml.before1980 # 0 or 1

# Y = dwellings_ml.iloc[:, -1] # 0 or 1 - target variable

dwellings_ml = pd.read_csv("https://github.com/byuidatascience/data4dwellings/raw/master/data-raw/dwellings_ml/dwellings_ml.csv")
dwellings_hot = pd.read_csv("https://github.com/byuidatascience/data4dwellings/raw/master/data-raw/dwellings_neighborhoods_ml/dwellings_neighborhoods_ml.csv")  

Y = dwellings_ml.filter(['parcel', 'before1980'])
X = dwellings_ml.filter(['parcel', 'sprice'])

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.34,random_state=76)
#X_test.head(10)
X_test.head(10).sprice.mean()
Y_test.head(10).astype({'before1980':'int'}).mean()
#Y_test.head(10)

# Y_test.head(10)
#%%

X_test.sprice.mean()

#%%


#############################################################

#%%
df = pd.read_csv("https://github.com/byuidatascience/data4dwellings/raw/master/data-raw/dwellings_ml/dwellings_ml.csv")
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

# feature engineering - limit #12
# df = df.filter(['quality_X', 'status_I', 'status_V','qualified_Q', 'qualified_U'])

# feature engineering - limit #13
# df = df.filter(['quality_X', 'qualified_Q', 'qualified_U'])

# feature engineering - limit #14
df = df.filter(['quality_X', 'status_I', 'status_V'])

# feature engineering - limit #15
df = df.filter(['quality_X', 'status_I', 'status_V'])

# feature engineering - limit #15
df = df.filter(['quality_X', 'status_I', 'status_V'])

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
iterations = 1000
# iterations = 5000
# iterations = 10000
# iterations = 20000
Logistic_Regression(X_train, Y_train, alpha, initial_theta, iterations)

#%%

#############################################################



#%%
# limit the number of features included in the model
# X = dwellings_ml.iloc[:,1:24]  #SelectKBest
# X1 = dwellings_ml.iloc[:,21:49] #SelectKBest
X = dwellings_ml.iloc[:, 1:49] #SelectKBest
# X = dwellings_ml.iloc[:, 1:14] # Extra Trees Classifier

# y includes our labels and x includes our features
# y = dwellings_ml.before1980 # 0 or 1
y = dwellings_ml.iloc[:, -1] # 0 or 1 - target variable

Markdown(X.tail(10).to_markdown())
#%%

#%%
x.describe()
#%%

#%%
x.info()
#%%

#%%
col = dwellings_ml.columns
print(col)
#%%


#%%
y.info()
#%%


#%%
# features = ... select the feature columns from the data frame
# targets = ... select the target column from the data frame

# Randomize and split the samples into two groups. 
# 25% of the samples will be used for testing.
# The other 75% will be used for training.


# features = X.filter(['netprice', 'livearea', 'basement', 'nocars', 'numbdrm',
                    #  'numbaths', 'stories', 'quality_B', 'quality_C', 'condition_AVG', 'quality'])

features = X.filter(['netprice', 'livearea', 'basement', 'stories'
                     'nocars', 'numbdrm', 'numbaths', 'stories', 
                     'quality_B', 'quality_C', 'condition_AVG', 'quality'])


target = y

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=.33) 
#%%

#%%
X_train.info()
#%%

#%%
y_train.info()
#%%

#%%
X_test.info()
#%%

#%%
y_test.info()
#%%

###############################################
# Exhaustive Feature Selection - Random Forest Classifier

#******************************************************

# Be careful running this classifier - it will run for hours

#%%
efs = ExhaustiveFeatureSelector(RandomForestClassifier(), 
                                min_features=4, 
                                max_features=12, 
                                scoring='roc_auc',         
                                cv=2)
RandomForestClassifier(n_jobs=-1), 
print_progress=True, 
scoring='accuracy',

# fit the object to the training data
efs = efs.fit(X_train.sample(500), y.sample(500))

# print the selected features
selected_features = X_train.columns[list(efs.best_idx_)]

# print the final prediction score
featureScores = efs.best_score
featureScores = featureScores.nlargest(15)
featureScores.reset_index(drop=True, inplace=True)
Markdown(featureScores.to_markdown())
#%%

#%%
feature_scores.plot(kind='barh', color='teal')
plt.show()
#%%


###############################################
# Univariate Selection - SelectKBest 1

# ************

#%%
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X_train,y_train)
dfscores = round(pd.DataFrame(fit.scores_), 2)
dfcolumns = pd.DataFrame(X_train.columns)
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']
featureScores = featureScores.nlargest(15,'Score')
featureScores = featureScores[featureScores.Specs != 'yrbuilt']
featureScores.reset_index(drop=True, inplace=True)
featureScores.info()
print(featureScores)
#%%

#%%
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
model.fit(X_train,y_train)
#use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X_train.columns)
feat_importances = feat_importances.nlargest(15)
print(feat_importances)
#%%

#%%
feat_importances.nlargest(20).plot(kind='barh', color='teal')
featureScores.reset_index(drop=True, inplace=True)
plt.show()
#%%

#%%
###################################################
# Fisher's Score

# Calculating Scores

# score = fisher_score.fisher_score(X.to_numpy(), y.to_numpy())
ranks = fisher_score.fisher_score(X_train, y_train)

print(ranks.head(10))
#%%

#%%
# Plotting the ranks of features
feat_importances = pd.Series(ranks, X_train.columns[0:len(dwellings_ml.columns)-1])
feat_importances.plot(kind='barh', color='teal')
plt.show()
########################################################
#%% 

#%%
#######################################################
# Mutual Information Method
#mutual information selecting all features

mutual = SelectKBest(score_func=mutual_info_classif, k='all')
#learn relationship from training data
mutual.fit(X_train, y_train)
# transform train input data
X_train_mut = mutual.transform(X_train)
# transform test input data
X_test_mut = mutual.transform(X_test)
#printing scores of the features
# for i in range(len(mutual.scores_)):
#     print('Feature %d: %f' % (i, mutual.scores_[i]))
print(mutual.scores_)
#%% 
 
####################################################
# run the GaussianNB classifier
#%%
classifier = GaussianNB()
classifier.fit(X_train, y_train)
GaussianNB()
print("Naive Bayes score: ", classifier.score(X_test, y_test))
#%%

# run the GaussianNB test data
#%%
# targets_pred = classifier.predict(X_test)
print("Naive Bayes score: ", classifier.score(X_test, y_test))
#%%

#########################################################
# run the Logistic Regression classifier
#%%
reg = LogisticRegression()
# train the model using the training sets
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)
score =accuracy_score(y_test,y_pred)
print(score)
#%%

#######################################################
# Correlation Matrix with Scatterplot & Heatmap
# run for 5 different feature sets

#************

# first ten features
#%%

# feature set 1
# h_subset = dwellings_ml.filter(['livearea', 'finbsmnt', 'basement', 
#     'yearbuilt', 'nocars', 'numbdrm', 'numbaths', 'before1980',
#     'stories', 'yrbuilt']).sample(500)

h_subset = X_train.sample(500)
#%%

#%%

# pairs plot 1 - aka scatterplot matrix
sns.pairplot(h_subset, hue='before1980')
#%%

#%%
corr = h_subset.drop(columns='before1980').corr()
#%%

#%%

# heatmap 1
sns.heatmap(corr, annot = True)
#%%




#######################################################


# Violin plot
#%%
# first ten features
data_dia = y
data = x
data_n_2 = (data - data.mean()) / (data.std())              # standardization
data = pd.concat([y,data_n_2.iloc[:,0:10]],axis=1)
data = pd.melt(data,id_vars="before1980",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", hue="before1980", data=data,split=True, inner="quartile")
plt.xticks(rotation=90)
#%%

# Swarm plot
#%%
sns.set(style="whitegrid", palette="muted")
data_dia = y
data = x
data_n_2 = (data - data.mean()) / (data.std())              # standardization
data = pd.concat([y,data_n_2.iloc[:,0:10]],axis=1)
data = pd.melt(data,id_vars="before1980",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
tic = time.time()
sns.swarmplot(x="features", y="value", hue="before1980", data=data)

plt.xticks(rotation=90)
#%%



#%%
alt.Chart(dwellings_denver).mark_point(color="purple")\
    .encode(x = alt.X('yrbuilt', scale=alt.Scale(domain=[1860, 2020]), sort=alt.SortField('yrbuilt')),\
            y = alt.Y('nbhd', scale=alt.Scale(domain=[0, 1000])),)\
    .properties(width=800, height=800)
#%%


#%%
    
# alt.Chart(dwellings_ml).mark_point(color="orange")\
#     .encode(x = alt.X('nbhd', scale=alt.Scale(domain=[0, 1]), sort=alt.SortField('yrbuilt')),\
#             y = alt.Y('AVG(sprice)', scale=alt.Scale(domain=[0, 20])),)\
#     .properties(width=600, height=300)
    
#%%


#%%
# convert negative values to 0
num_cols = dwellings_denver._get_numeric_data()
num_cols[num_cols < 0 ] = 0
num_cols = dwellings_ml._get_numeric_data()
num_cols[num_cols < 0 ] = 0
num_cols = dwellings_hot._get_numeric_data()
num_cols[num_cols < 0 ] = 0
#%%

#%%
dwellings_denver.replace('', np.nan, inplace=True)
dwellings_denver.replace(0, np.nan, inplace=True)
dwellings_denver.replace("n/a", np.nan, inplace=True)
dwellings_denver.replace("N/A", np.nan, inplace=True)
dwellings_denver.replace("NA", np.nan, inplace=True)
dwellings_denver.replace("?", np.nan, inplace=True)
dwellings_denver.reset_index(drop=True, inplace=True)
#%%

#%%
dwellings_ml.replace('', np.nan, inplace=True)
dwellings_ml.replace(0, np.nan, inplace=True)
dwellings_ml.replace("n/a", np.nan, inplace=True)
dwellings_ml.replace("N/A", np.nan, inplace=True)
dwellings_ml.replace("NA", np.nan, inplace=True)
dwellings_ml.replace("?", np.nan, inplace=True)
dwellings_ml.reset_index(drop=True, inplace=True)
#%%

#%%
dwellings_hot.replace('', np.nan, inplace=True)
dwellings_hot.replace(0, np.nan, inplace=True)
dwellings_hot.replace("n/a", np.nan, inplace=True)
dwellings_hot.replace("N/A", np.nan, inplace=True)
dwellings_hot.replace("NA", np.nan, inplace=True)
dwellings_hot.replace("?", np.nan, inplace=True)
dwellings_hot.reset_index(drop=True, inplace=True)
#%%

# # clean up the NaN values => convert them to ' 0 '
# mydat = mydat.fillna(0)

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
missing_checks(dwellings_denver, 'parcel')
missing_checks(dwellings_denver, 'nbhd')
missing_checks(dwellings_denver, 'livearea')
missing_checks(dwellings_denver, 'finbsmnt')
missing_checks(dwellings_denver, 'yrbuilt')
missing_checks(dwellings_denver, 'condition')
missing_checks(dwellings_denver, 'quality')
missing_checks(dwellings_denver, 'totunits')
missing_checks(dwellings_denver, 'stories')
missing_checks(dwellings_denver, 'gartype')
missing_checks(dwellings_denver, 'nocars')
missing_checks(dwellings_denver, 'floorlvl')
missing_checks(dwellings_denver, 'numbdrm')
missing_checks(dwellings_denver, 'numbaths')
missing_checks(dwellings_denver, 'arcstyle')
missing_checks(dwellings_denver, 'sprice')
missing_checks(dwellings_denver, 'deduct')
missing_checks(dwellings_denver, 'netprice')
missing_checks(dwellings_denver, 'tasp')
missing_checks(dwellings_denver, 'smonth')
missing_checks(dwellings_denver, 'syear')
missing_checks(dwellings_denver, 'qualified')
missing_checks(dwellings_denver, 'status')

#%%


#%%
missing_checks(dwellings_ml, 'parcel')
missing_checks(dwellings_ml, 'livearea')
missing_checks(dwellings_ml, 'finbsmnt')
missing_checks(dwellings_ml, 'basement')
missing_checks(dwellings_ml, 'yrbuilt')
missing_checks(dwellings_ml, 'totunits')
missing_checks(dwellings_ml, 'stories')
missing_checks(dwellings_ml, 'nocars')
missing_checks(dwellings_ml, 'numbdrm')
missing_checks(dwellings_ml, 'numbaths')
missing_checks(dwellings_ml, 'sprice')
missing_checks(dwellings_ml, 'deduct')
missing_checks(dwellings_ml, 'netprice')
missing_checks(dwellings_ml, 'tasp')
missing_checks(dwellings_ml, 'smonth')
missing_checks(dwellings_ml, 'syear')
missing_checks(dwellings_ml, 'condition_AVG')
missing_checks(dwellings_ml, 'condition_Excel')
missing_checks(dwellings_ml, 'condition_Fair')
missing_checks(dwellings_ml, 'condition_Good')
missing_checks(dwellings_ml, 'condition_VGood')
missing_checks(dwellings_ml, 'quality_A')
missing_checks(dwellings_ml, 'quality_B')
missing_checks(dwellings_ml, 'quality_C')
missing_checks(dwellings_ml, 'quality_D')
missing_checks(dwellings_ml, 'quality_X')
missing_checks(dwellings_ml, 'gartype_Att')
missing_checks(dwellings_ml, 'gartype_Att/Det')
missing_checks(dwellings_ml, 'gartype_CP')
missing_checks(dwellings_ml, 'gartype_Det')
missing_checks(dwellings_ml, 'gartype_None')
missing_checks(dwellings_ml, 'gartype_att/CP')
missing_checks(dwellings_ml, 'gartype_det/CP')
missing_checks(dwellings_ml, 'arcstyle_BI-LEVEL')
missing_checks(dwellings_ml, 'arcstyle_CONVERSIONS')
missing_checks(dwellings_ml, 'arcstyle_END UNIT')
missing_checks(dwellings_ml, 'arcstyle_MIDDLE UNIT')
missing_checks(dwellings_ml, 'arcstyle_ONE AND HALF-STORY')
missing_checks(dwellings_ml, 'arcstyle_ONE-STORY')
missing_checks(dwellings_ml, 'arcstyle_SPLIT LEVEL')
missing_checks(dwellings_ml, 'arcstyle_THREE-STORY')
missing_checks(dwellings_ml, 'arcstyle_TRI-LEVEL')
missing_checks(dwellings_ml, 'arcstyle_TRI-LEVEL WITH BASEMENT')
missing_checks(dwellings_ml, 'arcstyle_TWO AND HALF-STORY')
missing_checks(dwellings_ml, 'arcstyle_TWO-STORY')
missing_checks(dwellings_ml, 'qualified_Q')
missing_checks(dwellings_ml, 'qualified_U')
missing_checks(dwellings_ml, 'status_I')
missing_checks(dwellings_ml, 'status_V')
missing_checks(dwellings_ml, 'before1980')

#%%


#%%
dwelings_denver_old = dwellings_denver.query('yrbuilt < 1980')
# dwelings_denver_old = dwellings_denver.query('yrbuilt > 1700 and yrbuilt < 2022')
# dwelings_denver_old = dwellings_denver.query('yrbuilt == "N/A"')
# dwelings_denver_old = dwellings_denver.query('yrbuilt == "n/a"')
# dwelings_denver_old = dwellings_denver.query('yrbuilt == "NA"')
# dwelings_denver_old = dwellings_denver.query('yrbuilt == NaN')
# dwelings_denver_old = dwellings_denver.query('yrbuilt == nan')
# dwelings_denver_old.info()
dwelings_denver_old.describe()
# dwelings_denver_old.tail(-50)
# Markdown(dwelings_denver_old.head(100).to_markdown())
#%%


#%%
dwelings_ml_old = dwellings_ml.query('yrbuilt < 1980')
# dwelings_ml_old = dwellings_ml.query('yrbuilt == "N/A"')
# dwelings_ml_old = dwellings_ml.query('yrbuilt == "n/a"')
# dwelings_ml_old = dwellings_ml.query('yrbuilt == "NA"')
# dwelings_ml_old = dwellings_ml.query('yrbuilt == NaN')
# dwelings_ml_old = dwellings_ml.query('yrbuilt == nan')
# dwelings_ml_old.info()
dwelings_ml_old.describe()
# dwelings_ml_old.tail(-50)
# Markdown(dwelings_ml_old.head(100).to_markdown())
#%%


#%%
# dwelings_hot_nyr = dwellings_hot.query('yrbuilt != ""')
# dwelings_hot_nyr.info()
dwellings_hot_old.describe()
Markdown(dwellings_hot.old.head(100).to_markdown())
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

#%%


#%%


#%%


#%%


## GRAND QUESTION 2

'''
Build a classification model labeling houses as being built “before 1980” or “during or 
after 1980”. Your goal is to reach or exceed 90% accuracy. Explain your final model 
choice (algorithm, tuning parameters, etc) and describe what other models you tried.
'''


#%%


#%%


#%%

#%%


## GRAND QUESTION 3

'''
Justify your classification model by discussing the most important features selected by 
your model. This discussion should include a chart and a description of the features.
'''

#%%

#%%


#%%

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