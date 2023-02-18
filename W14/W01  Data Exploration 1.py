

#%%

import pandas as pd
import altair as alt
import seaborn as sns
sns.set_theme(style="whitegrid")
from IPython.display import Markdown, display
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/netflix_titles.csv"

netflix_df = pandas.read_csv(url)
netflix_df = netflix_df[netflix_df['type'] == "Movie"]
ratings = ['G', 'PG', 'PG-13', 'R', 'NC-17']
netflix_df = netflix_df[netflix_df['rating'].isin(ratings)]

#%%%


#%%
alt.Chart(netflix_df).mark_bar().encode(
    x='rating',
    y='count(rating)'
)

#%%

#%%
sns.countplot(x=netflix_df["rating"])
#%%

#%%
# Use pandas' built in plotting functions to create a count plot comparing the count of each movie rating
# This will be a little trickier than the other libraries, but one hint is that the pandas value_counts() function
# actually returns a dataframe.

box_df = netflix_df[ ['rating'] ].plot.box()

#%%%


