# %%
import pandas as pd 
import numpy as np 
import altair as alt 

# %%
# Upgrade pip
import sys
!{sys.executable} -m pip install --upgrade pip

# %%
# Now our new packages
# https://byuistats.github.io/CSE250-Course/course-materials/sql-for-data-science/
import sys
!{sys.executable} -m pip install datadotworld

# %%
import datadotworld as dw
#%%
dw config 
~/.dw/
#%%
export DW_AUTH_TOKEN=eyJhbGciOiJIUzUxMiJ9.eyJzdWIiOiJwcm9kLXVzZXItY2xpZW50OmNoYXpjbGFyayIsImlzcyI6ImFnZW50OmNoYXpjbGFyazo6NDFlMTEyODctNjFmYi00MzQ0LWEzY2MtMjMwOGRhNzkwOGIxIiwiaWF0IjoxNjIyMDAyOTk4LCJyb2xlIjpbInVzZXJfYXBpX3JlYWQiLCJ1c2VyX2FwaV93cml0ZSJdLCJnZW5lcmFsLXB1cnBvc2UiOnRydWUsInNhbWwiOnt9fQ.2hSf-pIUqUUC7gjoBT4ktNk6VD6E3wIdDxMfdj163VYk8yF1YkbbF0QefkJ7DVFTMdtlyHFzaRBXyJMryxYISg

# %%
results = dw.query('byuidss/cse-250-baseball-database', 
    'SELECT * FROM batting LIMIT 100')

batting5 = results.dataframe

# %%
# Now our new packages
# https://byuistats.github.io/CSE250-Course/course-materials/sql-for-data-science/
import sys
!{sys.executable} -m dw configure

# %%
batting5

# %%
# What columns do we want to select?
q = '''
SELECT  playerid, 
        teamid, 
        ab, 
        r
FROM batting
LIMIT 5
'''

dw.query('byuidss/cse-250-baseball-database', q).dataframe

# %%
# What calculation do we want to perform?

q = '''
SELECT playerid, teamid, ab, r, ab/r 
FROM batting
LIMIT 5
'''

batting_calc = (dw
    .query('byuidss/cse-250-baseball-database', q)
    .dataframe)

batting_calc
# %%

# What name do we give our calculated column?

q = '''
SELECT teamid, Sum(r) as Total_Runs
FROM batting
GROUP BY teamid
LIMIT 10000
'''

batting_calc = (dw
    .query('byuidss/cse-250-baseball-database', q)
    .dataframe)
batting_calc
# %%