---
# W3 - U1 Week B # 1 - Class Code Walkthrough

title: "Client Report - W01 Project )0: Introduction - MPG"
subtitle: "Course DS 250"
author: "Scott LeFoll"
Date: "09/17/2022"
format:
  html:
    self-contained: true
    page-layout: full
    title-block-banner: true
    toc: true
    toc-depth: 3
    toc-location: body
    number-sections: false
    html-math-method: katex
    code-fold: true
    code-summary: "Show the code"
    code-overflow: wrap
    code-copy: hover
    code-tools:
        source: false
        toggle: true
        caption: See code
---


```{python}

# W3 - U1 Week B #1- Class Code Walkthrough

#| label: libraries
#| include: false
import pandas as pd
import altair as alt
import numpy as np
import matplotlib.pyplot as plt
import altair_saver as alt_saver
# alt.data_transformers.enable('json')

from altair_saver import save
from IPython.display import Markdown
from IPython.display import display
from tabulate import tabulate
```


## Elevator pitch

The importance ...


```{python}
#| label: W01 Project 0 Data
#| code-summary: Read and format project data
# Include and execute your code here

#what is a dataframe
df = pd.DataFrame(
{"a" : [4 ,5, 6],
"b" : [7, 8, 9],
"c" : [10, 11, 12]},
index = [1, 2, 3])
print(df)

df_alt = pd.DataFrame(np.array(((
  [4 ,5, 6],
  [7, 8, 9],
  [10, 11, 12]))),
  index=[1, 2, 3],
  columns=['a','b','c'])
df_alt

url = 'https://github.com/byuidatascience/data4soils/raw/master/data-raw/cfbp_handgrenade/cfbp_handgrenade.csv'
dat = pd.read_csv(url)
dat
```



## GRAND QUESTION 1

W01 Task 1: Finish the readings and be prepared with any questions to get your environment working smoothly (class for on-campus and Slack for online).

W01 Task 2: In VS Code, write a python script to create the example Altair chart from 
section 3.2.2 of the textbook (part of the assigned readings). Note that you 
have to type chart to see the Altair chart after you create it.

W01 Task 3 Your final report should also include the markdown table.

Grand Question 1: What is the relationship between fuel efficiency and engine displacement among popular models in the years 1999 - 2008?


```{python}
#| label: GQ1
#| code-summary: Build Chart


#how to reference a value
df['a']
df.loc[1,'a']

#use of head with a aggregation
means = df.sort_values("a", ascending=False).head(2).c.mean()
means

#use of filter and query
df
#%%
(df.rename(columns = {'a':'duck'})
  .filter(['duck', 'b'])
  .query('b < 9')
  .duck
  .min()
)


#read in a dataset to a df
#chart it 
url = "https://github.com/byuidatascience/data4python4ds/raw/master/data-raw/mpg/mpg.csv"

mpg = pd.read_csv(url)

chart_loess = (alt.Chart(mpg)
  .encode(
    x = "displ",
    y = "hwy")
  .transform_loess("displ", "hwy")
  .mark_line()
)

chart_loess

# plot each samples hmx and rdx value
alt.Chart(dat).encode(x = 'hmx', y = 'rdx').mark_circle()

# alt.Chart(dat.head(200))\
#     .encode(x="displ", y="hwy")\
#     .mark_bar()\
#     .properties(
#         width=800,
#         height=300
#     )

```



```{python}
#| label: GQ2 chart
#| code-summary: Save chart 1 - Displacement vs. Highway MPG
#| fig-cap: "Popular Models, 1999-2008: Displacement vs. Highway MPG"
#| fig-align: center
# plot the mpg chart

# plot the hmx on their grid layout with a better color
chart = (alt.Chart(dat)
    .encode(
        x = 'row', 
        y = 'column', 
        color = alt.Color('hmx',
        scale = alt.Scale(scheme = 'goldorange')))
    .mark_square(size = 500))
chart
# chart.save("altair_viz_1.png")

# alt.Chart(dat.head(200))\
#     .encode(x = "displ", y = "hwy")\
#     .mark_bar()

# chart
# chart.save("screenshots/altair_viz_1_displ.html")
# chart.save("screenshots/altair_viz_1_displ.json")



```



```{python}
#| label: GQ3 table
#| code-summary: Display Table 1 in Terminal and HTML v.1
#| tbl-cap: "Popular Models Highway Fuel Efficiency"
#| tbl-cap-location: top
# Include and execute your code here

# Make a histogram of hmx and rdx
(alt.Chart(dat)
    .encode(
        x = alt.X('hmx', bin = alt.Bin(step = 0.05)), 
        y = 'count()')
    .mark_bar(color = 'red')
    .configure_title(fontSize = 20)
    .properties(title = "Distribution of HMX soild samples")

)
# print()
# print(mpg
#   .head(5)
#   .filter(["manufacturer", "model","year", "hwy", "displ"])
#   .to_markdown(index=False))
# # print()



```



```{python}
#| label: GQ3B chart
#| code-summary: Display Table 1 in HTML v.3
# Include and execute your code here

print(dat.head(3).to_markdown(showindex = False))

# Markdown(mydat.to_markdown(index=False))
```



```{python}
#| label: GQ3B chart
#| code-summary: Display Table 1 in HTML v.3
# Include and execute your code here

s = pd.Series(["elk", "pig", "dog", "quetzal"], name="animal")

# Markdown(mydat.to_markdown(index=False))
```


```{python}
#| label: GQ3B chart
#| code-summary: Display Table 1 in HTML v.3
# Include and execute your code here

print(s.to_markdown())

# Markdown(mydat.to_markdown(index=False))
```

