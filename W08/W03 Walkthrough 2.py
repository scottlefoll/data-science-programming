---
# W03 U1 Week B # 2
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
# W03 U1 Week B # 2

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

url = 'https://github.com/byuidatascience/data4names/raw/master/data-raw/names_year/names_year.csv'
dat = pd.read_csv(url)
dat.tail(-5)
```



## GRAND QUESTION 1



```{python}
#| label: GQ1
#| code-summary: Build Chart

#how to confirm you got all the info 
#what is unique do, what does size do
pd.unique(dat.name).size
dat.name.size

```



```{python}
#| label: GQ2 chart
#| code-summary: Save chart 1 - Displacement vs. Highway MPG
#| fig-cap: "Popular Models, 1999-2008: Displacement vs. Highway MPG"
#| fig-align: center
# plot the mpg chart

#how to use query with aggregation min, max, size
pd.unique(dat.query('name == "John"').year).min()


```



```{python}
#| label: GQ3 table
#| code-summary: Display Table 1 in Terminal and HTML v.1
#| tbl-cap: "Popular Models Highway Fuel Efficiency"
#| tbl-cap-location: top
# Include and execute your code here

pd.unique(dat.query('name == "John"').year).max()
```



```{python}
#| label: GQ3B chart
#| code-summary: Display Table 1 in HTML v.3
# Include and execute your code here

pd.unique(dat.query('name == "John"').year).size

```



```{python}
#| label: GQ3B chart
#| code-summary: Display Table 1 in HTML v.3
# Include and execute your code here

#what is group by and agg
dat_total = dat.groupby('name').agg(n = ('Total', 'sum')).reset_index()
dat_total
```


```{python}
#| label: GQ3B chart
#| code-summary: Display Table 1 in HTML v.3
# Include and execute your code here
#what is group by and agg
dat_total = dat.groupby('name').Total.max().reset_index()
dat_total

```


```{python}
#| label: GQ3B chart
#| code-summary: Display Table 1 in HTML v.3
# Include and execute your code here
#what is group by and agg

dat_total = dat.groupby('name').agg(n = ('Total', 'sum')).reset_index()
print(dat_total
    .query('n in [@dat_total.n.max(), @dat_total.n.min()]')
    .to_markdown(index = False, floatfmt=".0f"))
```

