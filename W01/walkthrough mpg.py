import pandas as pd   
import altair as alt
import numpy as np

from IPython.display import Markdown
from IPython.display import display
from tabulate import tabulate

#npm config set python python3.8
# npm install -g vega vega-lite vega-cli canvas
# !{sys.executable} -m pip install altair_saver
# !{sys.executable} -m pip install tabulate
#%%

# import sys
# !{sys.executable} -m pip install pandas altair numpy --user
#%%
# import sys
# !{sys.executable} -m pip install --upgrade pip --user
# %%
#alt.data_transformers.enable('json')

# %%
# mpg data 
url = "https://github.com/byuidatascience/data4python4ds/raw/master/data-raw/mpg/mpg.csv"
mpg = pd.read_csv(url)

# %%
# plot the displacement chart

chart = (alt.Chart(mpg)
  .encode(
    x='displ', 
    y='hwy')
  .mark_circle()
)

chart
chart.save("screenshots/altair_viz_1_displ.html")
chart.save("screenshots/altair_viz_1_displ.json")

# %%
print()
print(mpg
  .head(5)
  .filter(["manufacturer", "model","year", "hwy", "displ"])
  .to_markdown(index=False))
print()

# plot the # Cylinders chart

chart = (alt.Chart(mpg)
  .encode(
    x='displ', 
    y='hwy')
  .mark_circle()
)

chart
chart.save("screenshots/altair_viz_2_cyl.html")
chart.save("screenshots/altair_viz_2_cyl.json")

# %%
print()
print(mpg
  .head(5)
  .filter(["manufacturer", "model","year", "hwy", "cyl"])
  .to_markdown(index=False))
print()

# plot the Transmission Type chart

chart = (alt.Chart(mpg)
  .encode(
    x='displ', 
    y='hwy')
  .mark_circle()
)

chart
chart.save("screenshots/altair_viz_3_trans.html")
chart.save("screenshots/altair_viz_3.trans.json")

# %%
print()
print(mpg
  .head(5)
  .filter(["manufacturer", "model","year", "hwy", "trans"])
  .to_markdown(index=False))
print()


# %%

