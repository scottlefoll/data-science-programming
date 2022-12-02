import pandas as pd   
import altair as alt
from altair_saver import save
import numpy as np

url = "https://github.com/byuidatascience/data4python4ds/raw/master/data-raw/mpg/mpg.csv"
mpg = pd.read_csv(url)


chart = (alt.Chart(mpg)
  .encode(
    x='displ', 
    y='hwy')
  .mark_circle()
)

chart
chart.save("screenshots/altair_viz_1.html")
chart.save("screenshots/altair_viz_1.json")

print()
print(mpg
  .head(5)
  .filter(["manufacturer", "model","year", "hwy"])
  .to_markdown(index=False))
print()

# save(chart, "screenshots/altair_viz_1.png") 