import pandas as pd
import plotly.express as px

df = pd.read_csv("./data.csv", sep=",")

fig = px.parallel_coordinates(df, color="MEDV",)
fig.show()
