import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set(style="ticks")

df = pd.read_csv("./data.csv", sep=",")

# sns.pairplot(df)

# plt.savefig('ScatterMatrix')
columns = list(df.columns)

sns.PairGrid(df, hue='MEDV', vars=columns, corner=True).map(plt.scatter)

title = 'Scatter matrix of the Boston Housing dataset'
filename = 'housing_scatter_matrix.pdf'
plt.title(title)
plt.savefig(filename)
