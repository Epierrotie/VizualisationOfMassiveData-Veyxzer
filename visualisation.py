import csv

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# sns.set(style="ticks")

df = pd.read_csv("./data.csv", sep=",")

# sns.pairplot(df)

# plt.savefig('ScatterMatrix')
print(df.columns)

sns.PairGrid(df, hue='B', vars=df.columns, corner=True).map(plt.scatter)

plt.savefig('test')
