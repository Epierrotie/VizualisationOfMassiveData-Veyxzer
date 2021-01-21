import csv

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

fileName = 'data.csv'

df = pd.read_csv("./data.csv", sep=",")

sns.pairplot(df, hue="MEDV")

title = 'Scatter matrix of the houses value'
filename = 'test.pdf'

plt.title(title)
plt.savefig(filename)