import csv
# import matplotlib.pyplot as plt
# import seaborn as sns

import pandas as pd

fileName = 'data.csv'

df = pd.read_csv("./data.csv", sep=",")

print(df)