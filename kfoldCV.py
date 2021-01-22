#!/usr/bin/env python3

import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

""" 
    ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
    CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
    RM       average number of rooms per dwelling
    DIS      weighted distances to five Boston employment centres
    B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
"""

from keras import models
from keras import layers

def build_model(y_train):
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',input_shape=(y_train.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

if __name__ == '__main__':
    df = pd.read_csv('data.csv')
    prices = df['MEDV']
    features = df.drop('MEDV', axis = 1)

    try:
        """ Normalisation des datas """
        data = preprocessing.scale(features)
    except Exception as e:
        print(e)
        pass

    try:
        """ Print quelques stats """

        minimum_price = np.min(prices)
        maximum_price = np.max(prices)
        mean_price = np.mean(prices)
        median_price = np.median(prices)
        std_price = np.std(prices)
        first_quartile = np.percentile(prices, 25)
        third_quartile = np.percentile(prices, 75)
        inter_quartile = third_quartile - first_quartile
        print("Some Stats ($1000's):\n")
        print("Minimum price: ${:,.2f}".format(minimum_price*1000))
        print("Maximum price: ${:,.2f}".format(maximum_price*1000))
        print("Mean price: ${:,.2f}".format(mean_price*1000))
        print("Median price ${:,.2f}".format(median_price*1000))
        print("Standard deviation of prices: ${:,.2f}".format(std_price*1000))
        print("First quartile of prices: ${:,.2f}".format(first_quartile*1000))
        print("Second quartile of prices: ${:,.2f}".format(third_quartile*1000))
        print("Interquartile (IQR) of prices: ${:,.2f}".format(inter_quartile*1000))
    except Exception as e:
        print('Stat error : '+str(e))
        pass

    try:
        """ Corrélations des datas : pour choisir quelles features sont les + pertinentes """

        snsPlot = sns.heatmap(df.corr().round(2),cmap='coolwarm',annot=True)
        fig = snsPlot.get_figure()
        fig.savefig("./Prediction/correlations.png")
    except Exception as e:
        print('Correlation error : '+str(e))
        pass

    try:
        """ Moyenne (trait plein) et Mediane (pointillés) """

        clr = ['blue', 'green', 'red', 'yellow', 'orange', 'purple']
        plt.figure(1)
        for i, var in enumerate(['ZN', 'CHAS', 'RM', 'DIS', 'B']): #, 'DIS', 'B'
            sns.displot(df[var],  color = clr[i])
            plt.axvline(df[var].mean(), color=clr[5], linestyle='solid', linewidth=2)
            plt.axvline(df[var].median(), color=clr[5], linestyle='dashed', linewidth=2)
            plt.savefig('./Prediction/{0}.png'.format(var))
    except Exception as e:
        print('K Fold CV error : '+str(e))
        pass

    # try:
    #     """ Scatter plots Prices vs fetures """
    #     for i, var in enumerate(['ZN', 'CHAS', 'RM', 'DIS', 'B']):
    #         fig1=[]
    #         lm = sns.regplot(df[var], prices, ax = None, color=clr[i])
    #         lm.set(ylim=(0, 100))
    #         fig1.append(lm.get_figure())
    #         fig1[0].savefig('./Prediction/{0}PriceTrend.png'.format(var))
    # except Exception as e:
    #     print('Scatter plots error : '+str(e))
    #     pass

    try:
        SelectedColumns=['ZN', 'CHAS', 'RM', 'DIS', 'B']
        X_train, X_test, y_train, y_test = train_test_split(prices, features, test_size = 0.2, random_state = 0)
        print('done')
    except Exception as e:
        print(e)
        pass
    sys.exit(0)
