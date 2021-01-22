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
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

""" 
    ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
    CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
    RM       average number of rooms per dwelling
    DIS      weighted distances to five Boston employment centres
    B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
"""

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

    # try:
    #     """ Corrélations des datas : pour choisir quelles features sont les + pertinentes """

    #     snsPlot = sns.heatmap(df.corr().round(2),cmap='coolwarm',annot=True)
    #     fig = snsPlot.get_figure()
    #     fig.savefig("./Prediction/correlations.png")
    # except Exception as e:
    #     print('Correlation error : '+str(e))
    #     pass

    # try:
    #     """ Moyenne (trait plein) et Mediane (pointillés) """

    #     clr = ['blue', 'green', 'red', 'yellow', 'orange', 'purple']
    #     plt.figure(1)
    #     for i, var in enumerate(['ZN', 'CHAS', 'RM', 'DIS', 'B']): #, 'DIS', 'B'
    #         sns.displot(df[var],  color = clr[i])
    #         plt.axvline(df[var].mean(), color=clr[5], linestyle='solid', linewidth=2)
    #         plt.axvline(df[var].median(), color=clr[5], linestyle='dashed', linewidth=2)
    #         plt.savefig('./Prediction/{0}.png'.format(var))
    # except Exception as e:
    #     print('K Fold CV error : '+str(e))
    #     pass

    # # try:
    # #     """ Scatter plots Prices vs fetures """
    # #     for i, var in enumerate(['ZN', 'CHAS', 'RM', 'DIS', 'B']):
    # #         fig1=[]
    # #         lm = sns.regplot(df[var], prices, ax = None, color=clr[i])
    # #         lm.set(ylim=(0, 100))
    # #         fig1.append(lm.get_figure())
    # #         fig1[0].savefig('./Prediction/{0}PriceTrend.png'.format(var))
    # # except Exception as e:
    # #     print('Scatter plots error : '+str(e))
    # #     pass

    try:
        SelectedFeatures=['RM', 'LSTAT', 'PTRATIO']
        X = df[SelectedFeatures]
        y = prices
        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 0)
        lin_model = LinearRegression()
        lin_model.fit(train_X, train_y)
        y_train_predict = lin_model.predict(train_X)
        rmse = (np.sqrt(mean_squared_error(train_y, y_train_predict)))
        r2 = r2_score(train_y, y_train_predict)
        print("\n")
        print("The model performance for training set")
        print("--------------------------------------")
        print('RMSE is {}'.format(rmse))
        print('R2 score is {}'.format(r2))
        print("\n")
        y_test_predict = lin_model.predict(test_X)
        rmse = (np.sqrt(mean_squared_error(test_y, y_test_predict)))
        r2 = r2_score(test_y, y_test_predict)
        print("\n")
        print("The model performance for testing set")
        print("--------------------------------------")
        print('RMSE is {}'.format(rmse))
        print('R2 score is {}'.format(r2))
        print("\n")

        predicted = cross_val_predict(lin_model, X, y,cv=10)
        fig,ax = plt.subplots()
        ax.scatter(y, predicted, edgecolors=(0,0,0))
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
        ax.set_xlabel('Mesured')
        ax.set_ylabel('Predicted')
        plt.savefig('res.png')
    except Exception as e:
        print(e)
        pass
    sys.exit(0)
