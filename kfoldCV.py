#!/usr/bin/env python3

import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict


if __name__ == '__main__':
    """ 1.2.2 Quantitative analysis """
    df = pd.read_csv('data.csv')
    prices = df['MEDV']
    features = df.drop('MEDV', axis = 1)

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
        """ K-Fold Cross Validation """

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
        print("K-Fold CV error : "+str(e))
        pass
    sys.exit(0)
