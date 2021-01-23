#!/usr/bin/env python3

import sys
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict


def main(args):
    df = pd.read_csv('data.csv')

    selected_attributes = args.attributes or ['RM', 'LSTAT', 'PTRATIO']
    selected_attributes = [attribute.upper()
                           for attribute in selected_attributes]

    selected_start = args.start if args.start and args.start >= 0 and args.start < len(
        df.index) else 0
    selected_end = args.end if args.end and args.end < len(
        df.index) else len(df.index)

    if args.ascending:
        df = df.sort_values(by=args.ascending.upper(), ascending=True)
    if args.descending:
        df = df.sort_values(by=args.descending.upper(), ascending=False)

    arg_predict = args.predict.upper()
    corr_df = df[selected_attributes + [arg_predict]
                 ].iloc[selected_start:selected_end]
    prediction_col = df[arg_predict].iloc[selected_start:selected_end]
    df = df[selected_attributes].iloc[selected_start:selected_end]

    try:
        """ Print quelques stats """
        minimum_predict = np.min(prediction_col)
        maximum_predict = np.max(prediction_col)
        mean_predict = np.mean(prediction_col)
        median_predict = np.median(prediction_col)
        std_predict = np.std(prediction_col)
        first_quartile = np.percentile(prediction_col, 25)
        third_quartile = np.percentile(prediction_col, 75)
        inter_quartile = third_quartile - first_quartile
        print("Predicted attribut stats:\n")
        print("\tMinimum {}: {:,.2f}".format(
            arg_predict, minimum_predict * 1000))
        print("\tMaximum {}: {:,.2f}".format(
            arg_predict, maximum_predict * 1000))
        print("\tMean {}: {:,.2f}".format(arg_predict, mean_predict * 1000))
        print("\tMedian {}: {:,.2f}".format(
            arg_predict, median_predict * 1000))
        print("\tStandard deviation of {}: {:,.2f}".format(
            arg_predict, std_predict * 1000))
        print("\tFirst quartile of {}: {:,.2f}".format(
            arg_predict, first_quartile * 1000))
        print("\tSecond quartile of {}: {:,.2f}".format(
            arg_predict, third_quartile * 1000))
        print("\tInterquartile (IQR) of {}: {:,.2f}\n".format(
            arg_predict, inter_quartile * 1000))
    except Exception as e:
        print('Stat error : ' + str(e))
        pass

    try:
        """ Corrélations des datas : pour choisir quelles features sont les + pertinentes """

        snsPlot = sns.heatmap(corr_df.corr().round(2),
                              cmap='coolwarm', annot=True)
        fig = snsPlot.get_figure()
        fig.savefig("correlations.png")
    except Exception as e:
        print('Correlation error : ' + str(e))
        pass

    try:
        """ Train-Test Validation """

        X = df[selected_attributes]
        y = prediction_col
        train_X, test_X, train_y, test_y = train_test_split(
            X, y, test_size=0.2, random_state=0)
        lin_model = LinearRegression()
        lin_model.fit(train_X, train_y)
        y_train_predict = lin_model.predict(train_X)
        rmse = (np.sqrt(mean_squared_error(train_y, y_train_predict)))
        r2 = r2_score(train_y, y_train_predict)
        print("\nThe model performance for training set")
        print("--------------------------------------")
        print('RMSE is {}'.format(rmse))
        print('R2 score is {}\n'.format(r2))
        y_test_predict = lin_model.predict(test_X)
        rmse = (np.sqrt(mean_squared_error(test_y, y_test_predict)))
        r2 = r2_score(test_y, y_test_predict)
        print("The model performance for testing set")
        print("--------------------------------------")
        print('RMSE is {}'.format(rmse))
        print('R2 score is {}\n'.format(r2))

        predicted = cross_val_predict(lin_model, X, y, cv=10)
        fig, ax = plt.subplots()
        ax.scatter(y, predicted, edgecolors=(0, 0, 0))
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
        ax.set_xlabel('Mesured')
        ax.set_ylabel('Predicted')
        plt.savefig('prediction_{}.png'.format(args.predict))
    except Exception as e:
        print("Train-Test error : " + str(e))
        pass
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Program that generate a prediction')
    parser.add_argument('-a', '--attributes', nargs='*',
                        help='Attributes you want to use in the prediction')

    sort_group = parser.add_mutually_exclusive_group()
    sort_group.add_argument('-A', '--ascending', type=str, metavar='ATTR',
                            help='Sort one attribut, applied before start & end')
    sort_group.add_argument('-D', '--descending', type=str, metavar='ATTR',
                            help='Sort one attribut, applied before start & end')

    parser.add_argument('-s', '--start', type=int, metavar='INDEX',
                        help='Index of the first row to include, must be lower than end')
    parser.add_argument('-e', '--end', type=int, metavar='INDEX',
                        help='Index of the last row to include, must be higher than start')
    parser.add_argument('-p', '--predict', type=str, default='MEDV',
                        help='Attribute you want to predict')

    args = parser.parse_args()

    if args.start and args.end and args.start > args.end:
        parser.print_help(sys.stderr)
        exit(84)

    main(args)
