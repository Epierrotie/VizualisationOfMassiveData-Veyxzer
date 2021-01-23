import argparse

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set(style="ticks")


def main(args):
    df = pd.read_csv("./data.csv", sep=",")

    selected_attributes = args.attributes or list(df.columns)
    if not args.attributes:
        selected_attributes.remove('MEDV')

    print(args)
    selected_start = args.start or 0
    selected_end = args.end if args.end and args.end < len(
        df.index) else len(df.index) - 1

    print(selected_start, selected_end)
    sns.PairGrid(df, hue='MEDV', vars=selected_attributes,
                 corner=True).map(plt.scatter)

    title = 'Scatter matrix of the Boston Housing dataset'
    filename = 'housing_scatter_matrix.pdf'
    plt.title(title)
    plt.savefig(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Say hello')
    # parser.add_argument('-n', '--name', action='store_true',
    #                     help='your name, enter it')
    parser.add_argument('-a', '--attributes', nargs='*',
                        help='Attributes you want to see in the generated scatter matrix')
    parser.add_argument('-s', '--start', type=int,
                        help='Index of the first row to include')
    parser.add_argument('-e', '--end', type=int,
                        help='Index of the first row to include')
    args = parser.parse_args()

    main(args)
