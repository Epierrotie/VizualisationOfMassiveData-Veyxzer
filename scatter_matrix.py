import sys
import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="ticks")


def main(args):
    df = pd.read_csv("./data.csv", sep=",")

    selected_attributes = args.attributes or list(df.columns)

    selected_attributes = [attribute.upper()
                           for attribute in selected_attributes]
    if args.hue.upper() not in selected_attributes:
        selected_attributes.append(args.hue.upper())

    selected_start = args.start if args.start and args.start >= 0 and args.start < len(
        df.index) else 0
    selected_end = args.end if args.end and args.end < len(
        df.index) else len(df.index) - 1

    if args.ascending:
        df = df.sort_values(by=args.ascending.upper(), ascending=True)
    if args.descending:
        df = df.sort_values(by=args.descending.upper(), ascending=False)

    df = df[selected_attributes].iloc[selected_start:selected_end]

    selected_attributes.remove(args.hue.upper())

    sns.PairGrid(df, hue=args.hue, vars=selected_attributes,
                 corner=args.corner).map(plt.scatter)

    title = 'Scatter matrix of the Boston Housing dataset, {} highlighted'.format(
        args.hue)
    filename = 'housing_scatter_matrix.png'
    plt.title(title)
    plt.savefig(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Program that generate a scatter plot matrix for the Boston Housing dataset')
    parser.add_argument('-a', '--attributes', nargs='*',
                        help='Attributes you want to see in the generated scatter matrix')

    sort_group = parser.add_mutually_exclusive_group()
    sort_group.add_argument('-A', '--ascending', type=str, metavar='ATTR',
                            help='Sort one attribut, applied before start & end')
    sort_group.add_argument('-D', '--descending', type=str, metavar='ATTR',
                            help='Sort one attribut, applied before start & end')

    parser.add_argument('-s', '--start', type=int, metavar='INDEX',
                        help='Index of the first row to include, must be lower than end')
    parser.add_argument('-e', '--end', type=int, metavar='INDEX',
                        help='Index of the last row to include, must be higher than start')
    parser.add_argument('-H', '--hue', type=str, default='MEDV',
                        help='Attribute you want to highlight')

    parser.add_argument('-c', '--corner', action='store_false',
                        help="Use to add scatter to the upper off diagonal of the grid")
    args = parser.parse_args()

    if args.start and args.end and args.start > args.end:
        parser.print_help(sys.stderr)
        exit(84)

    main(args)
