import pandas as pd
import plotly.express as px

import argparse

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set(style="ticks")


def main(args):
    df = pd.read_csv("./data.csv", sep=",")

    selected_attributes = args.attributes or list(df.columns)

    selected_attributes = [attribute.upper()
                           for attribute in selected_attributes]

    selected_start = args.start if args.start and args.start >= 0 and args.start < len(
        df.index) else 0
    selected_end = args.end if args.end and args.end < len(
        df.index) else len(df.index) - 1

    df = df[selected_attributes].iloc[selected_start:selected_end]

    fig = px.parallel_coordinates(df, color='MEDV')
    fig.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Say hello')
    parser.add_argument('-c', '--corner', action='store_false',
                        help="If True, don't add axes to the upper off diagonal of the grid")
    parser.add_argument('-a', '--attributes', nargs='*',
                        help='Attributes you want to see in the generated scatter matrix')
    parser.add_argument('-s', '--start', type=int,
                        help='Index of the first row to include')
    parser.add_argument('-e', '--end', type=int,
                        help='Index of the first row to include')
    parser.add_argument('-h', '--hue', type=str,
                        help='Name of the attribute to highlight')
    args = parser.parse_args()

    main(args)
