import pandas as pd
import click


@click.command()
@click.argument('input_path', type=click.Path())
@click.argument('output_path', type=click.Path())
def add_features(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Create damp variables
    :param input_path:
    :param output_path:
    """
    df = pd.read_csv(input_path)

    col_categ = ['education', 'month', 'sna', 'first_time', 'region_rating', 'home_address', 'day']
    for x in col_categ:
        df1 = pd.get_dummies(df[col_categ], prefix=x)
        df = pd.concat([df, df1], axis=1)
        df.drop([x], axis=1, inplace=True)
    print('Dammi var:')
    print(df.head())
    df.to_csv(output_path)


if __name__ == '__main__':
    add_features()
