import pandas as pd
from sklearn.preprocessing import StandardScaler
import click

print('111')
@click.command()
@click.argument('input_path', type = click.Path(exists=True))
@click.argument('output_path', type = click.Path())
def normilize(input_path: str, output_path: str):
    print('1')
    '''
    Function for normalisation numerical data.
    :param input_path:
    :param output_path:
    '''
    df = pd.read_csv(input_path)
    print('2')
    print(df.columns)
    num_cols = ['decline_app_cnt', 'bki_request_cnt', 'income','age']
    scaler = StandardScaler()
    scaler.fit_transform(df[num_cols])
    df[num_cols] = scaler.transform(df[num_cols])
    print(f'Normalisation is done')
    print(df[num_cols].head())
    df.to_csv(output_path)


if __name__=='__main__':
    normilize()