import requests
import pandas as pd


year = 2021
month = '04'
categorical = ['PUlocationID', 'DOlocationID']

df = pd.read_parquet(f'./data/fhv_tripdata_{year}-{month}.parquet')
df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
dicts = df[categorical].to_dict(orient='records')

url = 'http://localhost:9696/predict'
response = requests.post(url, json=dicts)
print(response.json())