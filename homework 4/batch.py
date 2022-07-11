import pickle
import pandas as pd
import sys


with open('model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)


categorical = ['PUlocationID', 'DOlocationID']
year = int(sys.argv[1])
month = int(sys.argv[2])

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


df = read_data(f'fhv_tripdata_{year}-{month:02}.parquet')
df['ride_id'] = f'{year:04}/{month:02}_' + df.index.astype('str')


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = lr.predict(X_val)
y_val = df['duration'].values

print(y_pred.mean())




