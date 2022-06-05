import pickle

import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner

import datetime

import mlflow


@task
def read_data(path):
    df = pd.read_parquet(path)
    return df


@task
def get_path(date):
    if date is None:
        date = datetime.datetime.now()
    month = int(date.split('-')[1][1])
    m_train = month - 2
    m_val = month - 1
    if m_train <= 9:
        m_train = f'0{m_train}'
    else:
        m_train = str(m_train)
    if m_val <= 9:
        m_val = f'0{m_val}'
    else:
        m_val = str(m_val)
    train_path = f'./data/fhv_tripdata_2021-{m_train}.parquet'
    val_path = f'./data/fhv_tripdata_2021-{m_val}.parquet'
    return train_path, val_path


@task
def prepare_features(df, categorical, train=True):
    logger = get_run_logger()
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df


@task
def train_model(df, categorical):
    with mlflow.start_run():
        logger = get_run_logger()
        train_dicts = df[categorical].to_dict(orient='records')
        dv = DictVectorizer()
        X_train = dv.fit_transform(train_dicts)
        y_train = df.duration.values
        logger.info(f"The shape of X_train is {X_train.shape}")
        logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_train)
        mse = mean_squared_error(y_train, y_pred, squared=False)
        logger.info(f"The MSE of training is: {mse}")
        mlflow.log_metric("mse_train", mse)
    return lr, dv


@task
def run_model(df, categorical, dv, lr):
    with mlflow.start_run():
        logger = get_run_logger()
        val_dicts = df[categorical].to_dict(orient='records')
        X_val = dv.transform(val_dicts)
        y_pred = lr.predict(X_val)
        y_val = df.duration.values

        mse = mean_squared_error(y_val, y_pred, squared=False)
        logger.info(f"The MSE of validation is: {mse}")
        mlflow.log_metric("mse_valid", mse)
    return


@flow(task_runner=SequentialTaskRunner())
def main(date="2021-08-15"):
    train_path, val_path = get_path(date).result()
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("ny-taxi")
    mlflow.sklearn.autolog()
    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical).result()

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical).result()

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    run_model(df_val_processed, categorical, dv, lr)

    with open(f"models/model-{date}.pkl", "wb") as f_out:
        pickle.dump(lr, f_out)
    with open(f"models/dv-{date}.pkl", "wb") as f_out:
        pickle.dump(dv, f_out)


from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner

DeploymentSpec(
    flow=main,
    name="model_training",
    schedule=CronSchedule(cron="0 9 15 * *"),
    flow_runner=SubprocessFlowRunner(),
    tags=["ml"]
)
