import argparse
import os
from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine


def get_xgb_model_pipeline(categorical_cols, params):
    """
    Build and return the XGBoost model pipeline.
    """
    ordinal_encoder = preprocessing.OrdinalEncoder()
    preprocess = ColumnTransformer(
        [("Ordinal-Encoder", ordinal_encoder, categorical_cols)],
        remainder="passthrough",
    )
    xgb_model = xgb.XGBRegressor(**params)
    pipeline = Pipeline([("preprocess", preprocess), ("xgb_model", xgb_model)])
    return pipeline


def load_data(args):

    print("Fetching data from the PostgreSQL database...")
    conn = create_engine(
        f"postgresql://{args.POSTGRES_USER}:{args.POSTGRES_PASSWORD}@{args.POSTGRES_HOST}:{args.POSTGRES_PORT}/{args.POSTGRES_DB}"
    )

    sql_command = 'SELECT make, model, fuel, gear, "offerType", mileage_log, hp, age, price_log FROM cars_cleaned'
    data = pd.read_sql(sql=sql_command, con=conn)

    print(f"Data successfully loaded with shape: {data.shape}")

    categorical_cols = ["make", "model", "fuel", "gear", "offerType"]
    numerical_cols = ["mileage_log", "hp", "age", "price_log"]

    train, test = train_test_split(data, test_size=0.20, random_state=37)
    train_x = train.drop(["price_log"], axis=1)
    train_y = train[["price_log"]]

    test_x = test.drop(["price_log"], axis=1)
    test_y = test[["price_log"]]

    return (
        train_x,
        train_y,
        test_x,
        test_y,
        categorical_cols,
        numerical_cols,
    )


def train(args):
    """
    Train the XGBoost model using the pipeline.
    """
    train_x, train_y, test_x, test_y, categorical_cols, _ = load_data(args)

    params = {"max_depth": 8, "subsample": 0.7}

    print("Beginning model training...")
    pipeline = get_xgb_model_pipeline(categorical_cols=categorical_cols, params=params)
    pipeline.fit(train_x, train_y)

    # Model evaluation
    pred_y = pipeline.predict(test_x)
    eval_metric = root_mean_squared_error(test_y, pred_y)

    result = {"mse": eval_metric, "model": pipeline}
    return result


artifacts_path = Path("./artifacts")


def add_arguments_read_env(parser, arg, help):
    parser.add_argument(
        f"--{arg}", default=os.getenv(arg), required=os.getenv(arg) is None, help=help
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train an XGBoost model on German car data stored in a PostgreSQL database."
    )

    add_arguments_read_env(parser, "POSTGRES_USER", "PostgreSQL username")
    add_arguments_read_env(parser, "POSTGRES_PASSWORD", "PostgreSQL password")
    add_arguments_read_env(
        parser, "POSTGRES_HOST", "PostgreSQL server hostname or IP address"
    )
    add_arguments_read_env(parser, "POSTGRES_PORT", "PostgreSQL server port")
    add_arguments_read_env(parser, "POSTGRES_DB", "Target PostgreSQL database name")

    # Parse the arguments and run the training function
    args = parser.parse_args()
    result = train(args)
    print(f"Model training complete! Mean squared error (MSE): {result['mse']}")

    # Save the trained model
    model_name = "german_car_model.pkl"
    joblib.dump(result["model"], artifacts_path / model_name)
    print(f"Model '{model_name}' has been saved in the {artifacts_path} directory.")
