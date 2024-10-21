import argparse
import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine


def ingest_data(filepath, conn, table_name, chuncksize=10000):
    # Load CSV in chunks using an iterator
    df_iter = pd.read_csv(filepath, iterator=True, chunksize=chuncksize, index_col=0)
    print(f"Starting to insert records into table: {table_name}")

    # Loop through the chunks and insert them into the specified table
    for i, df in enumerate(df_iter):
        df.to_sql(
            name=table_name, con=conn, if_exists="replace" if i == 0 else "append"
        )
        print(f"Inserted the next batch of {df.shape[0]} records...")

    print(f"All records have been successfully inserted into {table_name}!")


def main(args):
    # Establish connection to the Postgres database
    conn = create_engine(
        f"postgresql://{args.POSTGRES_USER}:{args.POSTGRES_PASSWORD}@{args.POSTGRES_HOST}:{args.POSTGRES_PORT}/{args.POSTGRES_DB}"
    )

    # Ingest the cleaned and raw car datasets into the Postgres database
    path = Path(args.DATA_PATH)
    ingest_data(
        path / "autoscout24-germany-dataset-cleaned.csv",
        conn=conn,
        table_name="cars_cleaned",
    )
    ingest_data(path / "autoscout24-germany-dataset.csv", conn=conn, table_name="cars")


def add_arguments_read_env(parser, arg, help):
    parser.add_argument(
        f"--{arg}", default=os.getenv(arg), required=os.getenv(arg) is None, help=help
    )


if __name__ == "__main__":
    # Command-line argument parser for PostgreSQL connection details and CSV file path
    parser = argparse.ArgumentParser(
        description="Ingest Germany Cars CSV data into a Postgres Database"
    )

    add_arguments_read_env(parser, "POSTGRES_USER", "PostgreSQL username")
    add_arguments_read_env(parser, "POSTGRES_PASSWORD", "PostgreSQL password")
    add_arguments_read_env(
        parser, "POSTGRES_HOST", "PostgreSQL server hostname or IP address"
    )
    add_arguments_read_env(parser, "POSTGRES_PORT", "PostgreSQL server port")
    add_arguments_read_env(parser, "POSTGRES_DB", "Target PostgreSQL database name")
    add_arguments_read_env(
        parser, "DATA_PATH", "Directory path where the CSV files are located"
    )

    # Parse the arguments and run the main function
    args = parser.parse_args()
    main(args)
