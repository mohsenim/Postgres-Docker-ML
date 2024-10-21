# Ingesting Data into Postgres and Training a Machine Learning Model

This repository demonstrates how to use Docker to manage a Postgres database, ingest data into the database, and train a machine learning model using data stored in the database. The dataset used is the [Germany Cars Dataset](https://www.kaggle.com/datasets/ander289386/cars-germany). Our objective is to save this data into a Postgres database and train an XGBoost model to predict car prices.

The preprocessing and cleaning of the dataset are discussed in detail in [this Medium post](https://medium.com/@mohsenim/tracking-machine-learning-experiments-with-mlflow-and-dockerizing-trained-models-germany-car-price-e539303b6f97) and in [this GitHub repository](https://github.com/mohsenim/MLflow-XGBoost-Docker). In this project, we ingest both the original dataset and the preprocessed cleaned version into the database.

Since we assume the dataset is large, we load the data in chunks and iteratively save it into a database table to avoid memory constraints.

## Overview of Docker Containers

The repository uses three Docker containers:
1. **Postgres**: Manages the database.
2. **Data Ingestion**: Reads the dataset and ingests it into the Postgres database in chunks.
3. **Model Training**: Extracts data from the Postgres database and trains an XGBoost model to predict car prices.

## Docker Network Setup

To enable communication between Docker containers, we need to create a custom Docker network:

```
docker network create my-network
```

## Postgres Docker Image

We use the official Docker image for Postgres to set up a Postgres server. Here's how you can run the Postgres container:

'''
docker run -it \
  -e POSTGRES_USER="root" \
  -e POSTGRES_PASSWORD="root" \
  -e POSTGRES_DB="germany_cars_db" \
  -v $(pwd)/postgres_data:/var/lib/postgresql/data/ \
  -p 5432:5432 \
  --netwrok=my-network \
  --name=postgres-db
  postgres:13
'''

Postgres uses folder `postgres_data` to save data.

### Docker Image to Ingest Data
`ingest_data.py` reads csv files saved in folder `data` and inserts them into a database. This code can be run like:

```
python ingest_data.py --POSTGRES_USER root --POSTGRES_PASSWORD root --POSTGRES_HOST localhost --POSTGRES_PORT 5432 --POSTGRES_DB germany_cars_db --DATA_PATH ./data
```

The docker image can be built using `Dockerfile_ingest_data`, which contains the specifications of the docker image:

```
docker build . -t data_ingest:latest -f Dockerfile_ingest_data
```

and it can be run using this command:

```
docker run -it   --network=my-network \
  -v $(pwd)/data:/app/data \
  -e POSTGRES_USER="root"     \
  -e POSTGRES_PASSWORD="root"     \
  -e POSTGRES_DB="germany_cars_db"     \
  -e POSTGRES_PORT=5432          \
  -e POSTGRES_HOST="postgres-db" \
  -e DATA_PATH="./data/" \
  data-ingest:latest   
```

### Docker Image for Model Training

`train_model.py` extracts data from the database and trains an XGBoost model. It can be run with the following command:

```
python train.py --POSTGRES_USER root --POSTGRES_PASSWORD root --POSTGRES_HOST localhost --POSTGRES_PORT 5432 --POSTGRES_DB germany_cars_db
```

You can build the Docker image using the `Dockerfile_train_model`:

```
docker build . -t train-model:latest -f Dockerfile_train
```

Then, run the training container like this:

```
docker run -it   --network=my-network \
  -v $(pwd)/artifacts:/app/artifacts   \
  -e POSTGRES_USER="root" \
  -e POSTGRES_PASSWORD="root" \
  -e POSTGRES_DB="germany_cars_db" \
  -e POSTGRES_PORT="5432" \
  -e POSTGRES_HOST="postgres-db" \
  train-model:latest
```

The trained model is saved in folder `artifacts`.

## Using Docker Compose to Run all Images
As our project is a multi-container application, we can use Docker Compose to run all containers together. The `docker-compose.yaml` file defines the specifications for the containers. Using this file, we can easily run all containers with the following command:

```
docker compose up
```

<!-- Run by arguments:
docker run -it   --network=my-network \
  -v $(pwd)/data:/app/data \
  data-ingest:latest   \
  --POSTGRES_USER="root"     \
  --POSTGRES_PASSWORD="root"     \
  --POSTGRES_DB="germany_cars_db"     \
  --POSTGRES_PORT=5432          \
  --POSTGRES_HOST="postgres-db" \
  --DATA_PATH="./data/"   


Run by arguments:
docker run -it   --network=my-network \
  -v $(pwd)/artifacts:/app/artifacts   \
  train-model:latest     \
  --POSTGRES_USER="root"     \
  --POSTGRES_PASSWORD="root"     \
  --POSTGRES_DB="germany_cars_db"     \
  --POSTGRES_PORT=5432          \
  --POSTGRES_HOST="postgres-db"

 -->
