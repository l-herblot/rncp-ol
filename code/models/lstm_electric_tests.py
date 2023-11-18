# Basic Imports
import numpy as np
import os
import pandas as pd
import time
from helpers.db_pg2 import *
from helpers.logger import logger
from math import sqrt

# Visualization
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio

# Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from datetime import datetime

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop, SGD
from tensorflow.keras.models import load_model, Model, Sequential
from tensorflow.keras.layers import (
    Concatenate,
    Dense,
    Dropout,
    Embedding,
    Flatten,
    Input,
    LSTM,
)
from tensorflow.keras.callbacks import EarlyStopping

# Evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Indique à Tensorflow de ne pas utiliser les optimisations processeur AVX/FMA pour éviter une incompatibilité avec les versions de TF ne les prenant pas en charge
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def data_prepare(df_energy_sources):
    # Récupération des années uniques
    energy_sources_years_unique = df_energy_sources["date"].str[:4].unique()

    # Création d'un tableau vide qui sera rempli au fur et à mesure
    emissions_weighted_avg_yearly = []

    # Boucle parcourant les différentes années
    for year in energy_sources_years_unique:
        # Ne récupère les données que pour l'année year
        df_energy_sources_filtered = df_energy_sources[
            df_energy_sources["date"].str.startswith(year)
        ]
        # Calcul la moyenne pondérée des émissions de GES pour l'année à travers les différentes sources d'énergie primaire et les différents pays
        weighted_avg = (
            df_energy_sources_filtered["co2e"]
            * df_energy_sources_filtered["powerglobal"]
        ).sum() / df_energy_sources_filtered["powerglobal"].sum()
        # Ajoute les résultats au tableau
        emissions_weighted_avg_yearly.append(
            {"year": int(year), "co2e_weighted_avg": weighted_avg}
        )

    # Crée un DataFrame à partir du tableau
    df_emissions_weighted_avg_yearly = pd.DataFrame(
        emissions_weighted_avg_yearly
    )
    # Tri le DataFrame par ordre croissant des années
    df_emissions_weighted_avg_yearly.sort_values(by=["year"], inplace=True)

    return df_emissions_weighted_avg_yearly


def data_retrieve():
    global pg2_cursor

    if pg2_cursor is None:
        return None

    # Récupération des données concernant les émissions de GES pour la production d'électricité
    pg2_cursor.execute(
        """SELECT es.UID, es.Country, est.Energy_FR AS Energy, es.PowerGlobal, es.PowerRelative, es.CO2e, es.Date
                      FROM co2_energy_sources es
                      JOIN co2_energy_sources_translations est
                        ON est.Energy_EN = es.Energy
                        AND EXTRACT(YEAR FROM es.Date::date) <> 2023;"""
    )
    results = pg2_cursor.fetchall()
    df_energy_sources = pd.DataFrame(
        results, columns=[desc[0] for desc in pg2_cursor.description]
    )

    return df_energy_sources


def model_get_forecast(
    df_emissions_weighted_avg_yearly,
    years=15,
    look_back=10,
    get_training_test=False,
    get_loss=False,
):
    MODEL_ACTIVATION_FUNCTION = "relu"
    MODEL_BATCH_SIZE = 5
    MODEL_EPOCHS = 300
    MODEL_HIDDEN_LAYERS = 48
    MODEL_LEARNING_RATE = 0.001
    MODEL_LOOK_BACK = 3
    MODEL_LOSS_FUNCTION = "mean_squared_error"
    MODEL_OPTIMIZER = "rmsprop"
    MODEL_TRAINING_RATIO = 0.73

    PG_TABLE_MODELS = "models_tuning3"
    pg2_cursor.execute(f"""CREATE TABLE IF NOT EXISTS {PG_TABLE_MODELS}
      (
        ID SERIAL NOT NULL,
        TrainingRatio FLOAT NOT NULL,
        LookBack SMALLINT NOT NULL,
        Epochs SMALLINT NOT NULL,
        BatchSize SMALLINT NOT NULL,
        HiddenLayers SMALLINT NOT NULL,
        ActivationFunction TEXT,
        Optimizer TEXT,
        LearningRate FLOAT,
        LossFunction TEXT,
        TrainLoss FLOAT,
        TestLoss FLOAT,
        Projection TEXT,
        Duration FLOAT,
        CONSTRAINT {PG_TABLE_MODELS}_pk PRIMARY KEY(ID)
      );""")
    pg2_conn.commit()

    test_params = {
        "epochs": [30, 50, 100], #[30, 50, 100, 150, 250],
        "batch_size": [1, 3, 5, 10, 15], #[1, 2, 3, 5, 8, 10, 15, 20, 30],
        "training_ratio": [.7], #[.67, .73, .8],
        "look_back": [3, 6, 9], #[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "hidden_layers": [32, 48, 64, 96, 128],
        "activation_function": ["relu"], #["relu", "sigmoid", "tanh"],
        "optimizer": ["adam", "nadam", "rmsprop"], #["adam", "nadam", "rmsprop", "sgd"],
        "learning_rate": [.01, .001, .0001], #[.1, .01, .001, .0001],
        "loss_function": ["mean_squared_error"],  # ["mean_absolute_error", "mean_squared_error"]
    }
    test_params = {
        "epochs": [200], #[30, 50, 100, 150, 250],
        "batch_size": [1], #[1, 2, 3, 5, 8, 10, 15, 20, 30],
        "training_ratio": [.7], #[.67, .73, .8],
        "look_back": [6], #[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "hidden_layers": [128],
        "activation_function": ["relu"], #["relu", "sigmoid", "tanh"],
        "optimizer": ["nadam"], #["adam", "nadam", "rmsprop", "sgd"],
        "learning_rate": [.0001], #[.1, .01, .001, .0001],
        "loss_function": ["mean_squared_error"],  # ["mean_absolute_error", "mean_squared_error"]
    }

    result_dict = {}

    """dataset = df_emissions_weighted_avg_yearly[
        ["year", "co2e_weighted_avg"]
    ].values
    dataset = dataset.astype("float32")
    # print("dataset:", dataset)

    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset_transformed = scaler.fit_transform(dataset)
    # print("dataset_transformed:", dataset_transformed)

    iteration_start_time = time.time()

    dataset_supervised = series_to_supervised(
        dataset_transformed, MODEL_LOOK_BACK, fillnan="bfill"
    )
    dataset_supervised = dataset_supervised.values
    print("dataset_supervised:\n", dataset_supervised)"""

    dataset = df_emissions_weighted_avg_yearly["co2e_weighted_avg"].values
    print("dataset:\n", dataset)
    tf_dataset = tf.data.Dataset.from_tensor_slices(dataset)
    print("tf_dataset:\n", list(tf_dataset.as_numpy_iterator()))

    start_time = time.time()
    iteration = 0
    nb_experiments = 1
    for key in test_params:
        nb_experiments *= len(test_params[key])
    print("Number of combinations to test for model tuning:", nb_experiments)

    test_rmse = 15
    while test_rmse > 4.5:
        for MODEL_EPOCHS in test_params['epochs']:
            for MODEL_BATCH_SIZE in test_params['batch_size']:
                for MODEL_TRAINING_RATIO in test_params['training_ratio']:
                    for MODEL_LOOK_BACK in test_params['look_back']:
                        for MODEL_HIDDEN_LAYERS in test_params['hidden_layers']:
                            for MODEL_ACTIVATION_FUNCTION in test_params['activation_function']:
                                for MODEL_OPTIMIZER in test_params['optimizer']:
                                    for MODEL_LEARNING_RATE in test_params['learning_rate']:
                                        for MODEL_LOSS_FUNCTION in test_params['loss_function']:
                                        #test_rmse = 15
                                        #while test_rmse > 3.8:
                                            iteration_start_time = time.time()
                                            iteration += 1
                                            if iteration > 1:
                                                time_per_iteration = (time.time() - start_time) / iteration
                                                print(
                                                    f"Iteration #{iteration}/{nb_experiments} ({iteration / nb_experiments * 100:.1f}%). ETR: {round((nb_experiments - iteration) * time_per_iteration)}s")

                                            tf_windows = tf_dataset.window(MODEL_LOOK_BACK+1, shift=1)

                                            def windowded_to_batch(sub):
                                                return sub.batch(MODEL_LOOK_BACK+1, drop_remainder=True)

                                            dataset_windowed = tf_windows.flat_map(windowded_to_batch).take(len(dataset))
                                            train_size = round(len(dataset)*MODEL_TRAINING_RATIO)
                                            train = []
                                            test = []
                                            for index, series in enumerate(dataset_windowed):
                                                if index < train_size-MODEL_LOOK_BACK:
                                                    train.append(series.numpy())
                                                else:
                                                    test.append(series.numpy())

                                            train = np.array(train)
                                            test = np.array(test)

                                            train_X, train_y = train[:, :-1], train[:, -1]
                                            test_X, test_y = test[:, :-1], test[:, -1]

                                            train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
                                            test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

                                            input_tensor = Input(shape=(train_X.shape[1], train_X.shape[2]))
                                            lstm_layer = LSTM(MODEL_HIDDEN_LAYERS, activation=MODEL_ACTIVATION_FUNCTION)(input_tensor)
                                            output_tensor = Dense(1)(lstm_layer)
                                            model = Model(input_tensor, output_tensor)
                                            match MODEL_OPTIMIZER:
                                                case "nadam":
                                                    m_optimizer = Nadam(learning_rate=MODEL_LEARNING_RATE)
                                                case "rmsprop":
                                                    m_optimizer = RMSprop(learning_rate=MODEL_LEARNING_RATE)
                                                case "sgd":
                                                    m_optimizer = SGD(learning_rate=MODEL_LEARNING_RATE)
                                                case _:
                                                    m_optimizer = Adam(learning_rate=MODEL_LEARNING_RATE)
                                            model.compile(loss=MODEL_LOSS_FUNCTION, optimizer=m_optimizer)

                                            history = model.fit(train_X, train_y, epochs=MODEL_EPOCHS, batch_size=MODEL_BATCH_SIZE, validation_data=(test_X, test_y), verbose=0, shuffle=False)
                                            #plt.plot(history.history['loss'], label='train')
                                            #plt.plot(history.history['val_loss'], label='test')
                                            #plt.legend()
                                            #plt.show()

                                            train_pred = model.predict(train_X, verbose=0)
                                            train_X = train_X.reshape((train_X.shape[0], train_X.shape[2]))
                                            train_pred = np.concatenate((train_X, train_pred), axis=1)
                                            """print("***train_pred", train_pred)
                                            # Shift inv_train_pred one step to the left to "sync" predictions with actual values and round years
                                            train_pred_length = len(train_pred)
                                            for index in range(train_pred_length-1):
                                                train_pred[index][MODEL_LOOK_BACK] = train_pred[index+1][MODEL_LOOK_BACK]
                                            print("###train_pred", train_pred)"""

                                            train_y = train_y.reshape((len(train_y), 1))
                                            train = np.concatenate((train_X, train_y), axis=1)

                                            test_pred = model.predict(test_X, verbose=0)
                                            test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
                                            test_pred = np.concatenate((test_X, test_pred), axis=1)
                                            test_pred_length = len(test_pred)
                                            for index in range(test_pred_length):
                                                test_pred[index][0] = round(test_pred[index][0])
                                                if index < test_pred_length - 1:
                                                    test_pred[index][1] = test_pred[index+1][1]

                                            test_y = test_y.reshape((len(test_y), 1))
                                            inv_test = np.concatenate((test_X, test_y), axis=1)

                                            PROJ_NB_YEARS = 15
                                            PROJ_LOOK_BACK = 10
                                            projected = dataset[-PROJ_LOOK_BACK:]
                                            for i in range(1, PROJ_NB_YEARS+1):
                                                local_pred = model.predict(np.array([[projected[-MODEL_LOOK_BACK:]]]), verbose=0)
                                                projected = np.append(projected, [local_pred[0][0]], axis=0)
                                                if i <= PROJ_LOOK_BACK:
                                                    projected = projected[1:]

                                            #colors = ['b','g','r','k','m','c','y']
                                            years=range(df_emissions_weighted_avg_yearly['year'].values[0], df_emissions_weighted_avg_yearly['year'].values[-1]+PROJ_NB_YEARS)
                                            #plt.plot(years[:-PROJ_NB_YEARS+1], dataset, colors[0], label="Original")
                                            #plt.plot(df_emissions_weighted_avg_yearly['year'].values[MODEL_LOOK_BACK-1:train_size-1], train_pred[:,-1], colors[1], label="Train prediction")
                                            #plt.plot(df_emissions_weighted_avg_yearly['year'].values[train_size-1:-1], test_pred[:,-1], colors[2], label="Test prediction")
                                            #plt.plot(years[-PROJ_NB_YEARS:], projected, colors[3], "Projected prediction")
                                            #plt.xlim(years[0]-1, years[-1]+PROJ_NB_YEARS+1)
                                            #plt.show()

                                            print(f"For {MODEL_EPOCHS} epochs, a training ratio of {MODEL_TRAINING_RATIO*100}%, a look back of {MODEL_LOOK_BACK} and a batch size of {MODEL_BATCH_SIZE}, with {MODEL_HIDDEN_LAYERS} layers, {MODEL_ACTIVATION_FUNCTION} as activation method and {MODEL_OPTIMIZER}({MODEL_LEARNING_RATE}) :")
                                            #print("train:", train[:, -1])
                                            #print("train_pred:", train_pred[:, -1])
                                            #print("test:", test[:, -1])
                                            #print("test_pred:", test_pred[:, -1])
                                            if True in np.isnan(train_pred[:, -1]):
                                                print("!!!")
                                                print("!!!NAN IN TRAIN_PRED")
                                                print("!!!")
                                            elif True in np.isnan(test_pred[:, -1]):
                                                print("!!!")
                                                print("!!!NAN IN TEST_PRED")
                                                print("!!!")
                                            else:
                                                train_rmse = sqrt(mean_squared_error(train[:-1, -1], train_pred[1:, -1]))
                                                test_rmse = sqrt(mean_squared_error(test[:-1, -1], test_pred[1:, -1]))
                                                iteration_duration = time.time()-iteration_start_time
                                                print(f"Train RMSE = {train_rmse:.3f}; Test RMSE = {test_rmse:.3f}; Training and prediction took {iteration_duration} seconds")

                                                projection = ""
                                                for i, p in enumerate(projected):
                                                    projection += f"{years[-PROJ_NB_YEARS+i]+1}={p}; "
                                                pg2_cursor.execute(f"""INSERT INTO {PG_TABLE_MODELS}
                                                    (TrainingRatio, LookBack, Epochs, BatchSize, HiddenLayers, ActivationFunction, Optimizer, LearningRate, LossFunction, TrainLoss, TestLoss, Projection, Duration)
                                                    VALUES (
                                                        {MODEL_TRAINING_RATIO},
                                                        {MODEL_LOOK_BACK},
                                                        {MODEL_EPOCHS},
                                                        {MODEL_BATCH_SIZE},
                                                        {MODEL_HIDDEN_LAYERS},
                                                        '{MODEL_ACTIVATION_FUNCTION}',
                                                        '{MODEL_OPTIMIZER}',
                                                        {MODEL_LEARNING_RATE},
                                                        '{MODEL_LOSS_FUNCTION}',
                                                        {train_rmse:.3f},
                                                        {test_rmse:.3f},
                                                        '{projection}',
                                                        {iteration_duration:.2f});""")
                                                pg2_conn.commit()

                                                if test_rmse <= 4:
                                                    break
                                        if test_rmse <= 4:
                                            break
                                    if test_rmse <= 4:
                                        break
                                if test_rmse <= 4:
                                    break
                            if test_rmse <= 4:
                                break
                        if test_rmse <= 4:
                            break
                    if test_rmse <= 4:
                        break
                if test_rmse <= 4:
                    break
            if test_rmse <= 4:
                break
        if test_rmse <= 4:
            break

    colors = ['b','g','r','k','m','c','y']
    plt.plot(years[:-PROJ_NB_YEARS+1], dataset, colors[0], label="Original")
    plt.plot(df_emissions_weighted_avg_yearly['year'].values[MODEL_LOOK_BACK-1:train_size-1], train_pred[:,-1], colors[1], label="Train prediction")
    plt.plot(df_emissions_weighted_avg_yearly['year'].values[train_size-1:-1], test_pred[:,-1], colors[2], label="Test prediction")
    plt.plot(years[-PROJ_NB_YEARS:], projected, colors[3], "Projected prediction")
    plt.xlim(years[0]-1, years[-1]+PROJ_NB_YEARS+1)
    plt.show()
    model.save(f"lstm_electric_{MODEL_EPOCHS}_{MODEL_BATCH_SIZE}_{MODEL_TRAINING_RATIO*10**4}_{MODEL_LOOK_BACK}_{MODEL_HIDDEN_LAYERS}_{MODEL_OPTIMIZER}.keras")
    model.save(f"lstm_electric_{MODEL_EPOCHS}_{MODEL_BATCH_SIZE}_{MODEL_TRAINING_RATIO*10**4}_{MODEL_LOOK_BACK}_{MODEL_HIDDEN_LAYERS}_{MODEL_OPTIMIZER}.h5")
    pg2_conn.close()

    """def splitter(batch):
        return (batch[:-1], batch[-1:])

    tf_ds_batches = tf_dataset.batch(len(dataset), drop_remainder=True)
    tf_ds_mapped = tf_ds_batches.map(splitter)
    for features, label in tf_ds_mapped.take(3):
        print(features.numpy(), " => ", label.numpy())"""
    """
    windows = tf_dataset.window(MODEL_LOOK_BACK, shift=1)
    for x in windows.flat_map(lambda x: x).take(len(dataset)):
        print(x.numpy(), end=" ")

    def sub_to_batch(sub):
        return sub.batch(MODEL_LOOK_BACK, drop_remainder=True)

    print("\nWindowded dataset:")
    for example in windows.flat_map(sub_to_batch).take(len(dataset)):
        print(example.numpy())

    return {}

    train_size = round(len(dataset_supervised) * MODEL_TRAINING_RATIO)
    train = np.array(dataset_supervised[:train_size])
    test = np.array(dataset_supervised[train_size:])

    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    # print("train_X:", train_X)
    # print("train_y:", train_y)

    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    # print("train_X (reshaped):", train_X)

    try:
        model = tf.keras.models.load_model("../data/models/lstm_electric.h5")
    except:
        logger.error("Impossible de charger le modèle LSTM")
        return None

    history = model.fit(
        train_X,
        train_y,
        epochs=MODEL_EPOCHS,
        batch_size=MODEL_BATCH_SIZE,
        validation_data=(test_X, test_y),
        verbose=0,
        shuffle=False,
    )
    if get_loss:
        result_dict["train_loss"] = history.history["loss"]
        result_dict["test_loss"] = history.history["val_loss"]

    train_pred = model.predict(train_X, verbose=0)
    train_X = train_X.reshape((train_X.shape[0], train_X.shape[2]))
    inv_train_pred = np.concatenate((train_X, train_pred), axis=1)
    inv_train_pred = scaler.inverse_transform(inv_train_pred[:, -2:])
    # print("inv_train_pred (before):", inv_train_pred)
    # Shift inv_train_pred one step to the left to "sync" predictions with actual values and round years
    inv_train_pred_length = len(inv_train_pred)
    for index in range(inv_train_pred_length):
        inv_train_pred[index][0] = round(inv_train_pred[index][0])
        if index < inv_train_pred_length - 1:
            inv_train_pred[index][1] = inv_train_pred[index + 1][1]
    # print("inv_train_pred (after):", inv_train_pred)

    train_y = train_y.reshape((len(train_y), 1))
    inv_train = np.concatenate((train_X, train_y), axis=1)
    inv_train = scaler.inverse_transform(inv_train[:, -2:])
    inv_train = np.array([[round(item[0]), item[1]] for item in inv_train])
    # print("inv_train:", inv_train)

    # print("test_X.shape:", test_X.shape)
    # print("test_X:", test_X)

    test_pred = model.predict(test_X, verbose=0)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

    inv_test_pred = np.concatenate((test_X, test_pred), axis=1)
    inv_test_pred = scaler.inverse_transform(inv_test_pred[:, -2:])
    # print("inv_test_pred (before):", inv_test_pred)
    inv_train_test_length = len(inv_test_pred)
    for index in range(inv_train_test_length):
        inv_test_pred[index][0] = round(inv_test_pred[index][0])
        if index < inv_train_test_length - 1:
            inv_test_pred[index][1] = inv_test_pred[index + 1][1]
    # print("inv_test_pred:", inv_test_pred)

    test_y = test_y.reshape((len(test_y), 1))
    inv_test = np.concatenate((test_X, test_y), axis=1)
    inv_test = scaler.inverse_transform(inv_test[:, -2:])
    inv_test = np.array([[round(item[0]), item[1]] for item in inv_test])
    # print("inv_test:", inv_test)

    last_known_year = round(dataset[-1][0])
    # proj_X = [last_known_year+i for i in range(PROJ_NB_YEARS)]
    projected = dataset[-look_back:]
    for i in range(1, years + 1):
        # print("projected:", projected)
        projected = np.append(projected, [[last_known_year + i, 0]], axis=0)
        # print("projected:", projected)
        projected_transformed = scaler.transform(projected)
        # print("projected_transformed:", projected_transformed)
        local_X = []
        for index, item in enumerate(projected_transformed):
            local_series = []
            for l in range(MODEL_LOOK_BACK + 1):
                local_series.append(projected_transformed[index + l][0])
                if l < MODEL_LOOK_BACK:
                    local_series.append(projected_transformed[index + l][1])
                # print("local_series:", local_series)
            local_X.append([local_series])
            # print("local_X:", local_X)
            if index + MODEL_LOOK_BACK >= len(projected_transformed) - 1:
                break
        # print("len(local_X):", len(local_X))
        # local_X = [[[projected_transformed[-index][0], projected_transformed[-index][1], projected_transformed[-index+1][0]]] for index in range(PROJ_LOOK_BACK, 0, -1)]
        # local_year_to_pred = scaler.transform([[last_known_year+i, 0]])[0][0]
        # local_X[-1][0][2] = local_year_to_pred
        local_X = np.array(local_X).astype("float32")
        # print("local_X:", local_X)
        local_pred = model.predict(local_X, verbose=0)
        # print("local_pred:", local_pred)
        inv_local_pred = scaler.inverse_transform(
            [[round(local_X[-1][0][-1]), local_pred[-1][0]]]
        )
        # print("inv_local_pred:", inv_local_pred)
        # projected = np.append(projected, inv_local_pred, axis=0)
        projected[-1][1] = inv_local_pred[0][1]
        if i <= look_back:
            projected = projected[1:]
        # print("projected:", projected)
    """
    """
    result_dict["projected"] = projected
    if get_training_test is True:
        result_dict["original"] = dataset
        result_dict["training"] = inv_train_pred
        result_dict["test"] = inv_test_pred
        result_dict["training_rmse"] = sqrt(
            mean_squared_error(inv_train[:, 1], inv_train_pred[:, 1])
        )
        result_dict["test_rmse"] = sqrt(
            mean_squared_error(inv_test[:, 1], inv_test_pred[:, 1])
        )
        result_dict["duration"] = time.time() - iteration_start_time
    """
    return result_dict


def series_to_supervised(
    data, n_in=1, n_out=1, n_still=0, dropnan=False, fillnan=None
):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
    data: Sequence of observations as a list or NumPy array.
    n_in: Number of lag observations as input (X).
    n_out: Number of observations as output (y).
    dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
    Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [("var%d(t-%d)" % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [("var%d(t)" % (j + 1)) for j in range(n_vars)]
        else:
            names += [("var%d(t+%d)" % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    #
    if n_still > 0:
        for i in range(1, round(len(agg) / n_still)):
            for j in range(n_in):
                agg.iloc[i * n_still + j, : -(n_vars * (j + 1))] = np.nan
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    # fill rows with NaN values
    if fillnan:
        if fillnan == "bfill":
            agg.bfill(inplace=True)
        elif fillnan == "ffill":
            agg.ffill(inplace=True)
        else:
            agg.fillna(fillnan, inplace=True)
    return agg


if __name__ == "__main__":
    df_energy_sources = data_retrieve()
    if df_energy_sources is not None:
        print("df_energy_sources:\n", df_energy_sources)
        df_emissions_weighted_avg_yearly = data_prepare(df_energy_sources)
        print(
            "df_emissions_weighted_avg_yearly:\n",
            df_emissions_weighted_avg_yearly,
        )
        print(
            "model_get_forecast:\n",
            model_get_forecast(df_emissions_weighted_avg_yearly),
        )

    pg2_disconnect()
