# Basic Imports
import numpy as np
import os
import pandas as pd
import time
from helpers.db_pg2 import *
from math import sqrt

# Visualization
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.models import load_model, Model, Sequential
from tensorflow.keras.layers import Dense, Input, LSTM

# Evaluation
from sklearn.metrics import mean_squared_error

# Indique à Tensorflow de ne pas utiliser les optimisations processeur AVX/FMA pour éviter une incompatibilité
# avec les versions de Tensorflow ne les prenant pas en charge
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def data_prepare(df_energy_sources):
    # Récupère les années uniques
    energy_sources_years_unique = df_energy_sources["date"].str[:4].unique()

    # Crée un tableau vide qui sera rempli au fur et à mesure
    emissions_weighted_avg_yearly = []

    # Boucle parcourant les différentes années
    for year in energy_sources_years_unique:
        # Ne récupère les données que pour l'année year
        df_energy_sources_filtered = df_energy_sources[
            df_energy_sources["date"].str.startswith(year)
        ]
        # Calcule la moyenne pondérée des émissions de GES pour l'année à travers
        # les différentes sources d'énergie primaire et les différents pays
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
    # Définition des hyper paramètres
    MODEL_ACTIVATION_FUNCTION = "relu"
    MODEL_BATCH_SIZE = 1
    MODEL_EPOCHS = 200
    MODEL_HIDDEN_LAYERS = 128
    MODEL_LEARNING_RATE = 0.0001
    MODEL_LOOK_BACK = 6
    MODEL_TRAINING_RATIO = 0.7
    MODEL_TARGET_RMSE = 4.5

    # Création d'un dataset des valeurs d'émissions
    dataset = df_emissions_weighted_avg_yearly["co2e_weighted_avg"].values
    # Conversion en dataset Tensorflow
    tf_dataset = tf.data.Dataset.from_tensor_slices(dataset)

    # Initialisation des variables de monitoring (temps et nombre d'itérations)
    start_time = time.time()
    iteration = 0

    # Effectue plusieurs entraînements pour obtenir un modèle ayant une erreur moyenne faible
    test_rmse = 100
    while test_rmse > MODEL_TARGET_RMSE:
        # Mise-à-jour des variables de monitoring (temps et nombre d'itérations) et affichage
        iteration_start_time = time.time()
        iteration += 1
        print(f"Iteration #{iteration} (temps écoulé : {time.time() - start_time:.2f}s)")

        # Crée un dataset qui soit une suite de séries temporelles de taille maximale MODEL_LOOK_BACK+1
        # Chaque série est décalée (shifted) d'un cran par rapport à la précédente
        tf_windows = tf_dataset.window(MODEL_LOOK_BACK+1, shift=1)

        # Uniformise les séries afin de ne conserver que les séries entières (de taille MODEL_LOOK_BACK+1)
        dataset_windowed = tf_windows.flat_map(lambda x: x.batch(MODEL_LOOK_BACK+1, drop_remainder=True)) #.take(len(dataset))

        # Calcule la taille du tableau de données d'entraînement
        train_size = round(len(dataset)*MODEL_TRAINING_RATIO)
        # Construction des tableaux de données d'entraînement et de test
        train = []
        test = []
        for index, series in enumerate(dataset_windowed):
            if index < train_size-MODEL_LOOK_BACK:
                train.append(series.numpy())
            else:
                test.append(series.numpy())

        # Conversion en ndarray (pour correspondre au format attendu par le modèle Tensorflow)
        train = np.array(train)
        test = np.array(test)

        # Construction des tableaux de données d'entrée et de sortie
        train_X, train_y = train[:, :-1], train[:, -1]
        test_X, test_y = test[:, :-1], test[:, -1]

        # Augmentation de la dimensionnalité des tableaux d'entrées/sorties
        train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

        # Création du modèle LSTM simple à MODEL_HIDDEN_LAYERS couches
        input_tensor = Input(shape=(train_X.shape[1], train_X.shape[2]))
        lstm_layer = LSTM(MODEL_HIDDEN_LAYERS, activation=MODEL_ACTIVATION_FUNCTION)(input_tensor)
        output_tensor = Dense(1)(lstm_layer)
        model = Model(input_tensor, output_tensor)
        m_optimizer = Nadam(learning_rate=MODEL_LEARNING_RATE)

        # Configuration du modèle pour son entraînement
        model.compile(loss="mean_squared_error", optimizer=m_optimizer)

        # Entraînement du modèle sur MODEL_EPOCHS époques avec les données
        # de test comme données de validation pour évaluer le taux d'erreur
        model.fit(train_X, train_y, epochs=MODEL_EPOCHS, batch_size=MODEL_BATCH_SIZE, validation_data=(test_X, test_y), verbose=0, shuffle=False)

        # Prédiction des données d'entraînement
        train_pred = model.predict(train_X, verbose=0)

        # Prédiction des données de test
        test_pred = model.predict(test_X, verbose=0)

        # A ceci près que l'on décale les prédictions d'un cran vers la "gauche" (l'année précédente)
        # pour les faire correspondre aux données de sortie
        test_pred_length = len(test_pred)
        for index in range(test_pred_length-1):
            test_pred[index] = test_pred[index+1]

        PROJ_NB_YEARS = 15
        PROJ_LOOK_BACK = 10
        # Construction du dataset de données d'entrée pour les projections
        projected = dataset[-PROJ_LOOK_BACK:]
        # Prédiction des projections (une année après l'autre puisque la prédiction pour une année est dépendante
        # des valeurs d'émissions des PROJ_LOOK_BACK années précédentes)
        for i in range(1, PROJ_NB_YEARS+1):
            local_pred = model.predict(np.array([[projected[-MODEL_LOOK_BACK:]]]), verbose=0)
            projected = np.append(projected, [local_pred[0][0]], axis=0)
            # Supprime au fur et à mesure du tableau des prédictions les PROJ_LOOK_BACK premiers éléments
            if i <= PROJ_LOOK_BACK:
                projected = projected[1:]

        # Calcule et affiche les erreurs moyennes et la durée de l'itération
        train_rmse = sqrt(mean_squared_error(train[1:, -1], train_pred[:-1]))
        test_rmse = sqrt(mean_squared_error(test[:, -1], test_pred))
        iteration_duration = time.time()-iteration_start_time
        print(f"RMSE d'entraînement = {train_rmse:.3f}; RMSE de test = {test_rmse:.3f}; Temps d'entraînement et de prédiction : {iteration_duration:.2f}s")

    # Affiche le graphique des données d'origine et de prédiction
    years = range(df_emissions_weighted_avg_yearly['year'].values[0], df_emissions_weighted_avg_yearly['year'].values[-1] + PROJ_NB_YEARS)
    colors = ['b','g','r','k','m','c','y']
    plt.plot(years[:-PROJ_NB_YEARS+1], dataset, colors[0], label="Original")
    plt.plot(df_emissions_weighted_avg_yearly['year'].values[MODEL_LOOK_BACK-1:train_size-1], train_pred, colors[1], label="Train prediction")
    plt.plot(df_emissions_weighted_avg_yearly['year'].values[train_size:], test_pred, colors[2], label="Test prediction")
    plt.plot(years[-PROJ_NB_YEARS:], projected, colors[3], label="Projected prediction")
    plt.xlim(years[0]-1, years[-1]+PROJ_NB_YEARS+1)
    plt.show()

    # Enregistre le modèle sous le format Keras (préconisé) et H5 (pour la rétrocompatibilité)
    model.save(f"lstm_electric.keras")
    model.save(f"lstm_electric.h5")
    pg2_conn.close()

    # Retourne les projections sous la forme d'un dictionnaire {année: valeur}
    return dict(zip(years[-PROJ_NB_YEARS:], projected))


if __name__ == "__main__":
    df_energy_sources = data_retrieve()

    if df_energy_sources is not None:
        print("df_energy_sources:\n", df_energy_sources)
        df_emissions_weighted_avg_yearly = data_prepare(df_energy_sources)
        print(
            "df_emissions_weighted_avg_yearly:\n",
            df_emissions_weighted_avg_yearly,
        )

        print("\n--- ENTRAINEMENT DU MODELE ---")
        forecast = model_get_forecast(df_emissions_weighted_avg_yearly)

        print("\n--- RESULTATS DE LA PROJECTION ---")
        print(forecast)

    pg2_disconnect()
