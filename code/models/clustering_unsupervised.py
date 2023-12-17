import numpy as np
import pandas as pd
import plotly.express as px

from os.path import abspath, dirname
from plotly.subplots import make_subplots
from numpy import unique
from sklearn.cluster import (
    AffinityPropagation,
    AgglomerativeClustering,
    Birch,
    DBSCAN,
    KMeans,
    MiniBatchKMeans,
    MeanShift,
    OPTICS,
    SpectralClustering,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import classification_report
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder
from sklearn.utils._testing import ignore_warnings
from time import time, time_ns

from config.settings import settings
from helpers.db_pg2 import pg2_conn, pg2_cursor

# Ignore les avertissements indiquant que le modèle n'a pas pu converger notamment
ignore_warnings(category=ConvergenceWarning)
ignore_warnings(category=RuntimeWarning)
ignore_warnings(category=UserWarning)


def create_logging_table():
    """
    Crée la table de log pour comparer les performances des modèles
    :return:
    """
    global pg2_conn, pg2_cursor

    pg2_cursor.execute(
        f"""CREATE TABLE IF NOT EXISTS {settings['PG_TABLE_MODELS_CLUSTERING_UNSUPERVISED']}
          (
            ID SERIAL NOT NULL,
            ModelType TEXT,
            InputCols TEXT,
            NumSamples INT,
            Parameters TEXT,
            Duration FLOAT,
            PlotFile TEXT,
            Results TEXT,
            CONSTRAINT {settings['PG_TABLE_MODELS_CLUSTERING_UNSUPERVISED']}_pk PRIMARY KEY(ID)
          );"""
    )
    pg2_conn.commit()


def test_model(
    query_select,
    query_where,
    num_samples=0,
    model_type="AffinityPropagation",
    **model_params,
):
    """
    Teste le modèle demandé avec les paramètres spécifiés, affiche un graphique et enregistre les informations en base de données
    :param query_select: la nature des échantillons à récupérer dans la base de données
    :param query_where: la partie conditions de la requête de récupération des échantillons dans la base de données
    :param num_samples: le nombre maximum d'échantillons à récupérer dans la base de données, ou 0 pour toutes les valeurs disponibles
    :param model_type: le type de modèle à tester
    :return: accuracy, f1-score
    """
    # Récupération des échantillons dans la base de données
    pg2_cursor.execute(
        f"SELECT {query_select} FROM {settings['PG_TABLE_VEHICLES']} WHERE {query_where};",
    )

    # Construction du dataframe d'échantillons
    features = query_select.split(",")
    model_input = pd.DataFrame(
        [[r[i] for i in range(len(features))] for r in pg2_cursor.fetchall()],
        columns=[features],
    )

    # Mélange des échantillons pour assurer une certaine représentativité
    model_input = model_input.sample(frac=1, random_state=33)[
        : num_samples if num_samples > 0 else -1
    ]

    # Remplacement des valeurs nulles par des 0
    model_input.fillna(0, inplace=True)
    model_input.reset_index(drop=True, inplace=True)

    # Encodage des colonnes qui ne sont pas numériques
    for c in model_input.columns:
        if model_input.dtypes[c] == "object":
            label_encoders[c] = LabelEncoder()
            model_input[c] = label_encoders[c].fit_transform(model_input[c])

    # Sauvegarde de l'heure actuelle (à la nanoseconde)
    time_start = time_ns()

    # Définition du modèle
    match model_type:
        case "AgglomerativeClustering":
            model = AgglomerativeClustering(**model_params)
        case "BIRCH":
            model = Birch(**model_params)
        case "DBSCAN":
            model = DBSCAN(**model_params)
        case "GaussianMixture":
            model = GaussianMixture(**model_params)
        case "KMeans":
            model = KMeans(**model_params)
        case "MiniBatchKMeans":
            model = MiniBatchKMeans(**model_params)
        case "MeanShift":
            model = MeanShift(**model_params)
        case "OPTICS":
            model = OPTICS(**model_params)
        case "SpectralClustering":
            model = SpectralClustering(**model_params)
        case _:
            model = AffinityPropagation(**model_params)

    if model_type in [
        "AffinityPropagation",
        "BIRCH",
        "KMeans",
        "MiniBatchKMeans",
    ]:
        # Entraînement du modèle
        model.fit(model_input)
        # Récupèration des résultats de prédictions (étiquettes de cluster) pour chaque échantillon
        out = (
            1 - model.predict(model_input)
            if model_type in ("GaussianMixture", "BIRCH")
            else model.predict(model_input)
        )
    elif model_type in [
        "AgglomerativeClustering",
        "DBSCAN",
        "GaussianMixture",
        "MeanShift",
        "OPTICS",
        "SpectralClustering",
    ]:
        # Entraînement du modèle et récupèration les résultats de prédictions (étiquettes de cluster) pour chaque échantillon
        out = model.fit_predict(model_input)

    # Calcul du temps écoulé pour l'entraînement du modèle
    duration = (time_ns() - time_start) / 10**9

    # Génèration du rapport de classification
    report = classification_report(
        model_input.iloc[:, model_input.columns.get_loc(model_input.columns[-1])],
        out,
        output_dict=True,
        zero_division=1,
    )
    print(
        f"Pour {model_type}:",
        "Accuracy =",
        report["accuracy"],
        "\nMacro average =",
        report["macro avg"],
        "\nWeighted average :",
        report["weighted avg"],
    )

    # Récupération des classes
    clusters = unique(out)

    plot_filename = (
        dirname(abspath(__file__)) + f"/../tmp/models/{model_type}_{time()}.png"
    )

    # Enregistrement des données de test du modèle dans la base de données
    params_str = "; ".join([k + ":" + str(v) for k, v in model_params.items()])
    pg2_cursor.execute(
        f"INSERT INTO {settings['PG_TABLE_MODELS_CLUSTERING_UNSUPERVISED']} VALUES(DEFAULT, '{model_type}', '{query_select}', "
        f"'{str(len(out))}', '{params_str}', {duration:.3f}, '{plot_filename}', 'num_clusters:{len(clusters)}')"
    )
    pg2_conn.commit()

    # Création du dataframe final (résultant des "features" et de la prédiction)
    df_plot_ready = pd.concat([model_input, pd.DataFrame(out)], axis=1)
    columns = [feature for feature in features]
    columns.append("out")
    df_plot_ready.columns = columns

    # Affichage du graphique en nuage de points
    fig = px.scatter(
        data_frame=df_plot_ready,
        x=features[0],
        y=features[1],
        color=features[2] if len(features) - 1 > 2 else out,
        symbol=out if len(features) - 1 > 2 else None,
        title=model_type + " avec " + params_str,
    )
    fig.update_xaxes(range=[0, 1000])
    fig.update_yaxes(range=[0, 7000])
    fig.write_image(plot_filename)
    fig.show()

    return report["accuracy"], report["macro avg"]["f1-score"]


#
def plot_against(query):
    """
    Affiche un graphique des données réelles de la base de données à des fins comparatives
    :param query: la requête SQL de récupération des données à afficher dans le graphique
    :return:
    """
    global pg2_cursor

    # Récupération des données et population des ensembles d'entrée (X) et de sortie (y)
    pg2_cursor.execute(query)
    X = []
    y = []
    features = query[len("SELECT ") - 1 : query.find("FROM")].split(",")
    for r in pg2_cursor.fetchall():
        X.append([r[f] for f in range(len(features) - 1)])
        y.append(r[len(features) - 1])
    X = pd.DataFrame(X).fillna(0)
    out = np.array(y)

    # Affichage du graphique en nuage de points
    fig = px.scatter(
        x=X[0],
        y=X[1],
        color=X[2] if len(features) - 1 > 2 else out,
        symbol=out if len(features) - 1 > 2 else None,
        title=",".join(features[: len(features) - 1])
        + " vs "
        + features[len(features) - 1],
    )
    fig.update_xaxes(range=[0, 1000])
    fig.update_yaxes(range=[0, 7000])
    fig.show()


def run_tests(model_list, query_select, query_where, query_plot_against):
    """
    Lance les tests correspondant à l'ensemble des paramètres spécifiés, puis affiche le graphique de comparaison
    :param model_list: la liste des modèles à tester avec leurs paramètres
    :param query_select: les colonnes à sélectionner dans la base de données
    :param query_where: la clause de condition WHERE de la requête en base de données
    :param query_plot_against: la requête SQL de récupération des données à afficher dans le graphique
    :return:
    """
    best_model = ""
    max_f1_score = 0
    for test in model_list:
        accuracy, f1_score = test_model(
            query_select,
            query_where,
            test["max_samples"],
            test["model"],
            **test["params"],
        )
        if f1_score > max_f1_score:
            max_f1_score = f1_score
            best_model = test["model"]

    plot_against(query_plot_against)

    print(
        f"Le meilleur modèle semble être {best_model} avec un f1-score de {max_f1_score}"
    )


def run_gm_co2e_ec_mro():
    # Récupération des échantillons dans la base de données
    pg2_cursor.execute(
        f"SELECT CO2e, EC, (CASE WHEN MRO<1500 THEN 0 ELSE 1 END) Masse FROM {settings['PG_TABLE_VEHICLES']} WHERE CO2e IS NOT NULL AND EC IS NOT NULL AND (MRO<1500 OR MRO>2500);",
    )

    # Construction du dataframe d'échantillons
    model_input = pd.DataFrame(
        [[r[0], r[1], r[2]] for r in pg2_cursor.fetchall()],
        columns=["CO2e", "EC", "Masse"],
    )

    # Remplacement des valeurs nulles par des 0
    model_input.fillna(0, inplace=True)
    model_input.reset_index(drop=True, inplace=True)

    # Définition du modèle
    model = GaussianMixture(n_components=2, random_state=33)

    # Entraînement du modèle
    model.fit(model_input[["CO2e", "EC"]])
    # Récupèration des résultats de prédictions (étiquettes de cluster) pour chaque échantillon
    y = 1 - model.predict(model_input[["CO2e", "EC"]])

    # Création du dataframe final incluant les colonnes d'entrée et de résultat
    df_plot_ready = pd.concat([model_input, pd.DataFrame(y)], axis=1)
    df_plot_ready.columns = ["CO2e", "EC", "Masse", "Cluster"]
    print(df_plot_ready)

    # Comptage des erreurs et ajout de la colonne "Prédiction" qui permet de savoir si la prédiction a été un succès ou un échec
    error_count = 0
    for index, row in df_plot_ready.iterrows():
        if row["Masse"] != row["Cluster"]:
            error_count += 1
            df_plot_ready.at[index, "Prédiction"] = "Echec"
        else:
            df_plot_ready.at[index, "Prédiction"] = "Succès"

    print(
        f"{error_count} prédictions erronnées ont été trouvées, soit un ratio de {error_count/len(df_plot_ready)*100:.2f}%"
    )

    # Préparation de la zone de graphique (contenant deux graphiques)
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "Prédictions de la MRO en partant de CO2e et EC, avec GaussianMixture",
            "MRO réelle en fonction de CO2e et EC",
        ],
    )

    # Ajoute le premier graphique à la première colonne
    fig.add_trace(
        px.scatter(
            data_frame=df_plot_ready,
            x="CO2e",
            y="EC",
            color="Cluster",
        )[
            "data"
        ][0],
        row=1,
        col=1,
    )

    # Ajoute le deuxième graphique à la deuxième colonne
    fig.add_trace(
        px.scatter(
            data_frame=df_plot_ready,
            x="CO2e",
            y="EC",
            color="Masse",
        )[
            "data"
        ][0],
        row=1,
        col=2,
    )

    # Affichage de la figure (avec les deux graphiques)
    fig.show()

    # Affichage du troisième graphique (échec ou succès des prédictions)
    fig = px.scatter(
        data_frame=df_plot_ready,
        x="CO2e",
        y="EC",
        color="Prédiction",
        color_discrete_sequence=px.colors.qualitative.Set1,
        title=f"Comparaison des prédictions avec les valeurs réelles (taux d'erreur de {error_count/len(df_plot_ready)*100:.2f}%)",
    )
    fig.show()


test_model_list = [
    {
        "model": "AffinityPropagation",
        "params": {"damping": 0.9},
        "max_samples": 2000,
    },
    {
        "model": "AgglomerativeClustering",
        "params": {"n_clusters": 2},
        "max_samples": 25000,
    },
    {
        "model": "BIRCH",
        "params": {"threshold": 0.01, "n_clusters": 2},
        "max_samples": 40000,
    },
    {
        "model": "DBSCAN",
        "params": {"eps": 50, "min_samples": 25},
        "max_samples": 25000,
    },
    {
        "model": "KMeans",
        "params": {
            "n_clusters": 2,
            "n_init": "auto",
            "tol": 10**-4,
            "random_state": None,
        },
        "max_samples": 50000,
    },
    {
        "model": "MiniBatchKMeans",
        "params": {
            "n_clusters": 2,
            "n_init": "auto",
            "tol": 10**-4,
            "random_state": None,
        },
        "max_samples": 50000,
    },
    {
        "model": "MeanShift",
        "params": {"bandwidth": None, "cluster_all": False, "max_iter": 200},
        "max_samples": 10000,
    },
    {
        "model": "OPTICS",
        "params": {"max_eps": 5, "min_samples": 50, "min_cluster_size": 0.02},
        "max_samples": 15000,
    },
    {
        "model": "SpectralClustering",
        "params": {"n_clusters": 2},
        "max_samples": 1000,
    },
    {
        "model": "GaussianMixture",
        "params": {"n_components": 2},
        "max_samples": 10000,
    },
]

label_encoders = {}

if __name__ == "__main__":
    create_logging_table()

    run_tests(
        test_model_list,
        "CO2e, EC, (CASE WHEN MRO<1500 THEN 0 ELSE 1 END) Masse",
        "CO2e IS NOT NULL AND EC IS NOT NULL AND (MRO<1500 OR MRO>2500)",
        f"SELECT CO2e, EC, (CASE WHEN MRO<1500 THEN 0 ELSE 1 END) Masse FROM {settings['PG_TABLE_VEHICLES']} WHERE CO2e IS NOT NULL AND EC IS NOT NULL AND (MRO<1500 OR MRO>2500);",
    )
    # f"SELECT CO2e, EC, Make FROM {settings['PG_TABLE_VEHICLES']} WHERE EC IS NOT NULL AND EC>0 AND Make IN('MERCEDES-BENZ', 'VOLKSWAGEN', 'BMW', 'AUDI', 'FORD', 'RENAULT', 'SKODA', 'SEAT', 'OPEL', 'PEUGEOT')"
    # f"SELECT CO2e, EC, (CASE WHEN MRO<1000 THEN 'A' WHEN MRO<1500 THEN 'B' WHEN MRO>2500 THEN 'E' END) Mass FROM {settings['PG_TABLE_VEHICLES']} WHERE EC IS NOT NULL AND EC>0 AND (MRO<1500 OR MRO>2500)"
    # f"SELECT CO2e, MRO, (CASE WHEN EC<1000 THEN 'A' WHEN EC<1500 THEN 'B' WHEN EC<2000 THEN 'C' WHEN EC <2500 THEN 'D' ELSE 'E' END) capacity FROM {settings['PG_TABLE_VEHICLES']}_thermal WHERE CO2e IS NOT NULL AND MRO IS NOT NULL AND CO2e>100"

    # run_gm_co2e_ec_mro()
