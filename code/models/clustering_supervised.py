import numpy as np
import pandas as pd
import plotly.express as px
import random

from math import sqrt
from os.path import abspath, dirname
from plotly.subplots import make_subplots
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def get_best_model(df, X_cols, y_col, training_ratio):
    classifiers = {
        "Decision tree": DecisionTreeClassifier(criterion="gini", max_depth=None),
        "Gaussian Naive Bayes": GaussianNB(),
        "Gradient boosting": GradientBoostingClassifier(
            n_estimators=10, random_state=33
        ),
        "K-neighbors": KNeighborsClassifier(),
        "Random forest": RandomForestClassifier(
            n_estimators=200,
            criterion="entropy",
            max_depth=8,
            max_features="sqrt",
            bootstrap=True,
            ccp_alpha=0,
            random_state=42,
        ),
        "Support vector": SVC(),
    }

    df_classifiers_benchmark = pd.DataFrame()

    train_size = round(len(df) * training_ratio)
    X_train = pd.DataFrame(df.loc[:train_size, X_cols])
    X_test = pd.DataFrame(df.loc[train_size:, X_cols])
    y_train = pd.DataFrame(df.loc[:train_size, y_col])
    y_test = pd.DataFrame(df.loc[train_size:, y_col])

    rmse_train_min, rmse_test_min, best_model = (100, 100, "none")

    ci = 1
    best_model = None

    for name, model in classifiers.items():
        print(
            f"Test du modèle {ci}/{len(classifiers)} ({ci/len(classifiers)*100:.2f}%)..."
        )
        ci += 1

        y_train_ready = np.reshape(y_train.values, -1)
        y_test_ready = np.reshape(y_test.values, -1)
        model.fit(X_train, y_train_ready)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        r2_train = round(r2_score(y_train_ready, y_pred_train) * 100, 2)
        r2_test = round(r2_score(y_test_ready, y_pred_test) * 100, 2)
        mae_train = round(mean_absolute_error(y_train_ready, y_pred_train), 2)
        mae_test = round(mean_absolute_error(y_test_ready, y_pred_test), 2)
        rmse_train = round(
            sqrt(mean_squared_error(y_train_ready, y_pred_train)),
            2,
        )
        rmse_test = round(
            sqrt(mean_squared_error(y_test_ready, y_pred_test)),
            2,
        )
        if rmse_train + rmse_test < rmse_train_min + rmse_test_min:
            rmse_train_min = rmse_train
            rmse_test_min = rmse_test
            best_model = name
        model_result = {
            "model": name,
            "r2_train": r2_train,
            "r2_test": r2_test,
            "mae_train": mae_train,
            "mae_test": mae_test,
            "rmse_train": rmse_train,
            "rmse_test": rmse_test,
        }

        df_classifiers_benchmark = pd.concat(
            [
                df_classifiers_benchmark,
                pd.DataFrame(model_result, index=[name]),
            ],
            ignore_index=True,
        )

    print(df_classifiers_benchmark)
    print(f"Le meilleur modèle semble être : {best_model}")

    # Calcule la matrice de confusion pour l'ensemble de test avec le meilleur modèle
    y_pred_test = classifiers[best_model].predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred_test)

    # Affiche la matrice de confusion sous forme de heatmap
    fig = px.imshow(conf_matrix, text_auto=True)
    fig.update_layout(
        title_text=f"Matrice de confusion pour {best_model}",
        title_x=0.5,
        title_font={"size": 20},
    )
    fig.show()

    # Affiche le raport de classification
    print(
        "Rapport de classification:",
        classification_report(y_test, y_pred_test, zero_division=1),
    )

    return classifiers[best_model]


def get_data(augmented_data=False):
    base_dir = dirname(abspath(__file__)) + f"/../data/"

    # Source: https://github.com/dbouquin/IS_608/blob/master/NanosatDB_munging/Countries-Continents.csv
    df_continents = pd.read_csv(base_dir + "countries_continents.csv")

    # Source : https://github.com/albertyw/avenews/blob/master/old/data/average-latitude-longitude-countries.csv
    df_positions = pd.read_csv(base_dir + "countries_position.csv")
    df_positions.drop(columns=["ISO 3166 Country Code"], inplace=True)

    df_world = df_positions.merge(right=df_continents, on="Country")
    df_world["Longitude"] = df_world["Longitude"].astype(str)
    df_world["Latitude"] = df_world["Latitude"].astype(str)

    if augmented_data is True:
        # Source : généré manuellement
        df_augmented = pd.read_csv(base_dir + "augmented_position.csv")

        df_augmented["Longitude"] = df_augmented["Longitude"].astype(str)
        df_augmented["Latitude"] = df_augmented["Latitude"].astype(str)
        df_augmented = pd.concat(
            [
                pd.DataFrame(
                    ["Fake country #" + str(i) for i in range(len(df_augmented))],
                    columns=["Country"],
                ),
                df_augmented[["Latitude", "Longitude", "Continent"]],
            ],
            axis=1,
        )

        df_world = pd.concat([df_world, df_augmented], axis=0)

    label_encoder = LabelEncoder()
    df_world["Continent encodé"] = label_encoder.fit_transform(df_world["Continent"])

    df_world = df_world.sample(frac=1).reset_index(drop=True)

    return df_world, label_encoder


def generate_guess_set(continents=False):
    if continents is True:
        # Europe 40,-10;70,40
        guess_set = [
            (random.randrange(40, 70), random.randrange(-10, 40)) for i in range(100)
        ]
        # Afrique -18,-15;34,33
        guess_set += [
            (random.randrange(-18, 34), random.randrange(-15, 33)) for i in range(100)
        ]
        # Asie 13,44;71,170
        guess_set += [
            (random.randrange(13, 71), random.randrange(44, 170)) for i in range(100)
        ]
        # Amérique du Nord 13,-123;62,-57
        guess_set += [
            (random.randrange(13, 62), random.randrange(-123, -57)) for i in range(100)
        ]
        # Amérique du Sud -56,-83;6,-33
        guess_set += [
            (random.randrange(-56, 6), random.randrange(-83, -33)) for i in range(100)
        ]
        # Océanie -52,106;-14,170
        guess_set += [
            (random.randrange(-52, -14), random.randrange(106, 170)) for i in range(100)
        ]
    else:
        guess_set = [
            (random.randrange(-90, 90), random.randrange(-180, 180)) for i in range(500)
        ]

    return pd.DataFrame(
        guess_set,
        columns=["Latitude", "Longitude"],
    )


def get_predictions(df, X, model, label_encoder):
    train_size = round(len(df) * 0.8)
    X_train = pd.DataFrame(df.loc[:train_size, ["Latitude", "Longitude"]])
    X_test = pd.DataFrame(df.loc[train_size:, ["Latitude", "Longitude"]])
    y_train = df.loc[:train_size, ["Continent encodé"]].values
    y_train = np.reshape(y_train, -1)
    y_test = df.loc[train_size:, ["Continent encodé"]].values
    y_test = np.reshape(y_test, -1)

    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    print(
        "r2_train:",
        round(r2_score(y_train, y_pred_train) * 100, 2),
        "; r2_test:",
        round(r2_score(y_test, y_pred_test) * 100, 2),
        "; mae_train:",
        round(mean_absolute_error(y_train, y_pred_train), 2),
        "; mae_test:",
        round(mean_absolute_error(y_test, y_pred_test), 2),
        "; rmse_train:",
        round(sqrt(mean_squared_error(y_train, y_pred_train)), 2),
        "; rmse_test:",
        round(sqrt(mean_squared_error(y_test, y_pred_test)), 2),
    )

    predictions = model.predict(X)

    df_predictions = pd.concat(
        [X, pd.DataFrame(predictions, columns=["Continent encodé"])], axis=1
    )
    df_predictions["Continent"] = label_encoder.inverse_transform(
        df_predictions["Continent encodé"]
    )
    df_predictions["Longitude"] = df_predictions["Longitude"].astype(str)
    df_predictions["Latitude"] = df_predictions["Latitude"].astype(str)

    return df_predictions


if __name__ == "__main__":
    projection = "equirectangular"

    df_world, label_encoder = get_data()
    df_world_augmented, label_encoder_augmented = get_data(True)

    guess_set = generate_guess_set()

    model = get_best_model(
        df_world, ["Latitude", "Longitude"], ["Continent encodé"], 0.7
    )
    df_predictions = get_predictions(df_world, guess_set, model, label_encoder)

    model_augmented = get_best_model(
        df_world_augmented, ["Latitude", "Longitude"], ["Continent encodé"], 0.7
    )
    df_predictions_augmented = get_predictions(
        df_world_augmented, guess_set, model_augmented, label_encoder_augmented
    )

    plots = [
        {"data": df_world, "title": "Données de base (un point par pays)"},
        {"data": df_world_augmented, "title": "Données augmentées"},
        {
            "data": df_predictions,
            "title": "Prédictions (base : données de base)",
        },
        {
            "data": df_predictions_augmented,
            "title": "Prédictions (base : données augmentées)",
        },
    ]

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"type": "scattergeo"}, {"type": "scattergeo"}],
            [{"type": "scattergeo"}, {"type": "scattergeo"}],
        ],
        subplot_titles=[plot["title"] for plot in plots],
    )
    for index, plot in enumerate(plots):
        scatter_geo = px.scatter_geo(
            plot["data"],
            lon="Longitude",
            lat="Latitude",
            color="Continent",
            color_discrete_sequence=px.colors.qualitative.D3,
            projection=projection,
        )["data"]
        for sgp in scatter_geo:
            fig.add_trace(
                sgp,
                row=index // 2 + 1,
                col=index % 2 + 1,
            )

    fig.update_layout(
        title_text="Prédiction du continent en fonction des coordonnées géographiques",
        title_x=0.5,
        title_font={"size": 20},
    )
    fig.show()
