import numpy as np
import pandas as pd
import plotly.express as px
import random

from math import sqrt
from os.path import abspath, dirname
from plotly.subplots import make_subplots
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from helpers.db_pg2 import pg2_cursor


def run_decision_tree():
    pg2_cursor.execute(
        "SELECT DISTINCT ON (Country, Energy, PowerRelative, Date) Country, Energy, PowerRelative, Date FROM co2_energy_sources WHERE substr(date,0,5)>'2005' AND substr(date,0,5) NOT IN('2009','2014','2019') AND Energy IN('Biofuels', 'Hydro', 'Nuclear') ORDER BY Country, Energy, Date;"
    )
    df_energy_sources_train = pd.DataFrame(
        pg2_cursor.fetchall(), columns=["Country", "Energy", "PowerRelative", "Date"]
    )
    print(df_energy_sources_train)
    pg2_cursor.execute(
        "SELECT DISTINCT ON (Country, Energy, PowerRelative, Date) Country, Energy, PowerRelative, Date FROM co2_energy_sources WHERE substr(date,0,5) IN ('2009','2014','2019') AND Energy IN('Biofuels', 'Hydro', 'Nuclear') ORDER BY Country, Energy, Date;"
    )
    df_energy_sources_test = pd.DataFrame(
        pg2_cursor.fetchall(), columns=["Country", "Energy", "PowerRelative", "Date"]
    )
    print(df_energy_sources_test)

    energy_encoder = LabelEncoder()
    df_energy_sources_train["Energy"] = energy_encoder.fit_transform(
        df_energy_sources_train["Energy"]
    )
    df_energy_sources_test["Energy"] = energy_encoder.transform(
        df_energy_sources_test["Energy"]
    )

    country_encoder = LabelEncoder()
    df_energy_sources_train["Country"] = country_encoder.fit_transform(
        df_energy_sources_train["Country"]
    )
    df_energy_sources_test["Country"] = country_encoder.transform(
        df_energy_sources_test["Country"]
    )

    X_train = df_energy_sources_train[["Energy", "PowerRelative", "Date"]]
    X_test = df_energy_sources_test[["Energy", "PowerRelative", "Date"]]
    y_train = df_energy_sources_train["Country"]
    y_test = df_energy_sources_test["Country"]

    print("X_train:", X_train)
    print("y_train:", y_train)
    print("X_test:", X_test)
    print("y_test:", y_test)

    tree = DecisionTreeClassifier(criterion="entropy", random_state=33)
    tree.fit(X_train, y_train)

    print("Feature importances: {}".format(tree.feature_importances_))

    print(f"Accuracy on the training subset: {tree.score(X_train, y_train):.3f}")
    print(f"Accuracy on the test subset: {tree.score(X_test, y_test):.3f}")

    y_pred_train = tree.predict(X_train)
    y_pred_test = tree.predict(X_test)

    print("y_pred_train:", y_pred_train)
    print("y_pred_test:", y_pred_test)


def run_knn():
    pg2_cursor.execute(
        "SELECT DISTINCT ON (Country, Energy, PowerRelative, Date) Country, Energy, PowerRelative, Date FROM co2_energy_sources WHERE substr(date,0,5)>'2005' AND substr(date,0,5) NOT IN('2009','2014','2019') AND Energy IN('Biofuels', 'Hydro', 'Nuclear') ORDER BY Country, Energy, Date;"
    )
    # 'Biofuels', 'Hydro', 'Natural gas', 'Oil', 'Solar PC', 'Waste', 'Wind', 'Nuclear'
    df_energy_sources_train = pd.DataFrame(
        pg2_cursor.fetchall(), columns=["Country", "Energy", "PowerRelative", "Date"]
    )
    print(df_energy_sources_train)
    pg2_cursor.execute(
        "SELECT DISTINCT ON (Country, Energy, PowerRelative, Date) Country, Energy, PowerRelative, Date FROM co2_energy_sources WHERE substr(date,0,5) IN ('2009','2014','2019') AND Energy IN('Biofuels', 'Hydro', 'Nuclear') ORDER BY Country, Energy, Date;"
    )
    df_energy_sources_test = pd.DataFrame(
        pg2_cursor.fetchall(), columns=["Country", "Energy", "PowerRelative", "Date"]
    )
    print(df_energy_sources_test)

    energy_encoder = LabelEncoder()
    df_energy_sources_train["Energy"] = energy_encoder.fit_transform(
        df_energy_sources_train["Energy"]
    )
    df_energy_sources_test["Energy"] = energy_encoder.transform(
        df_energy_sources_test["Energy"]
    )

    country_encoder = LabelEncoder()
    df_energy_sources_train["Country"] = country_encoder.fit_transform(
        df_energy_sources_train["Country"]
    )
    df_energy_sources_test["Country"] = country_encoder.transform(
        df_energy_sources_test["Country"]
    )

    X_train = df_energy_sources_train[["Energy", "PowerRelative", "Date"]]
    X_test = df_energy_sources_test[["Energy", "PowerRelative", "Date"]]
    y_train = df_energy_sources_train["Country"]
    y_test = df_energy_sources_test["Country"]

    print("X_train:", X_train)
    print("y_train:", y_train)
    print("X_test:", X_test)
    print("y_test:", y_test)

    knn = KNeighborsClassifier(n_neighbors=3, p=1)
    # , weights="distance"
    knn.fit(X_train, y_train)

    print(f"Accuracy on the training subset: {knn.score(X_train, y_train):.3f}")
    print(f"Accuracy on the test subset: {knn.score(X_test, y_test):.3f}")

    y_pred_train = knn.predict(X_train)
    y_pred_test = knn.predict(X_test)

    print("y_pred_train:", y_pred_train)
    print("y_pred_test:", y_pred_test)


def run_random_forest():
    pg2_cursor.execute(
        "SELECT DISTINCT ON (Country, Energy, PowerRelative, Date) Country, Energy, PowerRelative, Date FROM co2_energy_sources WHERE substr(date,0,5)>'2005' AND substr(date,0,5) NOT IN('2009','2014','2019') AND Energy IN('Biofuels', 'Hydro', 'Nuclear') ORDER BY Country, Energy, Date;"
    )
    df_energy_sources_train = pd.DataFrame(
        pg2_cursor.fetchall(), columns=["Country", "Energy", "PowerRelative", "Date"]
    )
    print(df_energy_sources_train)
    pg2_cursor.execute(
        "SELECT DISTINCT ON (Country, Energy, PowerRelative, Date) Country, Energy, PowerRelative, Date FROM co2_energy_sources WHERE substr(date,0,5) IN ('2009','2014','2019') AND Energy IN('Biofuels', 'Hydro', 'Nuclear') ORDER BY Country, Energy, Date;"
    )
    df_energy_sources_test = pd.DataFrame(
        pg2_cursor.fetchall(), columns=["Country", "Energy", "PowerRelative", "Date"]
    )
    print(df_energy_sources_test)

    fig = px.scatter(
        df_energy_sources_train, x="Country", y="PowerRelative", color="Energy"
    )
    fig.show()

    energy_encoder = LabelEncoder()
    df_energy_sources_train["Energy"] = energy_encoder.fit_transform(
        df_energy_sources_train["Energy"]
    )
    df_energy_sources_test["Energy"] = energy_encoder.transform(
        df_energy_sources_test["Energy"]
    )

    country_encoder = LabelEncoder()
    df_energy_sources_train["Country"] = country_encoder.fit_transform(
        df_energy_sources_train["Country"]
    )
    df_energy_sources_test["Country"] = country_encoder.transform(
        df_energy_sources_test["Country"]
    )

    X_train = df_energy_sources_train[["Energy", "PowerRelative", "Date"]]
    X_test = df_energy_sources_test[["Energy", "PowerRelative", "Date"]]
    y_train = df_energy_sources_train["Country"]
    y_test = df_energy_sources_test["Country"]

    print("X_train:", X_train)
    print("y_train:", y_train)
    print("X_test:", X_test)
    print("y_test:", y_test)

    forest = RandomForestClassifier(
        n_estimators=100,
        criterion="entropy",
        max_features=None,
        bootstrap=False,
        random_state=33,
    )
    forest.fit(X_train, y_train)

    print(f"Accuracy on the training subset: {forest.score(X_train, y_train):.3f}")
    print(f"Accuracy on the test subset: {forest.score(X_test, y_test):.3f}")

    fig = px.bar(
        x=X_train.columns,
        y=forest.feature_importances_,
        labels={"x": "Caractéristiques", "y": "Degré d'importance"},
    )
    fig.update_layout(
        {
            "title": {
                "text": "Importance des fonctionnalités",
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
                "font": {"size": 20},
            }
        }
    )
    fig.show()

    y_pred_train = forest.predict(X_train)
    y_pred_test = forest.predict(X_test)

    print("y_pred_train:", y_pred_train)
    print("y_pred_test:", y_pred_test)


def run_svm():
    pg2_cursor.execute(
        "SELECT DISTINCT ON (Country, Energy, PowerRelative, Date) Country, Energy, PowerRelative, Date FROM co2_energy_sources WHERE substr(date,0,5)>'2005' AND substr(date,0,5) NOT IN('2009','2014','2019') AND Energy IN('Biofuels', 'Hydro', 'Nuclear') ORDER BY Country, Energy, Date;"
    )
    # 'Biofuels', 'Hydro', 'Natural gas', 'Oil', 'Solar PC', 'Waste', 'Wind', 'Nuclear'
    df_energy_sources_train = pd.DataFrame(
        pg2_cursor.fetchall(), columns=["Country", "Energy", "PowerRelative", "Date"]
    )
    print(df_energy_sources_train)
    pg2_cursor.execute(
        "SELECT DISTINCT ON (Country, Energy, PowerRelative, Date) Country, Energy, PowerRelative, Date FROM co2_energy_sources WHERE substr(date,0,5) IN ('2009','2014','2019') AND Energy IN('Biofuels', 'Hydro', 'Nuclear') ORDER BY Country, Energy, Date;"
    )
    df_energy_sources_test = pd.DataFrame(
        pg2_cursor.fetchall(), columns=["Country", "Energy", "PowerRelative", "Date"]
    )
    print(df_energy_sources_test)

    fig = px.scatter(
        df_energy_sources_train, x="Country", y="PowerRelative", color="Energy"
    )
    # fig = px.line(df_energy_sources, x="Country", y="PowerRelative", color="Energy")
    fig.show()

    """pg2_cursor.execute(
        "SELECT DISTINCT ON (Date, Energy, PowerRelative) Energy, PowerRelative, Date FROM co2_energy_sources WHERE Country='BE' ORDER BY Energy, Date;"
    )
    df_energy_sources = pd.DataFrame(
        pg2_cursor.fetchall(), columns=["Energy", "PowerRelative", "Date"]
    )
    print(df_energy_sources)

    # fig = px.scatter(df_energy_sources, x="Date", y="PowerRelative", color="Energy")
    fig = px.line(df_energy_sources, x="Date", y="PowerRelative", color="Energy")
    fig.show()"""

    df_energy_sources_train_bk = df_energy_sources_train.copy()
    df_energy_sources_test_bk = df_energy_sources_test.copy()

    energy_encoder = LabelEncoder()
    df_energy_sources_train["Energy"] = energy_encoder.fit_transform(
        df_energy_sources_train["Energy"]
    )
    df_energy_sources_test["Energy"] = energy_encoder.transform(
        df_energy_sources_test["Energy"]
    )

    country_encoder = LabelEncoder()
    df_energy_sources_train["Country"] = country_encoder.fit_transform(
        df_energy_sources_train["Country"]
    )
    df_energy_sources_test["Country"] = country_encoder.transform(
        df_energy_sources_test["Country"]
    )

    X_train = df_energy_sources_train[["Energy", "PowerRelative", "Date"]]
    X_test = df_energy_sources_test[["Energy", "PowerRelative", "Date"]]
    y_train = df_energy_sources_train["Country"]
    y_test = df_energy_sources_test["Country"]

    print("X_train:", X_train)
    print("y_train:", y_train)
    print("X_test:", X_test)
    print("y_test:", y_test)

    svm = SVC(C=5)
    svm.fit(X_train, y_train)

    print(f"Accuracy on the training subset: {svm.score(X_train, y_train):.3f}")
    print(f"Accuracy on the test subset: {svm.score(X_test, y_test):.3f}")

    y_pred_train = svm.predict(X_train)
    y_pred_test = svm.predict(X_test)

    print("y_pred_train:", y_pred_train)
    print("y_pred_test:", y_pred_test)


def run_benchmark(df, X_cols, y_col, training_ratio):
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

    """for name, model in classifiers.items():
        if name != "Abre de décision":
            y_train_ready = np.reshape(y_train.values, -1)
            y_test_ready = np.reshape(y_test.values, -1)
        else:
            y_train_ready = y_train.copy()
            y_test_ready = y_test.copy()
        model.fit(X_train, y_train_ready)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        r2_train = round(r2_score(y_train_ready, y_pred_train) * 100, 2)
        r2_test = round(r2_score(y_test_ready, y_pred_test) * 100, 2)
        mae_train = round(mean_absolute_error(y_train_ready, y_pred_train), 2)
        mae_test = round(mean_absolute_error(y_test_ready, y_pred_test), 2)
        rmse_train = round(sqrt(mean_squared_error(y_train_ready, y_pred_train)), 2)
        rmse_test = round(sqrt(mean_squared_error(y_test_ready, y_pred_test)), 2)
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
            [df_classifiers_benchmark, pd.DataFrame(model_result, index=[name])],
            ignore_index=True,
        )"""

    print(df_classifiers_benchmark)
    print(f"Le meilleur modèle est : {best_model}")

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
        # Continent, Latitude, Longitude
        # Country, Latitude, Longitude, Continent
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


def generate_guess_set():
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

    # model = GradientBoostingClassifier(n_estimators=10, random_state=33)
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

    model = run_benchmark(
        df_world, ["Latitude", "Longitude"], ["Continent encodé"], 0.7
    )
    df_predictions = get_predictions(df_world, guess_set, model, label_encoder)

    model_augmented = run_benchmark(
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

    # fig = px.scatter(df_world, x="Longitude", y="Latitude", color="Continent")
    # fig = px.line(df_world, x="Longitude", y="Latitude", color="Continent")
    # fig = px.density_contour(df_world, x="Longitude", y="Latitude", color="Continent")
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

    # print(df_world_augmented[df_world_augmented["Continent"] == "Here"])
