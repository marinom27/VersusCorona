from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, normalize
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers

import seaborn as sns
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

sns.set(style="ticks", color_codes=True)


def get_data_github():
    filenames_suffix = [
        "aa",
        "ab",
        "ac",
        "ad",
        "ae",
        "af",
        "ag",
        "ah",
        "ai",
        "aj",
        "ak",
        "al",
        "am",
    ]
    reisende_raws = []
    for suffix in filenames_suffix:
        url = f"https://raw.githubusercontent.com/marinom27/VersusCorona/master/data/vbz_fahrgastzahlen/REISENDE_PART{suffix}.csv"
        reisende_raws.append(pd.read_csv(url, sep=";", header=None, low_memory=False))

    url = f"https://raw.githubusercontent.com/marinom27/VersusCorona/master/data/vbz_fahrgastzahlen/LINIE.csv"
    linie = pd.read_csv(url, sep=";").set_index("Linien_Id")

    url = f"https://raw.githubusercontent.com/marinom27/VersusCorona/master/data/vbz_fahrgastzahlen/TAGTYP.csv"
    tagtyp = pd.read_csv(url, sep=";").set_index("Tagtyp_Id")

    url = f"https://raw.githubusercontent.com/marinom27/VersusCorona/master/data/vbz_fahrgastzahlen/HALTESTELLEN.csv"
    haltestellen = pd.read_csv(url, sep=";").set_index("Haltestellen_Id")

    reisende_raw = pd.concat(reisende_raws)
    new_columns = reisende_raw.iloc[0]
    reisende_raw = reisende_raw.iloc[1:]
    reisende_raw.columns = new_columns

    return reisende_raw, linie, tagtyp, haltestellen


def get_data_local():
    filenames_suffix = [
        "aa",
        "ab",
        "ac",
        "ad",
        "ae",
        "af",
        "ag",
        "ah",
        "ai",
        "aj",
        "ak",
        "al",
        "am",
    ]
    reisende_raws = []
    for suffix in filenames_suffix:
        url = f"../data/vbz_fahrgastzahlen/REISENDE_PART{suffix}.csv"
        reisende_raws.append(pd.read_csv(url, sep=";", header=None, low_memory=False))

    url = f"../data/vbz_fahrgastzahlen/LINIE.csv"
    linie = pd.read_csv(url, sep=";").set_index("Linien_Id")

    url = f"../data/vbz_fahrgastzahlen/TAGTYP.csv"
    tagtyp = pd.read_csv(url, sep=";").set_index("Tagtyp_Id")

    url = f"../data/vbz_fahrgastzahlen/HALTESTELLEN.csv"
    haltestellen = pd.read_csv(url, sep=";").set_index("Haltestellen_Id")

    reisende_raw = pd.concat(reisende_raws)
    new_columns = reisende_raw.iloc[0]
    reisende_raw = reisende_raw.iloc[1:]
    reisende_raw.columns = new_columns

    return reisende_raw, linie, tagtyp, haltestellen


def clean_reisende(reisende, linie, tagtyp, haltestellen):
    reisende = reisende.rename(
        columns={
            "Tagtyp_Id": "Tagtyp_Id",
            "Linienname": "Linie",
            "Richtung": "Richtung",
            "Sequenz": "Anzahl_Haltestellen",
            "Haltestellen_Id": "Haltestelle_Id",
            "Nach_Hst_Id": "Nachste_Haltestelle_Id",
            "FZ_AB": "Uhrzeit",
            "Anzahl_Messungen": "Anzahl_Messungen",
            "Einsteiger": "Einsteiger",
            "Aussteiger": "Aussteiger",
            "Besetzung": "Besetzung",
            "Distanz": "Distanz",
            "Tage_DTV": "Tage_DTV",
            "Tage_DWV": "Tage_DWV",
            "Tage_SA": "Tage_SA",
            "Tage_SO": "Tage_SO",
        }
    )
    reisende = reisende[
        [
            "Tagtyp_Id",
            "Linie",
            "Richtung",
            "Anzahl_Haltestellen",
            "Haltestelle_Id",
            "Nachste_Haltestelle_Id",
            "Uhrzeit",
            "Anzahl_Messungen",
            "Einsteiger",
            "Aussteiger",
            "Besetzung",
            "Distanz",
            "Tage_DTV",
            "Tage_DWV",
            "Tage_SA",
            "Tage_SO",
        ]
    ]

    id_to_name = haltestellen["Haltestellenlangname"]
    id_to_nummer = haltestellen["Haltestellennummer"]
    id_to_tagbemerkung = tagtyp["Bemerkung"]
    id_to_tage = {
        "3": 62,  # Sonntag
        "4": 52,  # Samstag
        "5": 48,  # Freitag
        "6": 251,  # Montag-Freitag
        "7": 203,  # Montag-Donnerstag
    }

    reisende["Tagtyp_Id"] = reisende["Tagtyp_Id"].astype("int32").astype("category")
    reisende["Tagtyp_Bemerkung"] = (
        reisende["Tagtyp_Id"].map(id_to_tagbemerkung).astype("category")
    )
    reisende["Tagtyp_Tage"] = reisende["Tagtyp_Id"].map(id_to_tage).astype("float32")

    reisende["Linie"] = reisende["Linie"].astype("str").astype("category")
    reisende["Richtung"] = reisende["Richtung"].astype("category")
    reisende["Anzahl_Haltestellen"] = reisende["Anzahl_Haltestellen"].astype("int32")

    reisende["Haltestelle_Id"].astype("int32")
    reisende["Haltestelle"] = (
        reisende["Haltestelle_Id"].map(id_to_name).astype("category")
    )
    reisende["Haltestelle_Nummer"] = (
        reisende["Haltestelle_Id"].map(id_to_nummer).astype("category")
    )
    reisende["Nachste_Haltestelle"] = (
        reisende["Nachste_Haltestelle_Id"].map(id_to_name).astype("category")
    )
    reisende["Nachste_Haltestelle_Nummer"] = (
        reisende["Nachste_Haltestelle_Id"].map(id_to_nummer).astype("category")
    )

    reisende["Uhrzeit"] = pd.to_datetime(
        reisende["Uhrzeit"], format="%H:%M:%S", errors="coerce"
    )

    reisende["Anzahl_Messungen"] = reisende["Anzahl_Messungen"].astype("int32")
    reisende["Einsteiger"] = reisende["Einsteiger"].astype("float32")
    reisende["Aussteiger"] = reisende["Aussteiger"].astype("float32")
    reisende["Besetzung"] = reisende["Besetzung"].astype("float32")
    reisende["Distanz"] = reisende["Distanz"].astype("float32")
    reisende["Tage_DTV"] = reisende["Tage_DTV"].astype("float32").replace(0, np.NaN)
    reisende["Tage_DWV"] = reisende["Tage_DWV"].astype("float32").replace(0, np.NaN)
    reisende["Tage_SA"] = reisende["Tage_SA"].astype("float32").replace(0, np.NaN)
    reisende["Tage_SO"] = reisende["Tage_SO"].astype("float32").replace(0, np.NaN)

    reisende["Durchschnitt_Tag"] = reisende["Besetzung"] * reisende["Tage_DTV"] / 365
    reisende["Durchschnitt_Wochentag"] = (
        reisende["Besetzung"] * reisende["Tage_DWV"] / 251
    )
    reisende["Durchschnitt_Samstag"] = reisende["Besetzung"] * reisende["Tage_SA"] / 52
    reisende["Durchschnitt_Sonntag"] = reisende["Besetzung"] * reisende["Tage_SO"] / 62
    reisende["Tag"] = "Wochentag"
    reisende["Tag"] = reisende["Tag"].where(
        reisende["Durchschnitt_Samstag"].isna(), other="Samstag"
    )
    reisende["Tag"] = reisende["Tag"].where(
        reisende["Durchschnitt_Sonntag"].isna(), other="Sonntag"
    )
    reisende["Tag"] = reisende["Tag"].astype("category")
    reisende["Durchschnitt"] = reisende["Durchschnitt_Wochentag"]
    reisende["Durchschnitt"] = reisende["Durchschnitt"].where(
        reisende["Durchschnitt_Samstag"].isna(), other=reisende["Durchschnitt_Samstag"]
    )
    reisende["Durchschnitt"] = reisende["Durchschnitt"].where(
        reisende["Durchschnitt_Sonntag"].isna(), other=reisende["Durchschnitt_Sonntag"]
    )
    return reisende


def clean_na(reisende_na):
    reisende = reisende_na.dropna(
        how="any",
        subset=[
            "Tagtyp_Id",
            "Linie",
            "Richtung",
            "Haltestelle",
            "Uhrzeit",
            "Tag",
            "Durchschnitt",
        ],
    )
    return reisende


def preprocess_df(X_df, categories):
    feature_names = ["Linie", "Richtung", "Haltestelle", "Uhrzeit_Bin", "Tag"]

    X = X_df[feature_names]
    X["Ort"] = (
        X["Linie"].astype(str)
        + " "
        + X["Richtung"].astype(str)
        + " "
        + X["Haltestelle"].astype(str)
    ).astype("category")
    categories = [X[name].cat.categories for name in X.columns]
    y = reisende_sample["Besetzung"].to_numpy().reshape(-1, 1)

    enc = OneHotEncoder(categories, handle_unknown="ignore").fit(X)
    X = enc.transform(X).toarray()
    return X, y


def fit_regression_model(reisende):
    X, y = preprocess(reisende)
    model = Ridge(alpha=10, fit_intercept=False)
    model.fit(X, y)
    return model


def fit_neural_network(reisende):
    X, y = preprocess_df(reisende)
    n, d = X.shape
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = Sequential()
    model.add(
        Dense(
            units=400,
            activation="relu",
            input_dim=d,
            kernel_regularizer=regularizers.l2(0.01),
        )
    )
    model.add(Dense(units=1))

    sgd = SGD(learning_rate=0.005, momentum=0.0, nesterov=False)

    model.compile(
        loss="mean_squared_error", optimizer=sgd, metrics=["mean_squared_error"]
    )

    model.fit(X_train, y_train, epochs=10, batch_size=64)
    loss_and_metrics = model.evaluate(X_test, y_test, batch_size=128)
    print(model.summary())
    print(loss_and_metrics)
    model.reset_metrics()
    model.save("vbz_model.h5")
    return model
