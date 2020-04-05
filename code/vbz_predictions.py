from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt

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


def longest_chain(chain, stations):
    last_station = chain[0]
    # print(last_station)
    next_stations = list(
        stations.loc[stations["Nachste_Haltestelle"] == last_station]["Haltestelle"]
    )
    # print(next_stations)
    if len(next_stations) == 0:
        return chain
    max_chain = chain
    for station in next_stations:
        if station == last_station:
            return chain
        # time.sleep(1)
        new_chain = [station] + chain
        new_longestchain = longest_chain(new_chain, stations)
        if len(new_longestchain) > len(max_chain):
            max_chain = new_longestchain
    return max_chain


def longest_chain(chain, stations, rec, maxrec, stop):
    last_station = chain[0]
    # print(last_station)
    next_stations = list(
        stations.loc[stations["Nachste_Haltestelle"] == last_station]["Haltestelle"]
    )
    # print(next_stations)
    if len(next_stations) == 0:
        return chain
    if stop == last_station:
        return chain
    if rec >= maxrec:
        return chain
    max_chain = chain
    for station in next_stations:
        # print(f"{last_station} -> {station}")
        if station == last_station:
            return chain
        # time.sleep(1)
        new_chain = [station] + chain
        new_longestchain = longest_chain(new_chain, stations, rec + 1, maxrec, stop)
        if len(new_longestchain) > len(max_chain):
            max_chain = new_longestchain
    return max_chain


def stationsbetween(line, richtung, a, b, numstations, reisende):
    stations = reisende.loc[
        (reisende["Linie"] == line) & (reisende["Richtung"] == richtung)
    ]
    stations = stations.drop_duplicates(subset=["Haltestelle", "Nachste_Haltestelle"])
    return longest_chain([b], stations, 0, numstations, a)


def plot_grid(line_stations, direction, reisende):
    reisende["Haltestelle_Linie"] = reisende["Haltestelle"].astype("str") + reisende[
        "Linie"
    ].astype("str")
    line_stations = [str(s) + str(l) for l, s in line_stations]
    data = reisende[reisende["Haltestelle_Linie"].isin(line_stations)]
    data = data[data["Richtung"] == 1]
    data["Haltestelle"] = data["Haltestelle"].cat.remove_unused_categories()
    data["Tag"] = data["Tag"].cat.remove_unused_categories()

    g = sns.FacetGrid(
        data,
        row="Haltestelle_Linie",
        col="Tag",
        hue="Tagtyp_Id",
        margin_titles=True,
        height=5,
    )
    (g.map(sns.scatterplot, "Uhrzeit_Bin", "Besetzung")).add_legend()


def plot(line, station, direction, reisende, predictions=None):
    data = reisende[
        (reisende["Linie"] == line)
        & (reisende["Haltestelle"] == station)
        & (reisende["Richtung"] == direction)
    ]
    # sns.scatterplot(x='Uhrzeit_Bin', y='Durchschnitt', data=data)
    sns.scatterplot(x="Uhrzeit_Bin", y="Besetzung", hue="Tag", data=data)
    if not predictions is None:
        print(len(predictions))
        sns.lineplot(x=range(num_bins), y=predictions)


def fit_model(reisende):
    X = reisende[["Linie", "Richtung", "Haltestelle", "Uhrzeit_Bin", "Tag"]]
    y = reisende["Besetzung"]

    enc = OneHotEncoder(handle_unknown="ignore").fit(X)

    X_hot = enc.transform(X)

    model = LinearRegression(fit_intercept=False)
    model.fit(X_hot, y)
    return enc, model


def get_vbz_context(bins_per_hour=4):
    """Here the model ist fittet and the lines are calculated.

    This returns a context object containing the model parameters and the stops for each line"""
    print("donwloading raw vbz data")
    reisende_raw, linie, tagtyp, haltestellen = get_data_local()
    print("cleaning vbz data")
    reisende_na = clean_reisende(reisende_raw, linie, tagtyp, haltestellen)
    reisende = clean_na(reisende_na)

    num_bins = 24 * bins_per_hour
    reisende["Uhrzeit_Bin"] = pd.cut(
        reisende["Uhrzeit"], num_bins, labels=range(num_bins)
    )
    print("training model")
    enc, model = fit_model(reisende)
    # print("building vbz network")
    # vbz_network = build_vbz_network(reisende_na)

    return enc, model, reisende  # , vbz_network


def get_tag(python_datetime):
    weekday = python_datetime.weekday()
    if weekday < 5:
        return "Wochentag"
    if weekday == 5:
        return "Samstag"
    if weekday == 6:
        return "Sonntag"


def get_time_bin(python_datetime, bins_per_hours=4):
    hours = python_datetime.hour
    minutes = python_datetime.minute
    return hours * 4 + (minutes % 15)


def predict_marino(a, a_time, b, b_time, numstations, line, direction, vbz_context):
    """
    dep: departure station
    dep_time:



    """
    enc, model, reisende = vbz_context
    stations = stationsbetween(line, direction, a, b, numstations, reisende)
    tag = get_tag(a_time)
    besetzungen = []
    # X_pred = [[line, direction, sation, time_bin, tag]]
    for station in stations:
        time_bin = get_time_bin(a_time)

        X_pred = [[line, direction, station, time_bin, tag]]
        X_pred_hot = enc.transform(X_pred)

        y_pred = model.predict(X_pred_hot)
        besetzungen.append(y_pred)
        print(y_pred)
        # plot(line, station, direction, reisende, y_pred)

    return max(besetzungen)
