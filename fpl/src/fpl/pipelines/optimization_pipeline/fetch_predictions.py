import os
import numpy as np
import pandas as pd
import difflib


def get_latest_prediction_csv(gameweeks):
    path = "data/raw/theFPLkiwi/FPL_projections_22_23/"
    files = [
        (int(f.strip("FPL_GW").strip(".csv")), os.path.join(path, f))
        for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f))
    ]
    files = [f for f in files if f[0] <= gameweeks[0]]
    index = np.argmax([f[0] for f in files])
    return files[index][1]


def get_pred_pts_data(gameweeks):
    latest_csv = get_latest_prediction_csv(gameweeks)
    fpl_name_dict = pd.read_csv(
        "data/raw/theFPLkiwi/ID_Dictionary.csv", encoding="cp1252"
    )[["Name", "FPL name"]]
    pred_pts_data = pd.read_csv(latest_csv)
    pred_pts_data = fpl_name_dict.merge(pred_pts_data, on="Name")
    empty_cols = ["GW xMins", "xPts if play", "Probability to play", "xPts", "npG"]
    pred_pts_data = pred_pts_data.drop(empty_cols, axis=1)
    pred_pts_data = pred_pts_data.filter(regex="^(?!Unnamed)")  # drop unnamed columns
    week_cols = pred_pts_data.columns[9:]
    new_col_name = {
        col: f"{empty_cols[round(float(col)%1*10)]}_{round(float(col))}"
        for col in week_cols
    }
    pred_pts_data = pred_pts_data.rename(columns=new_col_name)
    pred_pts_data = pred_pts_data[
        ["FPL name", "Team", "Pos", "Price"] + [f"xPts_{w}" for w in gameweeks]
    ]
    return pred_pts_data


def fuzzy_match(row, df_to_match):
    combined_name = row["FPL name"] + " " + row["Team"]
    df_to_match["combined_name"] = (
        df_to_match["web_name"] + " " + df_to_match["short_name"]
    )
    all_options = (df_to_match["combined_name"]).tolist()

    matches = difflib.get_close_matches(combined_name, all_options, n=1, cutoff=0.6)
    if matches:
        matched = next(iter(matches))
        return df_to_match.loc[
            df_to_match["combined_name"] == matched, "web_name"
        ].values[0]
    else:
        return None


def resolve_fpl_names(df, fpl_names):
    df["matched"] = df.apply(lambda row: fuzzy_match(row, fpl_names), axis=1)
    df["same"] = df["FPL name"] == df["matched"]
    df = df.sort_values(["same", "Price"], ascending=False)
    df = df.drop_duplicates(["Team", "matched"])
    df.loc[:, "FPL name"] = df["matched"]
    df = df.drop(["matched", "same"], axis=1)
    return df
