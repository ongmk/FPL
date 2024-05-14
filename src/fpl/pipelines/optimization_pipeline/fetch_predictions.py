import difflib
import os

import numpy as np
import pandas as pd
from tqdm import tqdm


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


def resolve_fpl_names(pred_pts_data):
    fpl_names_mapping = pd.read_csv(
        "./src/fpl/pipelines/optimization_pipeline/fpl_names_mapping.csv"
    )
    fpl_names_mapping = fpl_names_mapping.set_index("pred_pts_fpl_name")[
        "fpl_name"
    ].to_dict()
    pred_pts_data["FPL name"] = pred_pts_data["FPL name"].map(fpl_names_mapping)
    return pred_pts_data


def fuzzy_match(row, df_to_match):
    combined_name = row["pred_pts_fpl_name"] + " " + row["Team"]
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


def refresh_fpl_names_mapping():
    from fpl.pipelines.optimization_pipeline.fpl_api import get_fpl_base_data

    elements_team = pd.read_csv("./data/raw/backtest_data/merged_gw.csv")[
        ["name", "team", "position"]
    ]
    latest_elements_team, _, _, _ = get_fpl_base_data()
    elements_team = latest_elements_team.merge(
        elements_team,
        left_on=["full_name", "name"],
        right_on=["name", "team"],
        suffixes=("", "_y"),
    )
    elements_team = (
        elements_team[["web_name", "short_name", "position"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    path = "./data/raw/theFPLkiwi/FPL_projections_22_23/"
    files = [
        os.path.join(path, f)
        for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f))
    ]
    pred_pts_data = pd.DataFrame(columns=["FPL name", "Team", "Pos"])
    fpl_name_dict = pd.read_csv(
        "./data/raw/theFPLkiwi/ID_Dictionary.csv", encoding="cp1252"
    )[["Name", "FPL name"]]
    for file in files:
        df = pd.read_csv(file)
        df = fpl_name_dict.merge(df, on="Name")
        unique_rows = df[["FPL name", "Team", "Pos", "Price"]].drop_duplicates()
        pred_pts_data = pd.concat([pred_pts_data, unique_rows], ignore_index=True)

    pred_pts_data = pred_pts_data.rename({"FPL name": "pred_pts_fpl_name"}, axis=1)
    pred_pts_data = pred_pts_data.drop_duplicates(
        subset=["pred_pts_fpl_name", "Team", "Pos"], keep="last"
    ).reset_index(drop=True)

    tqdm.pandas(desc="Resolving FPL names in predicted pts data")
    pred_pts_data["matched"] = pred_pts_data.progress_apply(
        lambda row: fuzzy_match(row, elements_team), axis=1
    )
    pred_pts_data["same"] = (
        pred_pts_data["pred_pts_fpl_name"] == pred_pts_data["matched"]
    )
    pred_pts_data = pred_pts_data.sort_values(["same", "Price"], ascending=False)
    pred_pts_data = pred_pts_data.drop_duplicates(["Team", "matched"])
    pred_pts_data["fpl_name"] = pred_pts_data["matched"]
    pred_pts_data = pred_pts_data[["pred_pts_fpl_name", "fpl_name"]].reset_index(
        drop=True
    )

    pred_pts_data.to_csv(
        "./src/fpl/pipelines/optimization_pipeline/fpl_names_mapping.csv", index=False
    )
    return None
