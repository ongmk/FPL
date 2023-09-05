import os
from os.path import dirname, join

import pandas as pd


def clean_players_name_string(df):
    """Clean the imported file 'name' column because it has different patterns between seasons
    """
    # replace _ with space in name column
    df["full_name"] = df["name"].str.replace("_", " ")
    # remove number in name column
    df["element_id"] = df["name"].str.extract("(\d+)", expand=False).fillna(0).astype(int)
    df["full_name"] = df["full_name"].str.replace("\d+", "")
    # trim name column
    df["full_name"] = df["full_name"].str.strip()
    return df


def filter_players_exist_latest(df, col="position"):
    """Fill in null 'position' (data that only available in 20-21 season) into previous seasons.
    """

    df[col] = df.groupby("name")[col].apply(lambda x: x.ffill().bfill())
    # df = df[df[col].notnull()]
    return df


def get_team_name(df, team_id_col):
    """Find team name from master_team_list file and match with the merged df"""

    path = os.getcwd()
    team_path = join(path, "data\\raw\\backtest_data", "master_team_list.csv")
    df_team = pd.read_csv(team_path)

    # create id column for both df_team and df
    df["id"] = df["season"].astype(str) + "_" + df[team_id_col].astype(str)
    df_team["id"] = df_team["season"].astype(str) + "_" + df_team["team"].astype(str)

    # merge two dfs
    df_team.columns = [
        col + "_right" if (col in df.columns and col != "id") else col
        for col in df_team.columns
    ]
    df = pd.merge(df, df_team, on="id", how="left")

    # rename column
    df = df.rename(columns={"team_name": f"{team_id_col}_name"})
    return df


def export_cleaned_data(df):
    """Function to export merged df into specified folder
    Args:
        path (str): Path of the folder
        filename(str): Name of the file
    """

    path = os.getcwd()
    filename = "cleaned_merged_seasons.csv"
    filepath = join(dirname(dirname("__file__")), path, filename)
    df.to_csv(filepath, encoding="utf-8")
    return df
