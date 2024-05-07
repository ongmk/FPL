import logging

import pandas as pd

from src.fpl.pipelines.scraping_pipeline.BaseDriver import BaseDriver

logger = logging.getLogger(__name__)


class FBRefDriver(BaseDriver):
    """Custom web driver for FBRef.com"""

    def __init__(self, **kwargs):
        BaseDriver.__init__(self, **kwargs)
        self.base_url = "https://fbref.com"

    def get_team_season_links(self, season):
        relative_url = f"/en/comps/9/{season}/{season}-Premier-League-Stats"
        url = self.absolute_url(relative_url)
        logger.info(f"Crawling Season Result:\t{season}\t{url}")
        self.get(url)
        tree = self.get_tree_by_id(f"results{season}91_overall")
        links = tree.xpath("*/tbody/tr/td[1]/a")
        return [(l.text, self.absolute_url(l.get("href"))) for l in links]

    def get_player_season_links(
        self, season: str, current_season: str
    ) -> list[str, str, str]:
        if season == current_season:
            relative_url = f"/en/comps/9/stats/Premier-League-Stats"
        else:
            relative_url = f"/en/comps/9/{season}/stats/{season}-Premier-League-Stats"
        url = self.absolute_url(relative_url)
        logger.info(f"Crawling Season Result:\t{season}\t{url}")
        self.get(url)
        tree = self.get_tree_by_id(f"stats_standard")
        rows = tree.xpath("*/tbody/tr[not(@class)]")
        player_season_links = []
        for r in rows:
            player = r.xpath("./td[1]/a")[0].text
            pos = r.xpath("./td[3]")[0].text
            relative_url = r.xpath("./td[last()]/a")[0].get("href")

            if pos == "GK":
                relative_url = relative_url.replace("summary", "keeper")
            player_season_links.append((player, pos, self.absolute_url(relative_url)))
        return player_season_links

    def get_team_match_log(self, season, team, link):
        logger.info(f"Crawling Match Log:\t{season} {team}\t{link}")
        self.get(link)
        match_log_df = self.get_table_df_by_id("matchlogs_for")

        match_log_df = match_log_df.drop(
            ["Attendance", "Captain", "Formation", "Referee", "Match Report", "Notes"],
            axis=1,
        )
        match_log_df[["GF", "GA"]] = (
            match_log_df[["GF", "GA"]].stack().str.replace(" \(\d+\)", "").unstack()
        )
        numeric_columns = [
            c for c in match_log_df.columns if c in ["GF", "GA", "xG", "xGA", "Poss"]
        ]
        match_log_df[numeric_columns] = match_log_df[numeric_columns].apply(
            pd.to_numeric
        )
        match_log_df["Season"] = season
        match_log_df["Team"] = team
        match_log_df["Link"] = link

        return match_log_df

    def get_player_match_log(self, season, player, pos, link):
        logger.info(f"Crawling Match Log:\t{season} {player}\t{link}")
        self.get(link)
        match_log_df = self.get_table_df_by_id("matchlogs_all")

        useless_cols = [
            c
            for c in match_log_df.columns
            if c
            in [
                "CrdY",
                "CrdR",
                "Press",
                "Tkl",
                "Int",
                "Blocks",
                "Cmp",
                "Att",
                "Cmp%",
                "Prog",
                "Carries",
                "Prog",
                "Succ",
                "Att",
                "Match Report",
                "PKA",
                "PKsv",
                "PKm",
                "Att",
                "Thr",
                "Launch%",
                "AvgLen",
                "Launch%",
                "AvgLen",
                "Opp",
                "Stp",
                "Stp%",
                "#OPA",
                "AvgDist",
                "Fls",
                "Fld",
                "Off",
                "Crs",
                "TklW",
                "OG",
                "PKwon",
                "PKcon",
                "PrgP",
                "PrgC",
                "Att (GK)",
            ]
        ]
        match_log_df = match_log_df.drop(useless_cols, axis=1)
        numeric_columns = [
            c
            for c in match_log_df.columns
            if c
            in [
                "Min",
                "Gls",
                "Ast",
                "Pk",
                "PKatt",
                "Sh",
                "SoT",
                "Touches",
                "xG",
                "npxG",
                "xAG",
                "SCA",
                "GCA",
                "SoTA",
                "GA",
                "Saves",
                "Save%",
                "CS",
                "PSxG",
            ]
        ]
        match_log_df[numeric_columns] = match_log_df[numeric_columns].apply(
            pd.to_numeric
        )
        match_log_df["Start"] = match_log_df["Start"].str.contains("Y").mul(1)
        match_log_df["Pos"] = match_log_df["Pos"].fillna(
            match_log_df["Pos"].mode().values[0]
        )
        match_log_df = match_log_df.rename(columns={"Save%": "SavePCT"})

        match_log_df["Season"] = season
        match_log_df["Player"] = player
        match_log_df["Link"] = link

        return match_log_df


if __name__ == "__main__":
    with FBRefDriver(headless=False) as d:
        match_log_df = d.get_player_match_log(
            "season",
            "player",
            "pos",
            "https://fbref.com/en/players/2973d8ff/matchlogs/2016-2017/Michy-Batshuayi-Match-Logs",
        )
    pass
