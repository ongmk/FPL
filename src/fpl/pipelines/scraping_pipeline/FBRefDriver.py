import logging

import pandas as pd

from fpl.pipelines.scraping_pipeline.BaseDriver import BaseDriver

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
        match_played = tree.xpath("*/tbody/tr/td[@data-stat='games']")
        matched_played = sum([int(m.text) for m in match_played])
        if matched_played == 0:
            logger.warn(f"No match played for {season}")
            return []
        links = tree.xpath("*/tbody/tr/td[1]/a")
        return [(l.text, self.absolute_url(l.get("href"))) for l in links]

    def get_player_season_links(self, season: str) -> list[str, str, str]:
        url = self.absolute_url(
            f"/en/comps/9/{season}/stats/{season}-Premier-League-Stats"
        )
        logger.info(f"Crawling Season Result:\t{season}\t{url}")
        self.get(url)
        try:
            tree = self.get_tree_by_id(f"stats_standard")
        except Exception as e:
            logger.warn(f"No player match logs to fetch for {season}")
            return []
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

    def get_most_recent_game(self, current_season):
        url = self.absolute_url(
            f"/en/comps/9/{current_season}/schedule/{current_season}-Premier-League-Scores-and-Fixtures"
        )
        logger.info(f"Getting all fixtures:\t{url}")
        self.get(url)
        fixtures_df = self.get_table_df_by_id(f"sched_{current_season}_9_1")
        completed = fixtures_df.loc[~fixtures_df["Score"].isna()]
        if len(completed) == 0:
            return None, None, None
        most_recent_game = completed.sort_values(
            by=["Date", "Time"], ascending=False
        ).iloc[0]
        date = most_recent_game["Date"]
        home = most_recent_game["Home"]
        away = most_recent_game["Away"]
        return date, home, away

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
        pos_mode = match_log_df["Pos"].mode()
        fill_val = pos_mode.values[0] if len(pos_mode) > 0 else pos
        match_log_df["Pos"] = match_log_df["Pos"].fillna(fill_val)
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
