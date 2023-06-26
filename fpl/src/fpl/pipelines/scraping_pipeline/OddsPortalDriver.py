import logging
import re

import pandas as pd
from lxml import etree
from src.fpl.pipelines.scraping_pipeline.BaseDriver import BaseDriver, DelayedRequests

logger = logging.getLogger(__name__)


class OddsPortalDriver(BaseDriver):
    """Custom web driver for OddsPortal.com"""

    def __init__(self):
        BaseDriver.__init__(self)
        self.base_url = "https://www.oddsportal.com/"
        self.requests = DelayedRequests()

    def get_page_1_url(self):
        container = self.get_tree_by_xpath(
            '//div[@class="max-sx:!hidden"]/div/div[@id="pagination"]'
        )
        pagination_links = container.xpath("//a")
        hrefs = [element.get(":href") for element in pagination_links]

        page_1_urls = set()
        for url in hrefs:
            match = re.search(r"/page/(\d+)/", url)
            if match:
                page_1_url = re.sub(r"/page/\d+/", "/page/1/", url)
                page_1_urls.add(page_1_url)
        if len(page_1_urls) > 1:
            raise ValueError("More than one possible page 1 url.")
        return next(iter(page_1_urls))

    def get_match_links(self, season):
        relative_url = f"/football/england/premier-league-{season}/results/"
        url = self.absolute_url(relative_url)
        self.get(url)

        next_page = self.get_page_1_url()
        match_links = []
        while True:
            cookies: dict[str, str] = self.get_cookies()
            headers = {
                "authority": "www.oddsportal.com",
                "accept": "application/json, text/plain, */*",
                "accept-language": "en-GB,en;q=0.9,zh-HK;q=0.8,zh;q=0.7,en-US;q=0.6,zh-TW;q=0.5,zh-CN;q=0.4,ja;q=0.3",
                "cache-control": "no-cache",
                "content-type": "application/json",
                "dnt": "1",
                "pragma": "no-cache",
                "referer": "https://www.oddsportal.com/football/england/premier-league-2021-2022/results/",
                "sec-ch-ua": '"Not.A/Brand";v="8", "Chromium";v="114", "Google Chrome";v="114"',
                "sec-ch-ua-mobile": "?0",
                "sec-ch-ua-platform": '"Windows"',
                "sec-fetch-dest": "empty",
                "sec-fetch-mode": "cors",
                "sec-fetch-site": "same-origin",
                "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
                "x-requested-with": "XMLHttpRequest",
            }
            # headers["x-xsrf-token"] = cookies.pop("XSRF-TOKEN")
            headers["cookie"] = "; ".join(
                [f"{key}={value}" for key, value in cookies.items()]
            )

            response = self.requests.request("GET", next_page, headers=headers)
            response = response.json()
            rows = response["d"]["rows"]
            match_links += [(r["name"], r["url"]) for r in rows]
            pagination_html = response["d"]["paginationView"]
            pagination = etree.fromstring(pagination_html, parser=etree.HTMLParser())
            next_page = pagination.xpath("//div/div[3]/a[.//p[.='next']]")
            if len(next_page) == 1:
                next_page = next_page[0].get(":href")
                return match_links
            else:
                return match_links

    def get_row_data(self, row):
        h_score, a_score = row.xpath("./div[2]/p[1]")[0].text.split(":")
        odds = row.xpath("./div[3]/div/div/div/p")[0].text
        return int(h_score), int(a_score), float(odds)

    def get_match_odds_df(self, season, h_team, a_team, link):
        link = link + "#cs;2"
        logger.info(f"Crawling Match Odds:\t{season} {h_team}-{a_team}\t{link}")
        self.get(link)
        self.get_tree_by_xpath(
            '//*[@id="app"]/div/div[1]/div/main/div[2]/div[4]/div[1]/div/div[2]'
        )
        table = self.get_tree_by_xpath(
            '//*[@id="app"]/div/div[1]/div/main/div[2]/div[4]'
        )
        rows = table.xpath("./body/div/div[./*]")
        rows = [(self.get_row_data(r)) for r in rows]
        match_odds_df = pd.DataFrame(rows, columns=["h_score", "a_score", "odds"])
        match_odds_df = match_odds_df.sort_values("odds").head(10)
        match_odds_df["Season"] = season
        match_odds_df["h_team"] = h_team
        match_odds_df["a_team"] = a_team
        match_odds_df["Link"] = link

        return match_odds_df

    def expand_rows(self):
        self.click_elements_by_xpath(
            '//*[@id="app"]/div/div[1]/div/main/div[2]/div[4]/div/div/div[1]'
        )


if __name__ == "__main__":
    with OddsPortalDriver() as d:
        match_odds_df = d.get_match_odds_df(
            "season",
            "team",
            "opponent",
            "https://www.oddsportal.com/football/england/premier-league-2022-2023/arsenal-wolves-M1w8YmqE/",
        )
        print(match_odds_df)
