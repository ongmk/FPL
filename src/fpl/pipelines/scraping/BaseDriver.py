import logging
import time
from io import StringIO
from urllib.parse import urljoin

import pandas as pd
from lxml import etree
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.support.ui import WebDriverWait
from selenium_stealth import stealth
from tenacity import retry, stop_after_attempt

from fpl.utils import backup_latest_n

logger = logging.getLogger(__name__)


class BaseDriver:
    """Base Web Driver for handling timeouts"""

    def __init__(self, headless=True):
        chrome_options = webdriver.ChromeOptions()
        # https://github.com/GoogleChrome/chrome-launcher/blob/main/docs/chrome-flags-for-tools.md
        chrome_flags_for_tooling = [
            "--disable-client-side-phishing-detection",
            "--disable-component-extensions-with-background-pages",
            "--disable-extensions",
            "--disable-default-apps",
            "--disable-extensions",
            "--disable-features=InterestFeedContentSuggestions",
            "--disable-features=Translate",
            "--hide-scrollbars",
            "--mute-audio",
            "--no-default-browser-check",
            "--no-first-run",
            "--ash-no-nudges",
            "--disable-search-engine-choice-screen",
            "--disable-background-timer-throttling",
            "--disable-backgrounding-occluded-windows",
            "--disable-features=CalculateNativeWinOcclusion",
            "--disable-hang-monitor",
            "--disable-ipc-flooding-protection",
            "--disable-renderer-backgrounding",
            "--autoplay-policy=user-gesture-required",
            "--deny-permission-prompts",
            "--disable-external-intent-requests",
            "--disable-features=GlobalMediaControls",
            "--disable-features=ImprovedCookieControls",
            "--disable-features=PrivacySandboxSettings4",
            "--disable-notifications",
            "--disable-popup-blocking",
            "--disable-prompt-on-repost",
            "--noerrdialogs",
            "--enable-automation",
            "--disable-background-networking",
            "--disable-breakpad",
            "--disable-component-update",
            "--disable-domain-reliability",
            "--disable-features=AutofillServerCommunication",
            "--disable-features=CertificateTransparencyComponentUpdater",
            "--disable-sync",
            "--enable-crash-reporter-for-testing",
            "--metrics-recording-only",
            "--disable-features=OptimizationHints",
            "--disable-features=DialMediaRouteProvider",
            "--no-pings",
            "--no-sandbox",
            "--disable-gpu",
        ]
        for flag in chrome_flags_for_tooling:
            chrome_options.add_argument(flag)

        chrome_options.add_experimental_option(
            "prefs",
            {
                "profile.managed_default_content_settings.images": 2,
                "profile.default_content_setting_values.notifications": 2,
                "profile.managed_default_content_settings.stylesheets": 2,
                "profile.managed_default_content_settings.cookies": 2,
                "profile.managed_default_content_settings.popups": 2,
                "profile.managed_default_content_settings.geolocation": 2,
                "profile.managed_default_content_settings.media_stream": 2,
            },
        )  # disable stuff

        # Fix stuck on .get()
        chrome_options.add_argument("--disable-browser-side-navigation")

        # logging
        chrome_options.add_argument("--log-level=3")
        chrome_options.add_argument("--log-file=logs/chromedriver.log")

        if headless:
            chrome_options.add_argument("--headless")

        self.driver: webdriver.Chrome = webdriver.Chrome(
            options=chrome_options,
        )
        stealth(
            self.driver,
            languages=["en-US", "en"],
            vendor="Google Inc.",
            platform="Win32",
            webgl_vendor="Intel Inc.",
            renderer="Intel Iris OpenGL Engine",
            fix_hairline=True,
        )
        self.base_url = ""
        self.last_visit = 0

    def absolute_url(self, relative_url):
        return urljoin(self.base_url, relative_url)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.driver.close()

    def get_cookies(self):
        cookie_dict = {}
        for cookie in self.driver.get_cookies():
            cookie_dict[cookie["name"]] = cookie["value"]
        return cookie_dict

    def sleep(self, seconds=6):
        # One request every three seconds
        time_since_last_visit = time.time() - self.last_visit
        if time_since_last_visit <= seconds:
            logger.info(f"sleep for {seconds - time_since_last_visit:.2f}s")
            time.sleep(seconds - time_since_last_visit)
        self.last_visit = time.time()

    @retry(stop=stop_after_attempt(3))
    def get(self, url):
        self.sleep()
        res = self.driver.get(url)
        return res

    def get_tree_by_id(self, id):
        element = self.get_element_by_id(id)
        parser = etree.HTMLParser()
        tree = etree.parse(StringIO(element.get_attribute("innerHTML")), parser)
        return tree

    @staticmethod
    def save_debugging_html(html_content):
        filename = "tmp/html_debug/debug.html"
        with open(filename, "w", encoding="utf-8") as file:
            file.write(html_content)
        backup_latest_n(filename, 5)

    @retry(stop=stop_after_attempt(3))
    def get_element_by_id(self, id):
        try:
            element = WebDriverWait(self.driver, timeout=10).until(
                ec.presence_of_element_located((By.ID, id))
            )
            return element

        except TimeoutException as e:
            logger.warning(f"Can't find element with ID={id}. Retrying...")
            self.save_debugging_html(self.driver.page_source)
            self.sleep()
            self.driver.refresh()
            raise e

    @retry(stop=stop_after_attempt(3))
    def get_element_by_xpath(self, xpath):
        try:
            element = WebDriverWait(self.driver, timeout=10).until(
                ec.presence_of_element_located((By.XPATH, xpath))
            )
            return element
        except TimeoutException as e:
            logger.warning(f"Can't find element with XPath={xpath}. Retrying...")
            self.sleep()
            self.driver.refresh()
            raise e

    def get_tree_by_xpath(self, xpath):
        element = self.get_element_by_xpath(xpath)
        parser = etree.HTMLParser()
        tree = etree.parse(StringIO(element.get_attribute("innerHTML")), parser)
        return tree

    def get_table_text(self, cell):
        if list(cell) == []:
            return cell.text
        else:
            if cell.xpath("./a") != []:
                return cell.xpath("./a")[0].text
            if cell.xpath('./span[@class="venuetime"]') != []:
                return cell.xpath('./span[@class="venuetime"]')[0].text
            if cell.xpath("./small") != []:
                return cell.text
            if cell.xpath('./span[@class="bold"]') != []:
                return "".join(cell.itertext())
            return cell.text

    def get_table_df_by_id(self, id):
        tree = self.get_tree_by_id(id)

        columns = tree.xpath("*/thead/tr[not (@class)]/th")
        columns = [c.text for c in columns]
        rows = tree.xpath("*/tbody/tr[not (@class)]")
        content = []
        for tr in rows:
            td = tr.xpath(".//*[self::th or self::td]")
            # get last child element's text
            td = [self.get_table_text(d) for d in td]
            content.append(td)
        return pd.DataFrame(content, columns=columns)

    def inifinte_scroll(self, scroll_times=999, container_id=None):
        n = 0
        last_height = 0
        new_height = 1
        element = self.get_element_by_id(container_id)
        while new_height != last_height and n < scroll_times:
            logger.info(f"Scroll {n}\t{last_height}->{new_height}")
            last_height = new_height
            if container_id:
                self.driver.execute_script(
                    "arguments[0].scrollIntoView(false);", element
                )
                self.sleep()
                new_height = int(element.get_attribute("scrollHeight"))
            else:
                self.driver.execute_script(
                    "window.scrollTo(0, document.body.scrollHeight);"
                )
                self.sleep()
                new_height = int(
                    self.driver.execute_script("return document.body.scrollHeight")
                )
            n += 1
        return None


if __name__ == "__main__":
    driver = BaseDriver(headless=True)
    driver.get(
        "https://fbref.com/en/players/2973d8ff/matchlogs/2016-2017/Michy-Batshuayi-Match-Logs"
    )
    driver.save_debugging_html(driver.driver.page_source)
    pass
