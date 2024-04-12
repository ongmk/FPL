import logging
import time

import pandas as pd

logger = logging.getLogger(__name__)
from io import StringIO
from urllib.parse import urljoin

import requests
from lxml import etree
from selenium import webdriver
from selenium.common.exceptions import StaleElementReferenceException, TimeoutException
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.support.ui import WebDriverWait
from tenacity import retry, stop_after_attempt
from webdriver_manager.chrome import ChromeDriverManager


class DelayedRequests:
    def __init__(self, delay=3):
        self.last_visit = 0
        self.delay = delay

    def request(self, *args, **kwargs):
        delay = 3
        time_since_last_visit = time.time() - self.last_visit
        if time_since_last_visit <= delay:
            logger.info(f"sleep for {delay-time_since_last_visit:.2f}s")
            time.sleep(delay - time_since_last_visit)
        self.last_visit = time.time()
        return requests.request(*args, **kwargs)


class BaseDriver:
    """Base Web Driver for handling timeouts"""

    def __init__(self, headless=True):
        service = Service(
            ChromeDriverManager().install(), 
            log_path="logs/chromedriver.log",
            service_args=['--readable-timestamp', '--log-level=INFO']
            )
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

        chrome_options.add_experimental_option('prefs', {
            "profile.managed_default_content_settings.images":2,
            "profile.default_content_setting_values.notifications":2,
            "profile.managed_default_content_settings.stylesheets":2,
            "profile.managed_default_content_settings.cookies":2,
            "profile.managed_default_content_settings.popups":2,
            "profile.managed_default_content_settings.geolocation":2,
            "profile.managed_default_content_settings.media_stream":2,
        }) # disable stuff
        chrome_options.add_argument('--disable-browser-side-navigation') # Fix stuck on .get()

        # logging
        chrome_options.add_argument('--log-level=3')
        
        if headless: chrome_options.add_argument("--headless") 
        chrome_options.binary_location = (
            "chromedriver//chrome-win64//chrome.exe"
        )
        chrome_options.headless = headless

        self.driver: webdriver.Chrome = webdriver.Chrome(
            service=service, options=chrome_options
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

    def get(self, url):
        # One request every three seconds
        delay = 3
        time_since_last_visit = time.time() - self.last_visit
        if time_since_last_visit <= delay:
            logger.info(f"sleep for {delay-time_since_last_visit:.2f}s")
            time.sleep(delay - time_since_last_visit)
        self.last_visit = time.time()

        res = self.driver.get(url)
        return res

    @retry(stop=stop_after_attempt(3))
    def get_tree_by_id(self, id):
        try:
            element = WebDriverWait(self.driver, timeout=10).until(
                ec.presence_of_element_located((By.ID, id))
            )
            parser = etree.HTMLParser()
            tree = etree.parse(StringIO(element.get_attribute("innerHTML")), parser)
            return tree
        except TimeoutException as e:
            logger.warning(f"Can't find element with ID={id}. Retrying...")
            self.driver.refresh()
            raise e

    @retry(stop=stop_after_attempt(3))
    def get_tree_by_xpath(self, xpath):
        try:
            element = WebDriverWait(self.driver, timeout=10).until(
                ec.presence_of_element_located((By.XPATH, xpath))
            )
            parser = etree.HTMLParser()
            tree = etree.parse(StringIO(element.get_attribute("innerHTML")), parser)
            return tree
        except TimeoutException as e:
            logger.warning(f"Can't find element with XPath={xpath}. Retrying...")
            self.driver.refresh()
            raise e

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

    def click_elements_by_xpath(self, xpath, reverse=True):
        elements = self.driver.find_elements(By.XPATH, xpath)
        elements = list(reversed(elements)) if reverse else elements
        for e in elements:
            self.driver.execute_script("arguments[0].scrollIntoView();", e)
            self.driver.execute_script("arguments[0].click();", e)


if __name__ == "__main__":
    driver = BaseDriver(headless=False)
    driver.get("https://fbref.com/en/players/2973d8ff/matchlogs/2016-2017/Michy-Batshuayi-Match-Logs")
    pass