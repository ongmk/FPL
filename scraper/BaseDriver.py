import time
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import pandas as pd
from logger import logger
from lxml import etree
from io import StringIO
from urllib.parse import urljoin


from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
from selenium.common.exceptions import TimeoutException


class BaseDriver:
    """Base Web Driver for handling timeouts"""

    def __init__(self):
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument("--blink-settings=imagesEnabled=false")

        self.driver = webdriver.Chrome(options=chrome_options)
        self.base_url = ""
        self.last_visit = 0

    def absolute_url(self, relative_url):
        return urljoin(self.base_url, relative_url)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.driver.close()

    def get(self, url):
        # One request every three seconds
        delay = 3
        time_since_last_visit = time.time() - self.last_visit
        if time_since_last_visit <= delay:
            logger.info(f"sleep for {delay-time_since_last_visit}")
            time.sleep(delay - time_since_last_visit)
        self.last_visit = time.time()

        res = self.driver.get(url)
        return res

    def get_tree_by_id(self, id):
        for attempt in range(3):
            try:
                element = WebDriverWait(self.driver, timeout=30).until(
                    ec.presence_of_element_located((By.ID, id))
                )
                parser = etree.HTMLParser()
                tree = etree.parse(StringIO(element.get_attribute("innerHTML")), parser)
                return tree
            except TimeoutException as e:
                logger.warning("Webdriver Timeout. Retrying...")
                self.driver.refresh()
                continue
        raise Exception("Failed 3 retries!")

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
