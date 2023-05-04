from logzero import logger
import logzero

logzero.logfile("scraper.log", maxBytes=1e6, disableStderrLogger=True, encoding='utf-8')
