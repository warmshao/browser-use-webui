import asyncio
import pdb

from playwright.async_api import Browser as PlaywrightBrowser
from playwright.async_api import (
	BrowserContext as PlaywrightBrowserContext,
)
from playwright.async_api import (
	Playwright,
	async_playwright,
)
from browser_use.browser.browser import Browser
from browser_use.browser.context import BrowserContext, BrowserContextConfig
from playwright.async_api import BrowserContext as PlaywrightBrowserContext
import logging

from .custom_context import CustomBrowserContext

logger = logging.getLogger(__name__)

class CustomBrowser(Browser):

    async def new_context(
        self,
        config: BrowserContextConfig = BrowserContextConfig()
    ) -> CustomBrowserContext:
        return CustomBrowserContext(config=config, browser=self)

    async def _setup_browser(self, playwright: Playwright) -> PlaywrightBrowser:
        """Sets up and returns a Playwright Browser instance with anti-detection measures."""
        if self.config.wss_url:
            browser = await playwright.chromium.connect(self.config.wss_url)
            return browser
        elif self.config.chrome_instance_path:
            import subprocess

            import requests

            try:
                # Check if browser is already running
                response = requests.get('http://localhost:9222/json/version', timeout=2)
                if response.status_code == 200:
                    logger.info('Reusing existing Chrome instance')
                    browser = await playwright.chromium.connect_over_cdp(
                        endpoint_url='http://localhost:9222',
                        timeout=20000,  # 20 second timeout for connection
                    )
                    return browser
            except requests.ConnectionError:
                logger.debug('No existing Chrome instance found, starting a new one')

            # Start a new Chrome instance
            subprocess.Popen(
                [
                    self.config.chrome_instance_path,
                    '--remote-debugging-port=9222',
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # Attempt to connect again after starting a new instance
            for _ in range(10):
                try:
                    response = requests.get('http://localhost:9222/json/version', timeout=2)
                    if response.status_code == 200:
                        break
                except requests.ConnectionError:
                    pass
                await asyncio.sleep(1)

            try:
                browser = await playwright.chromium.connect_over_cdp(
                    endpoint_url='http://localhost:9222',
                    timeout=20000,  # 20 second timeout for connection
                )
                return browser
            except Exception as e:
                logger.error(f'Failed to start a new Chrome instance.: {str(e)}')
                raise RuntimeError(
                    ' To start chrome in Debug mode, you need to close all existing Chrome instances and try again otherwise we can not connect to the instance.'
                )

        else:
            try:
                disable_security_args = []
                if self.config.disable_security:
                    disable_security_args = [
                        '--disable-web-security',
                        '--disable-site-isolation-trials',
                        '--disable-features=IsolateOrigins,site-per-process',
                    ]

                browser = await playwright.chromium.launch(
                    headless=self.config.headless,
                    args=[
                             '--no-sandbox',
                             '--disable-blink-features=AutomationControlled',
                             '--disable-infobars',
                             '--disable-background-timer-throttling',
                             '--disable-popup-blocking',
                             '--disable-backgrounding-occluded-windows',
                             '--disable-renderer-backgrounding',
                             '--disable-window-activation',
                             '--disable-focus-on-load',
                             '--no-first-run',
                             '--no-default-browser-check',
                             '--no-startup-window',
                             '--window-position=0,0',
                             # '--window-size=1280,1000',
                         ]
                         + disable_security_args
                         + self.config.extra_chromium_args,
                    proxy=self.config.proxy,
                )

                return browser
            except Exception as e:
                logger.error(f'Failed to initialize Playwright browser: {str(e)}')
                raise
