# -*- coding: utf-8 -*-
# @Time    : 2025/1/2
# @Author  : wenshao
# @ProjectName: browser-use-webui
# @FileName: browser.py

import logging
from typing import Optional
import playwright.async_api
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContext, BrowserContextConfig, BrowserSession
from .custom_context import CustomBrowserContext

logger = logging.getLogger(__name__)

class CustomBrowser(Browser):
    def __init__(self, config: BrowserConfig):
        super().__init__(config)
        self._browser = None
        self._playwright = None

    @property
    def browser(self):
        return self._browser

    async def launch(self):
        """Launch the browser with configured settings"""
        if not self._playwright:
            import playwright.async_api
            self._playwright = await playwright.async_api.async_playwright().start()

        if not self._browser:
            self._browser = await self._playwright.chromium.launch(
                headless=self.config.headless,
                args=self.config.extra_chromium_args or [],
                executable_path=self.config.chrome_instance_path,
            )
        return self._browser

    async def new_context(
            self, config: BrowserContextConfig = BrowserContextConfig(), context: CustomBrowserContext = None
    ) -> BrowserContext:
        """Create a browser context with settings to prevent new windows and handle navigation."""
        if not self._browser:
            await self.launch()

        # Configure browser for better navigation handling
        browser_args = [
            '--disable-popup-blocking',
            '--no-sandbox',
            '--disable-setuid-sandbox',
            '--disable-dev-shm-usage',
            '--disable-accelerated-2d-canvas',
            '--no-first-run',
            '--no-zygote',
            '--disable-gpu',
            '--disable-background-timer-throttling',
            '--disable-backgrounding-occluded-windows',
            '--disable-renderer-backgrounding',
            '--disable-background-networking',
            '--window-size=1920,1080',
            '--disable-features=IsolateOrigins,site-per-process',
            '--disable-web-security',
            '--disable-site-isolation-trials'
        ]

        # Update browser configuration
        if not self.config.extra_chromium_args:
            self.config.extra_chromium_args = []
        self.config.extra_chromium_args.extend(browser_args)

        # Relaunch browser with updated settings if needed
        if self._browser:
            await self._browser.close()
            self._browser = None
            await self.launch()

        return CustomBrowserContext(browser=self, config=config, context=context)

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Ensure proper cleanup of resources"""
        if self._browser and not self.config.chrome_instance_path:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
