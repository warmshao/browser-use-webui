# -*- coding: utf-8 -*-
# @Time    : 2025/1/1
# @Author  : wenshao
# @Email   : wenshaoguo1026@gmail.com
# @Project : browser-use-webui
# @FileName: context.py

import json
import logging
import os

from browser_use.browser.browser import Browser
from browser_use.browser.context import BrowserContext, BrowserContextConfig
from playwright.async_api import Browser as PlaywrightBrowser
from playwright.async_api import BrowserContext as PlaywrightBrowserContext

from .config import BrowserPersistenceConfig
logger = logging.getLogger(__name__)


class CustomBrowserContext(BrowserContext):
    def __init__(
        self,
        browser: "Browser",
        config: BrowserContextConfig = BrowserContextConfig(),
        context: PlaywrightBrowserContext = None,
    ):
        super(CustomBrowserContext, self).__init__(browser=browser, config=config)
        self.context = context
        self._persistence_config = BrowserPersistenceConfig.from_env()

    async def _create_context(self, browser: PlaywrightBrowser) -> PlaywrightBrowserContext:
        """Creates a new browser context with anti-detection measures and loads cookies if available."""
        # If we have a context, return it directly
        if self.context:
            return self.context

        # Check if we should use existing context for persistence
        if self._persistence_config.persistent_session and len(browser.contexts) > 0:
            logger.info("Using existing persistent context")
            context = browser.contexts[0]
        else:
            # Original code for creating new context
            context = await browser.new_context(
                viewport=self.config.browser_window_size,
                no_viewport=False,
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36"
                ),
                java_script_enabled=True,
                bypass_csp=self.config.disable_security,
                ignore_https_errors=self.config.disable_security,
                record_video_dir=self.config.save_recording_path,
                record_video_size=self.config.browser_window_size,
            )

        if self.config.trace_path:
            await context.tracing.start(screenshots=True, snapshots=True, sources=True)

        # Load cookies if they exist
        if self.config.cookies_file and os.path.exists(self.config.cookies_file):
            with open(self.config.cookies_file, "r") as f:
                cookies = json.load(f)
                logger.info(
                    f"Loaded {len(cookies)} cookies from {self.config.cookies_file}"
                )
                await context.add_cookies(cookies)

        # Expose anti-detection scripts
        await context.add_init_script(
            """
            // Webdriver property
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });

            // Languages
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en']
            });

            // Plugins
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5]
            });

            // Chrome runtime
            window.chrome = { runtime: {} };

            // Permissions
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications' ?
                    Promise.resolve({ state: Notification.permission }) :
                    originalQuery(parameters)
            );
            """
        )

        return context

    async def close(self):
        """Override close to respect persistence setting"""
        if not self._persistence_config.persistent_session:
            await super().close()
