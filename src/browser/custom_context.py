# -*- coding: utf-8 -*-
# @Time    : 2025/1/1
# @Author  : wenshao
# @Email   : wenshaoguo1026@gmail.com
# @Project : browser-use-webui
# @FileName: context.py

import asyncio
import base64
import json
import logging
import os

from playwright.async_api import Browser as PlaywrightBrowser
from browser_use.browser.context import BrowserContext, BrowserContextConfig
from browser_use.browser.browser import Browser

from .config import BrowserPersistenceConfig

logger = logging.getLogger(__name__)


class CustomBrowserContext(BrowserContext):
    def __init__(
        self,
        browser: "Browser",
        config: BrowserContextConfig = BrowserContextConfig(),
        context: BrowserContext = None,
    ):
        super(CustomBrowserContext, self).__init__(browser, config)
        self.context = context
        self.persistence_config = BrowserPersistenceConfig.from_env()

    async def __aenter__(self):
        """Override aenter to handle persistence"""
        context = await super().__aenter__()
        return context

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Override aexit to respect persistence setting"""
        if not self.persistence_config.persistent_session:
            await super().__aexit__(exc_type, exc_val, exc_tb)
        return None

    async def _create_context(self, browser: PlaywrightBrowser):
        """Creates a new browser context with anti-detection measures and loads cookies if available."""
        if self.context:
            return self.context

        # Check for persistent session first
        if self.persistence_config.persistent_session and len(browser.contexts) > 0:
            logger.info("Using existing persistent browser context")
            return browser.contexts[0]

        # Fall back to chrome instance path check for backward compatibility
        if self.browser.config.chrome_instance_path and len(browser.contexts) > 0:
            # Connect to existing Chrome instance instead of creating new one
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
                record_video_size=self.config.browser_window_size,  # set record video size
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
        if not self.persistence_config.persistent_session:
            await super().close()
