# -*- coding: utf-8 -*-
# @Time    : 2025/1/2
# @Author  : wenshao
# @ProjectName: browser-use-webui
# @FileName: browser.py

from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContextConfig, BrowserContext
from .config import BrowserPersistenceConfig

from .custom_context import CustomBrowserContext


class CustomBrowser(Browser):
    async def new_context(
        self,
        config: BrowserContextConfig = BrowserContextConfig(),
        context: CustomBrowserContext = None, # type: ignore
    ) -> BrowserContext:
        """Create a browser context"""
        return CustomBrowserContext(config=config, browser=self, context=context)

    async def close(self):
        """Override close to respect persistence setting"""
        # Check if persistence is enabled before closing
        persistence_config = BrowserPersistenceConfig.from_env()
        if not persistence_config.persistent_session:
            await super().close()
