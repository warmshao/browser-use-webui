# -*- coding: utf-8 -*-
# @Time    : 2025/1/2
# @Author  : wenshao
# @ProjectName: browser-use-webui
# @FileName: browser.py

from browser_use.browser.browser import Browser
from browser_use.browser.context import BrowserContext, BrowserContextConfig
from playwright.async_api import BrowserContext as PlaywrightBrowserContext
import logging

from .config import BrowserPersistenceConfig
from .custom_context import CustomBrowserContext

logger = logging.getLogger(__name__)

class CustomBrowser(Browser):
    _global_context = None

    async def new_context(
        self,
        config: BrowserContextConfig = BrowserContextConfig(),
        context: PlaywrightBrowserContext = None,
    ) -> CustomBrowserContext:
        """Create a browser context with persistence support"""
        persistence_config = BrowserPersistenceConfig.from_env()
        
        if persistence_config.persistent_session:
            if CustomBrowser._global_context is not None:
                logger.info("Reusing existing persistent browser context")
                return CustomBrowser._global_context
            
            context_instance = CustomBrowserContext(config=config, browser=self, context=context)
            CustomBrowser._global_context = context_instance
            logger.info("Created new persistent browser context")
            return context_instance
        
        logger.info("Creating non-persistent browser context")
        return CustomBrowserContext(config=config, browser=self, context=context)

    async def close(self):
        """Override close to respect persistence setting"""
        persistence_config = BrowserPersistenceConfig.from_env()
        if not persistence_config.persistent_session:
            if CustomBrowser._global_context is not None:
                await CustomBrowser._global_context.close()
                CustomBrowser._global_context = None
            await super().close()
        else:
            logger.info("Skipping browser close due to persistent session")
