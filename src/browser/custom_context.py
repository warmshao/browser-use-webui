# -*- coding: utf-8 -*-
# @Time    : 2025/1/1
# @Author  : wenshao
# @Email   : wenshaoguo1026@gmail.com
# @Project : browser-use-webui
# @FileName: context.py

import asyncio
import base64
import logging
import json
import os
from typing import Optional, Type, Dict, List
from playwright.async_api import Browser as PlaywrightBrowser, BrowserContext as PlaywrightContext
from browser_use.browser.browser import Browser
from browser_use.browser.context import BrowserContext, BrowserContextConfig, BrowserSession
from browser_use.browser.views import BrowserState, TabInfo
from browser_use.dom.views import DOMElementNode, DOMBaseNode

logger = logging.getLogger(__name__)


class CustomBrowserContext(BrowserContext):

    def __init__(
            self,
            browser: Browser,
            config: BrowserContextConfig = BrowserContextConfig(),
            context: PlaywrightContext = None
    ):
        """Initialize custom browser context with proper argument order"""
        super().__init__(browser, config)
        self._context = context
        self.session = None

    async def _create_context(self, browser: PlaywrightBrowser):
        """Creates a new browser context with anti-detection measures and loads cookies if available."""
        if self._context:
            return self._context

        try:
            context = await browser.new_context(
                viewport=self.config.browser_window_size,
                no_viewport=False,
                user_agent=(
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                    '(KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36'
                ),
                java_script_enabled=True,
                bypass_csp=True,
                ignore_https_errors=True,
                record_video_dir=self.config.save_recording_path,
                record_video_size=self.config.browser_window_size,
            )

            # Create initial page
            page = await context.new_page()
            await page.goto("about:blank", wait_until="domcontentloaded")

            # Add event listener to handle new windows
            async def handle_popup(popup):
                try:
                    url = popup.url
                    main_page = context.pages[0]
                    await main_page.goto(url)
                    await popup.close()
                except Exception as e:
                    logger.error(f"Error handling popup: {str(e)}")

            context.on("page", handle_popup)

            # Configure tracing if path is set
            if self.config.trace_path:
                await context.tracing.start(screenshots=True, snapshots=True, sources=True)

            # Load cookies if they exist
            if self.config.cookies_file and os.path.exists(self.config.cookies_file):
                try:
                    with open(self.config.cookies_file, 'r') as f:
                        cookies = json.load(f)
                        logger.info(f'Loaded {len(cookies)} cookies from {self.config.cookies_file}')
                        await context.add_cookies(cookies)
                except Exception as e:
                    logger.error(f"Error loading cookies: {str(e)}")

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

                // Handle window.open
                window.open = (url) => {
                    window.location.href = url;
                    return null;
                };
                """
            )

            return context

        except Exception as e:
            logger.error(f"Error creating browser context: {str(e)}")
            raise

    async def __aenter__(self):
        """Override the base context's enter to handle navigation properly"""
        if not self._context:
            self._context = await self._create_context(self.browser.browser)
        
        # Create session without waiting for title
        if not self.session:
            # Get the first page or create one
            page = self._context.pages[0] if self._context.pages else await self._context.new_page()
            
            # Create empty DOM tree
            empty_tree = DOMElementNode(
                is_visible=True,
                parent=None,
                tag_name="html",
                xpath="/html",
                attributes={},
                children=[],
                is_interactive=False,
                is_top_element=True,
                shadow_root=False,
                highlight_index=None
            )

            # Get current page info
            try:
                title = await page.title() or "New Page"
            except Exception:
                title = "New Page"

            # Create empty state
            state = BrowserState(
                element_tree=empty_tree,
                selector_map={0: empty_tree},
                url=page.url,
                title=title,
                tabs=[TabInfo(
                    page_id=0,
                    url=page.url,
                    title=title
                )],
                screenshot=None
            )

            self.session = BrowserSession(
                context=self._context,
                current_page=page,
                cached_state=state
            )

            # Set up page event listeners for popup handling
            async def handle_popup(popup):
                try:
                    url = popup.url
                    await page.goto(url)
                    await popup.close()
                except Exception as e:
                    logger.error(f"Error handling popup: {str(e)}")

            page.on("popup", handle_popup)
        
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup resources properly"""
        if self.config.trace_path and self._context:
            await self._context.tracing.stop()
        if self._context and not self.browser.config.chrome_instance_path:
            await self._context.close()

    @property
    def context(self) -> PlaywrightContext:
        return self._context
