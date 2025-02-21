import pdb

import pyperclip
from typing import Optional, Type
from pydantic import BaseModel
from browser_use.agent.views import ActionResult
from browser_use.browser.context import BrowserContext
from browser_use.controller.service import Controller, DoneAction
from main_content_extractor import MainContentExtractor
from browser_use.controller.views import (
    ClickElementAction,
    DoneAction,
    ExtractPageContentAction,
    GoToUrlAction,
    InputTextAction,
    OpenTabAction,
    ScrollAction,
    SearchGoogleAction,
    SendKeysAction,
    SwitchTabAction,
)
import logging

logger = logging.getLogger(__name__)


class CustomController(Controller):
    def __init__(self, exclude_actions: list[str] = [],
                 output_model: Optional[Type[BaseModel]] = None
                 ):
        super().__init__(exclude_actions=exclude_actions, output_model=output_model)
        self._register_custom_actions()

    def _register_custom_actions(self):
        """Register all custom browser actions"""

        @self.registry.action("Copy text to clipboard")
        def copy_to_clipboard(text: str):
            pyperclip.copy(text)
            return ActionResult(extracted_content=text)

        @self.registry.action("Paste text from clipboard", requires_browser=True)
        async def paste_from_clipboard(browser: BrowserContext):
            text = pyperclip.paste()
            # send text to browser
            page = await browser.get_current_page()
            await page.keyboard.type(text)

            return ActionResult(extracted_content=text)

        @self.registry.action(
            'Extract page content to get the pure text or markdown with links if include_links is set to true',
            param_model=ExtractPageContentAction,
            requires_browser=True,
        )
        async def extract_content(params: ExtractPageContentAction, browser: BrowserContext):
            page = await browser.get_current_page()
            url = page.url
            
            # Special handling for X/Twitter
            if 'x.com' in url or 'twitter.com' in url:
                # Get tweets directly from the page
                tweets = await page.query_selector_all('article[data-testid="tweet"]')
                content = []
                for tweet in tweets:
                    try:
                        author = await tweet.query_selector('div[data-testid="User-Name"]')
                        text = await tweet.query_selector('div[data-testid="tweetText"]')
                        if author and text:
                            author_text = await author.inner_text()
                            tweet_text = await text.inner_text()
                            content.append(f"Author: {author_text}\nTweet: {tweet_text}\n")
                    except Exception as e:
                        continue
                content = "\n".join(content) if content else "No tweets found or unable to access content"
            else:
                # Use jina reader for other sites
                jina_url = f"https://r.jina.ai/{url}"
                await page.goto(jina_url)
                output_format = 'markdown' if params.include_links else 'text'
                content = MainContentExtractor.extract(  # type: ignore
                    html=await page.content(),
                    output_format=output_format,
                )
                # go back to org url
                await page.go_back()
            
            msg = f'Extracted page content:\n {content}\n'
            logger.info(msg)
            return ActionResult(extracted_content=msg)
