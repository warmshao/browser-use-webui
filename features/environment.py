from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContextConfig, BrowserContextWindowSize
from src.utils import utils
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

def before_all(context):
    # Create a single event loop for the entire test run
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    context.loop = loop
    
    # Set up browser configuration
    window_w, window_h = 1920, 1080
    context.browser = Browser(
        config=BrowserConfig(
            headless=False,
            disable_security=True,
            chrome_instance_path=os.getenv("CHROME_PATH"),
            extra_chromium_args=[f"--window-size={window_w},{window_h}"],
        )
    )
    
    # Set up LLM using environment variables
    llm_provider = os.getenv('TEST_LLM_PROVIDER', 'ollama')
    llm_model = os.getenv('TEST_LLM_MODEL', 'deepseek-r1:14b')
    
    context.llm = utils.get_llm_model(
        provider=llm_provider,
        model_name=llm_model,
        temperature=0.5
    )

def before_scenario(context, scenario):
    # Create new browser context using the existing event loop
    async def setup_browser():
        context.browser_context = await context.browser.new_context(
            config=BrowserContextConfig(
                trace_path="./tmp/traces",
                save_recording_path="./tmp/record_videos",
                no_viewport=False,
                browser_window_size=BrowserContextWindowSize(
                    width=1920,
                    height=1080
                ),
            )
        )
    context.loop.run_until_complete(setup_browser())

def after_scenario(context, scenario):
    if hasattr(context, 'browser_context'):
        async def cleanup_browser():
            await context.browser_context.close()
        context.loop.run_until_complete(cleanup_browser())

def after_all(context):
    if hasattr(context, 'browser'):
        async def cleanup():
            await context.browser.close()
        context.loop.run_until_complete(cleanup())
        
        # Clean up the event loop at the very end
        context.loop.close()
        asyncio.set_event_loop(None) 