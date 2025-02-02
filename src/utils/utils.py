import base64
import os
import time
from pathlib import Path
from typing import Dict, Optional
import requests

from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from .llm import DeepSeekR1ChatOpenAI, DeepSeekR1ChatOllama

from .llm_providers import (
    get_provider_config, get_config_value,
    PROVIDER_DISPLAY_NAMES, MODEL_NAMES
)

def get_llm_model(provider: str, **kwargs):
    """
    获取LLM 模型
    :param provider: 模型类型
    :param kwargs:
    :return:
    """
    if provider not in {"ollama"}:
        api_key = get_config_value(provider, "api_key", **kwargs)
        if not api_key:
            raise MissingAPIKeyError(provider)

    base_url = get_config_value(provider, "base_url", **kwargs)
    model_name = get_config_value(provider, "model", **kwargs)
    temperature = kwargs.get("temperature", 0.0)

    if provider == "anthropic":
        return ChatAnthropic(
            model=model_name,
            temperature=temperature,
            base_url=base_url,
            api_key=api_key,
        )
    elif provider == "mistral":
        return ChatMistralAI(
            model=model_name,
            temperature=temperature,
            base_url=base_url,
            api_key=api_key,
        )
    elif provider == "openai":
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            base_url=base_url,
            api_key=api_key,
        )
    elif provider == "deepseek":
        if model_name == "deepseek-reasoner":
            return DeepSeekR1ChatOpenAI(
                model=model_name,
                temperature=temperature,
                base_url=base_url,
                api_key=api_key,
            )
        else:
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                base_url=base_url,
                api_key=api_key,
            )
    elif provider == "google":
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
        )
    elif provider == "ollama":
        num_ctx = kwargs.get("num_ctx", 32000)
        if "deepseek-r1" in model_name:
            return DeepSeekR1ChatOllama(
                model=kwargs.get("model_name", "deepseek-r1:14b"),
                temperature=temperature,
                num_ctx=num_ctx,
                base_url=base_url,
            )
        else:
            return ChatOllama(
                model=model_name,
                temperature=temperature,
                num_ctx=num_ctx,
                num_predict=kwargs.get("num_predict", 1024),
                base_url=base_url,
            )
    elif provider == "azure_openai":
        api_version = get_config_value(provider, "api_version", **kwargs)
        return AzureChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_version=api_version,
            azure_endpoint=base_url,
            api_key=api_key,
        )
    elif provider == "alibaba":
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            base_url=base_url,
            api_key=api_key,
        )

    elif provider == "moonshot":
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            base_url=base_url,
            api_key=api_key,
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

class MissingAPIKeyError(Exception):
    """Custom exception raised when an API key is missing."""
    def __init__(self, provider: str):
        self.provider = provider
        super().__init__(self.message())

    def message(self):
        provider_display = PROVIDER_DISPLAY_NAMES.get(self.provider, self.provider.upper())
        config = get_provider_config(self.provider)
        env_var = config.get("api_key_env")
        return (f"💥 {provider_display} API key not found! 🔑 Please set the "
                f"`{env_var}` environment variable or provide it in the UI.")

def update_model_dropdown(llm_provider, api_key=None, base_url=None):
    """
    Update the model name dropdown with predefined models for the selected provider.
    """
    import gradio as gr
    # Use API keys from .env if not provided
    if not api_key:
        api_key = get_config_value(llm_provider, "api_key")
    if not base_url:
        base_url = get_config_value(llm_provider, "base_url")

    # Use predefined models for the selected provider
    if llm_provider in MODEL_NAMES:
        return gr.Dropdown(choices=MODEL_NAMES[llm_provider], value=MODEL_NAMES[llm_provider][0], interactive=True)
    return gr.Dropdown(choices=[], value="", interactive=True, allow_custom_value=True)

def encode_image(img_path):
    if not img_path:
        return None
    with open(img_path, "rb") as fin:
        image_data = base64.b64encode(fin.read()).decode("utf-8")
    return image_data

def get_latest_files(directory: str, file_types: list = ['.webm', '.zip']) -> Dict[str, Optional[str]]:
    """Get the latest recording and trace files"""
    latest_files: Dict[str, Optional[str]] = {ext: None for ext in file_types}

    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        return latest_files

    for file_type in file_types:
        try:
            matches = list(Path(directory).rglob(f"*{file_type}"))
            if matches:
                latest = max(matches, key=lambda p: p.stat().st_mtime)
                # Only return files that are complete (not being written)
                if time.time() - latest.stat().st_mtime > 1.0:
                    latest_files[file_type] = str(latest)
        except Exception as e:
            print(f"Error getting latest {file_type} file: {e}")

    return latest_files

async def capture_screenshot(browser_context):
    """Capture and encode a screenshot"""
    # Extract the Playwright browser instance
    playwright_browser = browser_context.browser.playwright_browser  # Ensure this is correct.

    # Check if the browser instance is valid and if an existing context can be reused
    if playwright_browser and playwright_browser.contexts:
        playwright_context = playwright_browser.contexts[0]
    else:
        return None

    # Access pages in the context
    pages = None
    if playwright_context:
        pages = playwright_context.pages

    # Use an existing page or create a new one if none exist
    if pages:
        active_page = pages[0]
        for page in pages:
            if page.url != "about:blank":
                active_page = page
    else:
        return None

    # Take screenshot
    try:
        screenshot = await active_page.screenshot(
            type='jpeg',
            quality=75,
            scale="css"
        )
        encoded = base64.b64encode(screenshot).decode('utf-8')
        return encoded
    except Exception:
        return None
