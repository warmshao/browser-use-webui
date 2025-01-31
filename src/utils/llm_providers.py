import os

PROVIDER_DISPLAY_NAMES = {
    "openai": "OpenAI",
    "azure_openai": "Azure OpenAI",
    "anthropic": "Anthropic",
    "deepseek": "DeepSeek",
    "google": "Google",
    "mistral": "Mistral",
    "ollama": "Ollama"
}

PROVIDER_CONFIGS = {
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_ENDPOINT",
        "default_base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o"
    },
    "azure_openai": {
        "api_key_env": "AZURE_OPENAI_API_KEY",
        "base_url_env": "AZURE_OPENAI_ENDPOINT",
        "default_model": "gpt-4o"
    },
    "anthropic": {
        "api_key_env": "ANTHROPIC_API_KEY",
        "base_url_env": "ANTHROPIC_ENDPOINT",
        "default_base_url": "https://api.anthropic.com",
        "default_model": "claude-3-5-sonnet-20240620"
    },
    "google": {
        "api_key_env": "GOOGLE_API_KEY",
        "default_model": "gemini-2.0-flash-exp"
    },
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url_env": "DEEPSEEK_ENDPOINT",
        "default_model": "deepseek-chat"
    },
    "mistral": {
        "api_key_env": "MISTRAL_API_KEY",
        "base_url_env": "MISTRAL_ENDPOINT",
        "default_base_url": "https://api.mistral.ai/v1",
        "default_model": "mistral-large-latest"
    },
    "ollama": {
        "base_url_env": "OLLAMA_ENDPOINT",
        "default_base_url": "http://localhost:11434",
        "default_model": "qwen2.5:7b"
    }
}

# Predefined model names for common providers
MODEL_NAMES = {
    "anthropic": ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229"],
    "openai": ["gpt-4o", "gpt-4", "gpt-3.5-turbo", "o3-mini"],
    "deepseek": ["deepseek-chat", "deepseek-reasoner"],
    "google": ["gemini-2.0-flash-exp", "gemini-2.0-flash-thinking-exp", "gemini-1.5-flash-latest", "gemini-1.5-flash-8b-latest", "gemini-2.0-flash-thinking-exp-01-21"],
    "ollama": ["qwen2.5:7b", "llama2:7b", "deepseek-r1:14b", "deepseek-r1:32b"],
    "azure_openai": ["gpt-4o", "gpt-4", "gpt-3.5-turbo"],
    "mistral": ["pixtral-large-latest", "mistral-large-latest", "mistral-small-latest", "ministral-8b-latest"]
}

def get_provider_config(provider: str):
    return PROVIDER_CONFIGS.get(provider, {})

def get_env_value(provider: str, key_type: str):
    config = get_provider_config(provider)
    env_key = config.get(f"{key_type}_env")
    if not env_key:
        return None
    return os.getenv(env_key)

def get_default_base_url(provider: str):
    config = get_provider_config(provider)
    return config.get("default_base_url")

def get_default_model(provider: str):
    config = get_provider_config(provider)
    return config.get("default_model")