import os

PROVIDER_DISPLAY_NAMES = {
    "openai": "OpenAI",
    "azure_openai": "Azure OpenAI",
    "anthropic": "Anthropic",
    "deepseek": "DeepSeek",
    "google": "Google",
    "mistral": "Mistral",
    "alibaba": "Alibaba",
    "moonshot": "MoonShot"
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
        "api_version_env": "AZURE_OPENAI_API_VERSION",
        "default_api_version": "2025-01-01-preview",
        "default_model": "gpt-4o"
    },
    "anthropic": {
        "api_key_env": "ANTHROPIC_API_KEY",
        "base_url_env": "ANTHROPIC_ENDPOINT",
        "default_base_url": "https://api.anthropic.com",
        "default_model": "claude-3-5-sonnet-20241022"
    },
    "google": {
        "api_key_env": "GOOGLE_API_KEY",
        "default_model": "gemini-2.0-flash-exp"
    },
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url_env": "DEEPSEEK_ENDPOINT",
        "default_base_url": "https://api.deepseek.com",
        "default_model": "deepseek-chat"
    },
    "mistral": {
        "api_key_env": "MISTRAL_API_KEY",
        "base_url_env": "MISTRAL_ENDPOINT",
        "default_base_url": "https://api.mistral.ai/v1",
        "default_model": "mistral-large-latest"
    },
    "alibaba": {
        "api_key_env": "ALIBABA_API_KEY",
        "base_url_env": "ALIBABA_ENDPOINT",
        "default_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "default_model": "qwen-plus"
    },
    "moonshot": {
        "api_key_env": "MOONSHOT_API_KEY",
        "base_url_env": "MOONSHOT_ENDPOINT",
        "default_base_url": "https://api.moonshot.cn/v1",
        "default_model": "moonshot-v1-32k-vision-preview"
    },
    "ollama": {
        "base_url_env": "OLLAMA_ENDPOINT",
        "default_base_url": "http://localhost:11434",
        "default_model": "qwen2.5:7b"
    }
}

# Predefined model names for common providers
MODEL_NAMES = {
    "anthropic": ["claude-3-5-sonnet-20241022", "claude-3-5-sonnet-20240620", "claude-3-opus-20240229"],
    "openai": ["gpt-4o", "gpt-4", "gpt-3.5-turbo", "o3-mini"],
    "deepseek": ["deepseek-chat", "deepseek-reasoner"],
    "google": ["gemini-2.0-flash", "gemini-2.0-flash-thinking-exp", "gemini-1.5-flash-latest", "gemini-1.5-flash-8b-latest", "gemini-2.0-flash-thinking-exp-01-21", "gemini-2.0-pro-exp-02-05"],
    "ollama": ["qwen2.5:7b", "qwen2.5:14b", "qwen2.5:32b", "qwen2.5-coder:14b", "qwen2.5-coder:32b", "llama2:7b", "deepseek-r1:14b", "deepseek-r1:32b"],
    "azure_openai": ["gpt-4o", "gpt-4", "gpt-3.5-turbo"],
    "mistral": ["pixtral-large-latest", "mistral-large-latest", "mistral-small-latest", "ministral-8b-latest"],
    "alibaba": ["qwen-plus", "qwen-max", "qwen-turbo", "qwen-long"],
    "moonshot": ["moonshot-v1-32k-vision-preview", "moonshot-v1-8k-vision-preview"],
}

def get_provider_config(provider: str):
    return PROVIDER_CONFIGS.get(provider, {})

def get_config_value(provider: str, key: str, **kwargs):
    config = get_provider_config(provider)
    
    if key in kwargs and kwargs[key]:
        return kwargs[key]
    
    env_key = config.get(f"{key}_env")
    if env_key:
        env_value = os.getenv(env_key)
        if env_value:
            return env_value
    
    return config.get(f"default_{key}")
