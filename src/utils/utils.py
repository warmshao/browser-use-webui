# -*- coding: utf-8 -*-
# @Time    : 2025/1/1
# @Author  : wenshao
# @Email   : wenshaoguo1026@gmail.com
# @Project : browser-use-webui
# @FileName: utils.py

import base64
import os

from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage


def get_llm_model(provider: str, **kwargs):
    """
    获取LLM 模型
    :param provider: 模型类型
    :param kwargs:
    :return:
    """
    if provider == 'anthropic':
        if not kwargs.get("base_url", ""):
            base_url = "https://api.anthropic.com"
        else:
            base_url = kwargs.get("base_url")

        if not kwargs.get("api_key", ""):
            api_key = os.getenv("ANTHROPIC_API_KEY", "")
        else:
            api_key = kwargs.get("api_key")

        return ChatAnthropic(
            model_name=kwargs.get("model_name", 'claude-3-5-sonnet-20240620'),
            temperature=kwargs.get("temperature", 0.0),
            base_url=base_url,
            api_key=api_key
        )
    elif provider == 'openai':
        if not kwargs.get("base_url", ""):
            base_url = os.getenv("OPENAI_ENDPOINT", "https://api.openai.com/v1")
        else:
            base_url = kwargs.get("base_url")

        if not kwargs.get("api_key", ""):
            api_key = os.getenv("OPENAI_API_KEY", "")
        else:
            api_key = kwargs.get("api_key")

        return ChatOpenAI(
            model=kwargs.get("model_name", 'gpt-4o'),
            temperature=kwargs.get("temperature", 0.0),
            base_url=base_url,
            api_key=api_key
        )
    elif provider == 'deepseek':
        if not kwargs.get("base_url", ""):
            base_url = os.getenv("DEEPSEEK_ENDPOINT", "")
        else:
            base_url = kwargs.get("base_url")

        if not kwargs.get("api_key", ""):
            api_key = os.getenv("DEEPSEEK_API_KEY", "")
        else:
            api_key = kwargs.get("api_key")

        model_name = kwargs.get("model_name", 'deepseek-chat')
        
        if model_name == 'deepseek-reasoner':
            # Custom handling for deepseek-reasoner
            class DeepseekReasonerChat(ChatOpenAI):
                def invoke(self, messages):
                    # Ensure messages alternate between user and assistant
                    interleaved_messages = []
                    last_role = None
                    
                    for msg in messages:
                        current_role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
                        
                        # If same role as last message, combine them
                        if current_role == last_role and interleaved_messages:
                            if isinstance(msg.content, dict):
                                text = msg.content.get('text', '')
                                if 'reasoning_content' in msg.content:
                                    del msg.content['reasoning_content']
                            else:
                                text = str(msg.content)
                            interleaved_messages[-1].content += "\n" + text
                        else:
                            # Clean message content if needed
                            if isinstance(msg.content, dict):
                                if 'reasoning_content' in msg.content:
                                    del msg.content['reasoning_content']
                                msg.content = msg.content.get('text', '')
                            interleaved_messages.append(msg)
                            last_role = current_role

                    response = super().invoke(interleaved_messages)
                    return response

            return DeepseekReasonerChat(
                model=model_name,
                temperature=kwargs.get("temperature", 0.0),
                base_url=base_url,
                api_key=api_key
            )
        else:
            # Default handling for other Deepseek models
            return ChatOpenAI(
                model=model_name,
                temperature=kwargs.get("temperature", 0.0),
                base_url=base_url,
                api_key=api_key
            )
    elif provider == 'gemini':
        if not kwargs.get("api_key", ""):
            api_key = os.getenv("GOOGLE_API_KEY", "")
        else:
            api_key = kwargs.get("api_key")
        return ChatGoogleGenerativeAI(
            model=kwargs.get("model_name", 'gemini-2.0-flash-exp'),
            temperature=kwargs.get("temperature", 0.0),
            google_api_key=api_key,
        )
    elif provider == 'ollama':
        return ChatOllama(
            model=kwargs.get("model_name", 'qwen2.5:7b'),
            temperature=kwargs.get("temperature", 0.0),
        )
    elif provider == "azure_openai":
        if not kwargs.get("base_url", ""):
            base_url = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        else:
            base_url = kwargs.get("base_url")
        if not kwargs.get("api_key", ""):
            api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        else:
            api_key = kwargs.get("api_key")
        return AzureChatOpenAI(
            model=kwargs.get("model_name", 'gpt-4o'),
            temperature=kwargs.get("temperature", 0.0),
            api_version="2024-05-01-preview",
            azure_endpoint=base_url,
            api_key=api_key
        )
    else:
        raise ValueError(f'Unsupported provider: {provider}')


def encode_image(img_path):
    if not img_path:
        return None
    with open(img_path, "rb") as fin:
        image_data = base64.b64encode(fin.read()).decode("utf-8")
    return image_data
