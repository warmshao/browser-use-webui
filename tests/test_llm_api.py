import os
import pdb

from dotenv import load_dotenv

load_dotenv()

import sys

sys.path.append(".")


def test_openai_model():
    from langchain_core.messages import HumanMessage
    from src.utils import utils

    llm = utils.get_llm_model(
        provider="openai",
        model_name="gpt-4o",
        temperature=0.8,
        base_url=os.getenv("OPENAI_ENDPOINT", ""),
        api_key=os.getenv("OPENAI_API_KEY", "")
    )
    image_path = "assets/examples/test.png"
    image_data = utils.encode_image(image_path)
    message = HumanMessage(
        content=[
            {"type": "text", "text": "describe this image"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
            },
        ]
    )
    ai_msg = llm.invoke([message])
    print(ai_msg.content)


def test_gemini_model():
    # you need to enable your api key first: https://ai.google.dev/palm_docs/oauth_quickstart
    from langchain_core.messages import HumanMessage
    from src.utils import utils

    llm = utils.get_llm_model(
        provider="gemini",
        model_name="gemini-2.0-flash-exp",
        temperature=0.8,
        api_key=os.getenv("GOOGLE_API_KEY", "")
    )

    image_path = "assets/examples/test.png"
    image_data = utils.encode_image(image_path)
    message = HumanMessage(
        content=[
            {"type": "text", "text": "describe this image"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
            },
        ]
    )
    ai_msg = llm.invoke([message])
    print(ai_msg.content)


def test_azure_openai_model():
    from langchain_core.messages import HumanMessage
    from src.utils import utils

    llm = utils.get_llm_model(
        provider="azure_openai",
        model_name="gpt-4o",
        temperature=0.8,
        base_url=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        api_key=os.getenv("AZURE_OPENAI_API_KEY", "")
    )
    image_path = "assets/examples/test.png"
    image_data = utils.encode_image(image_path)
    message = HumanMessage(
        content=[
            {"type": "text", "text": "describe this image"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
            },
        ]
    )
    ai_msg = llm.invoke([message])
    print(ai_msg.content)


def test_deepseek_model():
    from langchain_core.messages import HumanMessage
    from src.utils import utils

    llm = utils.get_llm_model(
        provider="deepseek",
        model_name="deepseek-chat",
        temperature=0.8,
        base_url=os.getenv("DEEPSEEK_ENDPOINT", ""),
        api_key=os.getenv("DEEPSEEK_API_KEY", "")
    )
    message = HumanMessage(
        content=[
            {"type": "text", "text": "who are you?"}
        ]
    )
    ai_msg = llm.invoke([message])
    print(ai_msg.content)

def test_deepseek_r1_model():
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    from src.utils import utils

    llm = utils.get_llm_model(
        provider="deepseek",
        model_name="deepseek-reasoner",
        temperature=0.8,
        base_url=os.getenv("DEEPSEEK_ENDPOINT", ""),
        api_key=os.getenv("DEEPSEEK_API_KEY", "")
    )
    messages = []
    sys_message = SystemMessage(
        content=[{"type": "text", "text": "you are a helpful AI assistant"}]
    )
    messages.append(sys_message)
    user_message = HumanMessage(
        content=[
            {"type": "text", "text": "9.11 and 9.8, which is greater?"}
        ]
    )
    messages.append(user_message)
    ai_msg = llm.invoke(messages)
    print(ai_msg.reasoning_content)
    print(ai_msg.content)
    print(llm.model_name)
    pdb.set_trace()

def test_ollama_model():
    from langchain_ollama import ChatOllama

    llm = ChatOllama(model="qwen2.5:7b")
    ai_msg = llm.invoke("Sing a ballad of LangChain.")
    print(ai_msg.content)


if __name__ == '__main__':
    # test_openai_model()
    # test_gemini_model()
    # test_azure_openai_model()
    # test_deepseek_model()
    # test_ollama_model()
    test_deepseek_r1_model()
