from openai import OpenAI
import logging
import traceback
from langchain_openai import ChatOpenAI
from langchain_core.globals import get_llm_cache
from langchain_core.language_models.base import (
    BaseLanguageModel,
    LangSmithParams,
    LanguageModelInput,
)
from langchain_core.load import dumpd, dumps
from langchain_core.messages import (
    AIMessage,
    SystemMessage,
    AnyMessage,
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    convert_to_messages,
    message_chunk_to_message,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
    LLMResult,
    RunInfo,
)
from langchain_ollama import ChatOllama
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Optional,
    Union,
    cast,
)

logger = logging.getLogger(__name__)

class ModelResponseProcessor:
    """Utility class for extracting and processing model responses."""
    
    @staticmethod
    def extract_reasoning_content(content: str) -> tuple[str, str]:
        """Extract reasoning content from various formats."""
        reasoning_content = ""
        processed_content = content
        
        # Try different formats
        if "<think>" in content and "</think>" in content:
            # DeepSeek format with <think> tags
            parts = content.split("</think>", 1)
            reasoning_content = parts[0].replace("<think>", "").strip()
            processed_content = parts[1].strip() if len(parts) > 1 else content
        elif "Reasoning:" in content and "Action:" in content:
            # Format with explicit Reasoning/Action sections
            parts = content.split("Action:", 1)
            reasoning_content = parts[0].replace("Reasoning:", "").strip()
            processed_content = parts[1].strip() if len(parts) > 1 else content
            
        return reasoning_content, processed_content
    
    @staticmethod
    def extract_json_content(content: str) -> str:
        """Extract JSON content from various formats."""
        processed_content = content
        
        # Try JSON code blocks
        if "```json" in content and "```" in content:
            try:
                json_parts = content.split("```json", 1)
                if len(json_parts) > 1:
                    code_parts = json_parts[1].split("```", 1)
                    if code_parts:
                        processed_content = code_parts[0].strip()
            except Exception:
                pass
                
        # Try JSON Response marker
        elif "**JSON Response:**" in content:
            try:
                json_parts = content.split("**JSON Response:**", 1)
                if len(json_parts) > 1:
                    processed_content = json_parts[1].strip()
            except Exception:
                pass
                
        return processed_content
    
    @staticmethod
    def process_response(response: AIMessage) -> AIMessage:
        """Process a response to extract reasoning and content."""
        try:
            if not hasattr(response, "content") or not response.content:
                return AIMessage(content="", reasoning_content="")
                
            content = response.content
            
            # Extract reasoning content
            reasoning_content, processed_content = ModelResponseProcessor.extract_reasoning_content(content)
            
            # Extract JSON content if present
            processed_content = ModelResponseProcessor.extract_json_content(processed_content)
            
            return AIMessage(content=processed_content, reasoning_content=reasoning_content)
            
        except Exception as e:
            logger.error(f"Error processing response: {e}")
            # Return original message if processing fails
            return response


class EnhancedChatOpenAI(ChatOpenAI):
    """Enhanced ChatOpenAI that handles reasoning extraction."""
    
    def __init__(self, *args, **kwargs):
        self.extract_reasoning = kwargs.pop("extract_reasoning", False)
        super().__init__(*args, **kwargs)
        
    async def ainvoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AIMessage:
        try:
            response = await super().ainvoke(input=input, config=config, stop=stop, **kwargs)
            
            if self.extract_reasoning:
                return ModelResponseProcessor.process_response(response)
            return response
            
        except Exception as e:
            logger.error(f"Error in EnhancedChatOpenAI.ainvoke: {e}")
            # Return a minimal AIMessage
            return AIMessage(content=f"Error: {str(e)}")
            
    def invoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AIMessage:
        try:
            response = super().invoke(input=input, config=config, stop=stop, **kwargs)
            
            if self.extract_reasoning:
                return ModelResponseProcessor.process_response(response)
            return response
            
        except Exception as e:
            logger.error(f"Error in EnhancedChatOpenAI.invoke: {e}")
            # Return a minimal AIMessage
            return AIMessage(content=f"Error: {str(e)}")


class EnhancedChatOllama(ChatOllama):
    """Enhanced ChatOllama that handles reasoning extraction."""
    
    extract_reasoning: bool = True
    
    def __init__(self, *args, **kwargs):
        if "extract_reasoning" in kwargs:
            self.extract_reasoning = kwargs.pop("extract_reasoning")
        super().__init__(*args, **kwargs)
                
    async def ainvoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AIMessage:
        try:
            response = await super().ainvoke(input=input, config=config, stop=stop, **kwargs)
            
            if self.extract_reasoning:
                return ModelResponseProcessor.process_response(response)
            return response
            
        except Exception as e:
            logger.error(f"Error in EnhancedChatOllama.ainvoke: {e}\n{traceback.format_exc()}")
            # Return a minimal AIMessage
            return AIMessage(content=f"Error: {str(e)}")
            
    def invoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AIMessage:
        try:
            # Try special API for compatible models
            if hasattr(self, "client") and hasattr(self.client, "chat") and \
               any(name in self.model_name for name in ["deepseek-r1", "command-r"]):
                try:
                    message_history = []
                    for input_ in input:
                        if isinstance(input_, SystemMessage):
                            message_history.append({"role": "system", "content": input_.content})
                        elif isinstance(input_, AIMessage):
                            message_history.append({"role": "assistant", "content": input_.content})
                        else:
                            message_history.append({"role": "user", "content": input_.content})
                    
                    api_response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=message_history
                    )
                    
                    content = getattr(api_response.choices[0].message, "content", "")
                    reasoning_content = getattr(api_response.choices[0].message, "reasoning_content", "")
                    
                    if content and reasoning_content:
                        return AIMessage(content=content, reasoning_content=reasoning_content)
                except Exception as api_err:
                    logger.warning(f"Special API approach failed, falling back: {api_err}")
            
            # Standard approach
            response = super().invoke(input=input, config=config, stop=stop, **kwargs)
            
            if self.extract_reasoning:
                return ModelResponseProcessor.process_response(response)
            return response
            
        except Exception as e:
            logger.error(f"Error in EnhancedChatOllama.invoke: {e}\n{traceback.format_exc()}")
            # Return a minimal AIMessage
            return AIMessage(content=f"Error: {str(e)}")


class DeepSeekR1ChatOpenAI(EnhancedChatOpenAI):
    """Specialized class for DeepSeek models via OpenAI compatible API."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(extract_reasoning=True, *args, **kwargs)
        self.client = OpenAI(
            base_url=kwargs.get("base_url"),
            api_key=kwargs.get("api_key")
        )
        
    async def ainvoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AIMessage:
        try:
            message_history = []
            for input_ in input:
                if isinstance(input_, SystemMessage):
                    message_history.append({"role": "system", "content": input_.content})
                elif isinstance(input_, AIMessage):
                    message_history.append({"role": "assistant", "content": input_.content})
                else:
                    message_history.append({"role": "user", "content": input_.content})
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=message_history
            )

            reasoning_content = getattr(response.choices[0].message, "reasoning_content", "")
            content = getattr(response.choices[0].message, "content", "")
            
            return AIMessage(content=content, reasoning_content=reasoning_content)
        except Exception as e:
            logger.error(f"Error in DeepSeekR1ChatOpenAI.ainvoke: {e}\n{traceback.format_exc()}")
            return AIMessage(content=f"Error processing DeepSeek model: {str(e)}")