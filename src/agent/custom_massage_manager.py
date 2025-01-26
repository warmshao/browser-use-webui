from __future__ import annotations

import logging
from typing import List, Optional, Type

from browser_use.agent.message_manager.service import MessageManager
from browser_use.agent.message_manager.views import MessageHistory
from browser_use.agent.prompts import SystemPrompt
from browser_use.agent.views import ActionResult, AgentStepInfo
from browser_use.browser.views import BrowserState
from langchain_core.language_models import BaseChatModel
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
	AIMessage,
	BaseMessage,
	HumanMessage,
)
from langchain_openai import ChatOpenAI
from ..utils.llm import DeepSeekR1ChatOpenAI
from .custom_prompts import CustomAgentMessagePrompt

logger = logging.getLogger(__name__)


class CustomMassageManager(MessageManager):
    def __init__(
            self,
            llm: BaseChatModel,
            task: str,
            action_descriptions: str,
            system_prompt_class: Type[SystemPrompt],
            max_input_tokens: int = 128000,
            estimated_tokens_per_character: int = 3,
            image_tokens: int = 800,
            include_attributes: list[str] = [],
            max_error_length: int = 400,
            max_actions_per_step: int = 10,
            tool_call_in_content: bool = False,
            use_function_calling: bool = True
    ):
        super().__init__(
            llm=llm,
            task=task,
            action_descriptions=action_descriptions,
            system_prompt_class=system_prompt_class,
            max_input_tokens=max_input_tokens,
            estimated_tokens_per_character=estimated_tokens_per_character,
            image_tokens=image_tokens,
            include_attributes=include_attributes,
            max_error_length=max_error_length,
            max_actions_per_step=max_actions_per_step,
            tool_call_in_content=tool_call_in_content,
        )
        self.use_function_calling = use_function_calling
        # Custom: Move Task info to state_message
        self.history = MessageHistory()
        self._add_message_with_tokens(self.system_prompt)
        
        if self.use_function_calling:
            tool_calls = [
                {
                    'name': 'CustomAgentOutput',
                    'args': {
                        'current_state': {
                            'prev_action_evaluation': 'Unknown - No previous actions to evaluate.',
                            'important_contents': '',
                            'completed_contents': '',
                            'thought': 'Now Google is open. Need to type OpenAI to search.',
                            'summary': 'Type OpenAI to search.',
                        },
                        'action': [],
                    },
                    'id': '',
                    'type': 'tool_call',
                }
            ]
            if self.tool_call_in_content:
                # openai throws error if tool_calls are not responded -> move to content
                example_tool_call = AIMessage(
                    content=f'{tool_calls}',
                    tool_calls=[],
                )
            else:
                example_tool_call = AIMessage(
                    content=f'',
                    tool_calls=tool_calls,
                )

            self._add_message_with_tokens(example_tool_call)

    def cut_messages(self):
        """Get current message list, potentially trimmed to max tokens"""
        diff = self.history.total_tokens - self.max_input_tokens
        while diff > 0 and len(self.history.messages) > 1:
            self.history.remove_message(1) # alway remove the oldest one
            diff = self.history.total_tokens - self.max_input_tokens
        
    def add_state_message(
            self,
            state: BrowserState,
            result: Optional[List[ActionResult]] = None,
            step_info: Optional[AgentStepInfo] = None,
    ) -> None:
        """Add browser state as human message"""
        # otherwise add state message and result to next message (which will not stay in memory)
        state_message = CustomAgentMessagePrompt(
            state,
            result,
            include_attributes=self.include_attributes,
            max_error_length=self.max_error_length,
            step_info=step_info,
        ).get_user_message()
        self._add_message_with_tokens(state_message)
    
    def _count_text_tokens(self, text: str) -> int:
        if isinstance(self.llm, (ChatOpenAI, ChatAnthropic, DeepSeekR1ChatOpenAI)):
            try:
                tokens = self.llm.get_num_tokens(text)
            except Exception:
                tokens = (
					len(text) // self.ESTIMATED_TOKENS_PER_CHARACTER
				)  # Rough estimate if no tokenizer available
        else:
            tokens = (
				len(text) // self.ESTIMATED_TOKENS_PER_CHARACTER
			)  # Rough estimate if no tokenizer available
        return tokens
