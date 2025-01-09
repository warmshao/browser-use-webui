# -*- coding: utf-8 -*-
# @Time    : 2025/1/2
# @Author  : wenshao
# @ProjectName: browser-use-webui
# @FileName: custom_massage_manager.py

from __future__ import annotations

import logging
from typing import List, Optional, Type

from browser_use.agent.message_manager.service import MessageManager
from browser_use.agent.message_manager.views import MessageHistory
from browser_use.agent.prompts import SystemPrompt
from browser_use.agent.views import ActionResult, AgentStepInfo
from browser_use.browser.views import BrowserState
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    HumanMessage,
    AIMessage
)

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

        # Custom: Move Task info to state_message
        self.history = MessageHistory()
        self._add_message_with_tokens(self.system_prompt)
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

    def add_state_message(
            self,
            state: BrowserState,
            result: Optional[List[ActionResult]] = None,
            step_info: Optional[AgentStepInfo] = None,
    ) -> None:
        """Add browser state as human message"""

        # if keep in memory, add to directly to history and add state without result
        if result:
            for r in result:
                if r.include_in_memory:
                    if r.extracted_content:
                        msg = HumanMessage(content=str(r.extracted_content))
                        self._add_message_with_tokens(msg)
                    if r.error:
                        msg = HumanMessage(
                            content=str(r.error)[-self.max_error_length:]
                        )
                        self._add_message_with_tokens(msg)
                    result = None  # if result in history, we dont want to add it again

        # otherwise add state message and result to next message (which will not stay in memory)
        state_message = CustomAgentMessagePrompt(
            state,
            result,
            include_attributes=self.include_attributes,
            max_error_length=self.max_error_length,
            step_info=step_info,
        ).get_user_message()
        self._add_message_with_tokens(state_message)
