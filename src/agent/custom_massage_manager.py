# -*- coding: utf-8 -*-
# @Time    : 2025/1/2
# @Author  : wenshao
# @ProjectName: browser-use-webui
# @FileName: custom_massage_manager.py

from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional, Type

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
)
from langchain_openai import ChatOpenAI

from browser_use.agent.message_manager.views import MessageHistory, MessageMetadata
from browser_use.agent.prompts import AgentMessagePrompt, SystemPrompt
from browser_use.agent.views import ActionResult, AgentOutput, AgentStepInfo
from browser_use.browser.views import BrowserState
from browser_use.agent.message_manager.service import MessageManager

from .custom_prompts import CustomAgentMessagePrompt

logger = logging.getLogger(__name__)


class CustomMassageManager(MessageManager):
    def __init__(
            self,
            llm: BaseChatModel,
            task: str,
            action_descriptions: str,
            system_prompt_class: Type[SystemPrompt],
            max_input_tokens: int = 65000,
            estimated_tokens_per_character: int = 3,
            image_tokens: int = 800,
            include_attributes: list[str] = [],
            max_error_length: int = 400,
            max_actions_per_step: int = 10,
    ):
        super().__init__(llm, task, action_descriptions, system_prompt_class, max_input_tokens,
                         estimated_tokens_per_character, image_tokens, include_attributes, max_error_length,
                         max_actions_per_step)

        # Move Task info to state_message
        self.history = MessageHistory()
        self._add_message_with_tokens(self.system_prompt)

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
                        msg = HumanMessage(content=str(r.error)[-self.max_error_length:])
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
