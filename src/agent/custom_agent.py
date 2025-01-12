# -*- coding: utf-8 -*-
# @Time    : 2025/1/2
# @Author  : wenshao
# @ProjectName: browser-use-webui
# @FileName: custom_agent.py

import json
import logging
import os
import pdb
import traceback
import src.utils.utils as utils
from datetime import datetime
from typing import Optional, Type
from PIL import Image, ImageDraw, ImageFont
import os
import base64
import io

from browser_use.agent.prompts import SystemPrompt
from browser_use.agent.service import Agent
from browser_use.agent.views import (
    ActionResult,
    AgentHistoryList,
    AgentOutput,
)
from browser_use.browser.browser import Browser
from browser_use.browser.context import BrowserContext
from browser_use.controller.service import Controller
from browser_use.telemetry.views import (
    AgentEndTelemetryEvent,
    AgentRunTelemetryEvent,
    AgentStepErrorTelemetryEvent,
)
from browser_use.utils import time_execution_async
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
)

from .custom_massage_manager import CustomMassageManager
from .custom_views import CustomAgentOutput, CustomAgentStepInfo

logger = logging.getLogger(__name__)


class CustomAgent(Agent):
    def __init__(
            self,
            task: str,
            llm: BaseChatModel,
            add_infos: str = "",
            browser: Browser | None = None,
            browser_context: BrowserContext | None = None,
            controller: Controller = Controller(),
            use_vision: bool = True,
            save_conversation_path: Optional[str] = None,
            max_failures: int = 5,
            retry_delay: int = 10,
            system_prompt_class: Type[SystemPrompt] = SystemPrompt,
            max_input_tokens: int = 128000,
            validate_output: bool = False,
            include_attributes: list[str] = [
                "title",
                "type",
                "name",
                "role",
                "tabindex",
                "aria-label",
                "placeholder",
                "value",
                "alt",
                "aria-expanded",
            ],
            max_error_length: int = 400,
            max_actions_per_step: int = 10,
            tool_call_in_content: bool = True,
    ):
        super().__init__(
            task=task,
            llm=llm,
            browser=browser,
            browser_context=browser_context,
            controller=controller,
            use_vision=use_vision,
            save_conversation_path=save_conversation_path,
            max_failures=max_failures,
            retry_delay=retry_delay,
            system_prompt_class=system_prompt_class,
            max_input_tokens=max_input_tokens,
            validate_output=validate_output,
            include_attributes=include_attributes,
            max_error_length=max_error_length,
            max_actions_per_step=max_actions_per_step,
            tool_call_in_content=tool_call_in_content,
        )
        self.add_infos = add_infos
        self.message_manager = CustomMassageManager(
            llm=self.llm,
            task=self.task,
            action_descriptions=self.controller.registry.get_prompt_description(),
            system_prompt_class=self.system_prompt_class,
            max_input_tokens=self.max_input_tokens,
            include_attributes=self.include_attributes,
            max_error_length=self.max_error_length,
            max_actions_per_step=self.max_actions_per_step,
            tool_call_in_content=tool_call_in_content,
        )

    def _setup_action_models(self) -> None:
        """Setup dynamic action models from controller's registry"""
        # Get the dynamic action model from controller's registry
        self.ActionModel = self.controller.registry.create_action_model()
        # Create output model with the dynamic actions
        self.AgentOutput = CustomAgentOutput.type_with_custom_actions(self.ActionModel)

    def _log_response(self, response: CustomAgentOutput) -> None:
        """Log the model's response"""
        if "Success" in response.current_state.prev_action_evaluation:
            emoji = "✅"
        elif "Failed" in response.current_state.prev_action_evaluation:
            emoji = "❌"
        else:
            emoji = "🤷"

        logger.info(f"{emoji} Eval: {response.current_state.prev_action_evaluation}")
        logger.info(f"🧠 New Memory: {response.current_state.important_contents}")
        logger.info(f"⏳ Task Progress: {response.current_state.completed_contents}")
        logger.info(f"🤔 Thought: {response.current_state.thought}")
        logger.info(f"🎯 Summary: {response.current_state.summary}")
        for i, action in enumerate(response.action):
            logger.info(
                f"🛠️  Action {i + 1}/{len(response.action)}: {action.model_dump_json(exclude_unset=True)}"
            )

    def update_step_info(
            self, model_output: CustomAgentOutput, step_info: CustomAgentStepInfo = None
    ):
        """
        update step info
        """
        if step_info is None:
            return

        step_info.step_number += 1
        important_contents = model_output.current_state.important_contents
        if (
                important_contents
                and "None" not in important_contents
                and important_contents not in step_info.memory
        ):
            step_info.memory += important_contents + "\n"

        completed_contents = model_output.current_state.completed_contents
        if completed_contents and "None" not in completed_contents:
            step_info.task_progress = completed_contents

    @time_execution_async("--get_next_action")
    async def get_next_action(self, input_messages: list[BaseMessage]) -> AgentOutput:
        """Get next action from LLM based on current state"""
        try:
            structured_llm = self.llm.with_structured_output(self.AgentOutput, include_raw=True)
            response: dict[str, Any] = await structured_llm.ainvoke(input_messages)  # type: ignore

            parsed: AgentOutput = response['parsed']
            # cut the number of actions to max_actions_per_step
            parsed.action = parsed.action[: self.max_actions_per_step]
            self._log_response(parsed)
            self.n_steps += 1

            return parsed
        except Exception as e:
            # If something goes wrong, try to invoke the LLM again without structured output,
            # and Manually parse the response. Temporarily solution for DeepSeek
            ret = self.llm.invoke(input_messages)
            if isinstance(ret.content, list):
                parsed_json = json.loads(ret.content[0].replace("```json", "").replace("```", ""))
            else:
                parsed_json = json.loads(ret.content.replace("```json", "").replace("```", ""))
            parsed: AgentOutput = self.AgentOutput(**parsed_json)
            if parsed is None:
                raise ValueError(f'Could not parse response.')

            # cut the number of actions to max_actions_per_step
            parsed.action = parsed.action[: self.max_actions_per_step]
            self._log_response(parsed)
            self.n_steps += 1

            return parsed

    @time_execution_async("--step")
    async def step(self, step_info: Optional[CustomAgentStepInfo] = None) -> None:
        """Execute one step of the task"""
        logger.info(f"\n📍 Step {self.n_steps}")
        state = None
        model_output = None
        result: list[ActionResult] = []

        try:
            state = await self.browser_context.get_state(use_vision=self.use_vision)
            self.message_manager.add_state_message(state, self._last_result, step_info)
            input_messages = self.message_manager.get_messages()
            model_output = await self.get_next_action(input_messages)
            self.update_step_info(model_output, step_info)
            logger.info(f"🧠 All Memory: {step_info.memory}")
            self._save_conversation(input_messages, model_output)
            self.message_manager._remove_last_state_message()  # we dont want the whole state in the chat history
            self.message_manager.add_model_output(model_output)

            result: list[ActionResult] = await self.controller.multi_act(
                model_output.action, self.browser_context
            )
            self._last_result = result

            if len(result) > 0 and result[-1].is_done:
                logger.info(f"📄 Result: {result[-1].extracted_content}")

            # Persist the result
            data_dir = os.getenv("DATA_DIRECTORY", "")
            try:
                if data_dir != "":
                    task_name = utils.sanitize_directory_name(self.task)
                    os.makedirs(data_dir, exist_ok=True)
                    task_dir = os.path.join(data_dir, task_name, )
                    os.makedirs(task_dir, exist_ok=True)
                    file_name = f"step_{step_info.step_number}.txt"
                    file_path = os.path.join(task_dir, file_name, )

                    with open(file_path, "a", encoding='utf-8') as file:
                        file.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Content of the step #{step_info.step_number} is:\n{result[-1].extracted_content}.\n")
                        file.write("--------------------------------------------------------\n")
                    logger.info(f"📄 Saved in: {file_path}")

            except Exception as E:
                pass

            self.consecutive_failures = 0

        except Exception as e:
            result = self._handle_step_error(e)
            self._last_result = result

        finally:
            if not result:
                return
            for r in result:
                if r.error:
                    self.telemetry.capture(
                        AgentStepErrorTelemetryEvent(
                            agent_id=self.agent_id,
                            error=r.error,
                        )
                    )
            if state:
                self._make_history_item(model_output, state, result)
    def create_history_gif(
            self,
            output_path: str = 'agent_history.gif',
            duration: int = 3000,
            show_goals: bool = True,
            show_task: bool = True,
            show_logo: bool = False,
            font_size: int = 40,
            title_font_size: int = 56,
            goal_font_size: int = 44,
            margin: int = 40,
            line_spacing: float = 1.5,
    ) -> None:
        """Create a GIF from the agent's history with overlaid task and goal text."""
        if not self.history.history:
            logger.warning('No history to create GIF from')
            return

        images = []
        # if history is empty or first screenshot is None, we can't create a gif
        if not self.history.history or not self.history.history[0].state.screenshot:
            logger.warning('No history or first screenshot to create GIF from')
            return

        # Try to load nicer fonts
        try:
            # Try different font options in order of preference
            font_options = ['Helvetica', 'Arial', 'DejaVuSans', 'Verdana']
            font_loaded = False

            for font_name in font_options:
                try:
                    import platform
                    if platform.system() == "Windows":
                        # Need to specify the abs font path on Windows
                        font_name = os.path.join(os.getenv("WIN_FONT_DIR", "C:\\Windows\\Fonts"), font_name + ".ttf")
                    regular_font = ImageFont.truetype(font_name, font_size)
                    title_font = ImageFont.truetype(font_name, title_font_size)
                    goal_font = ImageFont.truetype(font_name, goal_font_size)
                    font_loaded = True
                    break
                except OSError:
                    continue

            if not font_loaded:
                raise OSError('No preferred fonts found')

        except OSError:
            regular_font = ImageFont.load_default()
            title_font = ImageFont.load_default()

            goal_font = regular_font

        # Load logo if requested
        logo = None
        if show_logo:
            try:
                logo = Image.open('./static/browser-use.png')
                # Resize logo to be small (e.g., 40px height)
                logo_height = 150
                aspect_ratio = logo.width / logo.height
                logo_width = int(logo_height * aspect_ratio)
                logo = logo.resize((logo_width, logo_height), Image.Resampling.LANCZOS)
            except Exception as e:
                logger.warning(f'Could not load logo: {e}')

        # Create task frame if requested
        if show_task and self.task:
            task_frame = self._create_task_frame(
                self.task,
                self.history.history[0].state.screenshot,
                title_font,
                regular_font,
                logo,
                line_spacing,
            )
            images.append(task_frame)

        # Process each history item
        for i, item in enumerate(self.history.history, 1):
            if not item.state.screenshot:
                continue

            # Convert base64 screenshot to PIL Image
            img_data = base64.b64decode(item.state.screenshot)
            image = Image.open(io.BytesIO(img_data))

            if show_goals and item.model_output:
                image = self._add_overlay_to_image(
                    image=image,
                    step_number=i,
                    goal_text=item.model_output.current_state.thought,
                    regular_font=regular_font,
                    title_font=title_font,
                    margin=margin,
                    logo=logo,
                )

            images.append(image)

        if images:
            # Save the GIF
            images[0].save(
                output_path,
                save_all=True,
                append_images=images[1:],
                duration=duration,
                loop=0,
                optimize=False,
            )
            logger.info(f'Created GIF at {output_path}')
        else:
            logger.warning('No images found in history to create GIF')

    async def run(self, max_steps: int = 100) -> AgentHistoryList:
        """Execute the task with maximum number of steps"""
        try:
            logger.info(f"🚀 Starting task: {self.task}")

            self.telemetry.capture(
                AgentRunTelemetryEvent(
                    agent_id=self.agent_id,
                    task=self.task,
                )
            )

            step_info = CustomAgentStepInfo(
                task=self.task,
                add_infos=self.add_infos,
                step_number=1,
                max_steps=max_steps,
                memory="",
                task_progress="",
            )

            for step in range(max_steps):
                if self._too_many_failures():
                    break

                await self.step(step_info)

                if self.history.is_done():
                    if (
                            self.validate_output and step < max_steps - 1
                    ):  # if last step, we dont need to validate
                        if not await self._validate_output():
                            continue

                    logger.info("✅ Task completed successfully")
                    break
            else:
                logger.info("❌ Failed to complete task in maximum steps")

            return self.history

        finally:
            self.telemetry.capture(
                AgentEndTelemetryEvent(
                    agent_id=self.agent_id,
                    task=self.task,
                    success=self.history.is_done(),
                    steps=len(self.history.history),
                )
            )
            if not self.injected_browser_context:
                await self.browser_context.close()

            if not self.injected_browser and self.browser:
                await self.browser.close()

            if self.generate_gif:
                self.create_history_gif()
