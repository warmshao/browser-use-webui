import json
import logging
import pdb
import traceback
from typing import Optional, Type, List, Dict, Any, Callable
from PIL import Image, ImageDraw, ImageFont
import os
import base64
import io
import platform
from browser_use.agent.prompts import SystemPrompt, AgentMessagePrompt
from browser_use.agent.service import Agent
from browser_use.agent.views import (
    ActionResult,
    ActionModel,
    AgentHistoryList,
    AgentOutput,
    AgentHistory,
)
from browser_use.browser.browser import Browser
from browser_use.browser.context import BrowserContext
from browser_use.browser.views import BrowserStateHistory
from browser_use.controller.service import Controller
from browser_use.telemetry.views import (
    AgentEndTelemetryEvent,
    AgentRunTelemetryEvent,
    AgentStepTelemetryEvent,
)
from browser_use.utils import time_execution_async
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage
)
from browser_use.agent.prompts import PlannerPrompt

from json_repair import repair_json
from src.utils.agent_state import AgentState

from .custom_message_manager import CustomMessageManager
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
            use_vision_for_planner: bool = False,
            save_conversation_path: Optional[str] = None,
            save_conversation_path_encoding: Optional[str] = 'utf-8',
            max_failures: int = 3,
            retry_delay: int = 10,
            system_prompt_class: Type[SystemPrompt] = SystemPrompt,
            agent_prompt_class: Type[AgentMessagePrompt] = AgentMessagePrompt,
            max_input_tokens: int = 128000,
            validate_output: bool = False,
            message_context: Optional[str] = None,
            generate_gif: bool | str = True,
            sensitive_data: Optional[Dict[str, str]] = None,
            available_file_paths: Optional[list[str]] = None,
            include_attributes: list[str] = [
                'title',
                'type',
                'name',
                'role',
                'tabindex',
                'aria-label',
                'placeholder',
                'value',
                'alt',
                'aria-expanded',
            ],
            max_error_length: int = 400,
            max_actions_per_step: int = 10,
            tool_call_in_content: bool = True,
            initial_actions: Optional[List[Dict[str, Dict[str, Any]]]] = None,
            # Cloud Callbacks
            register_new_step_callback: Callable[['BrowserState', 'AgentOutput', int], None] | None = None,
            register_done_callback: Callable[['AgentHistoryList'], None] | None = None,
            tool_calling_method: Optional[str] = 'auto',
            page_extraction_llm: Optional[BaseChatModel] = None,
            planner_llm: Optional[BaseChatModel] = None,
            planner_interval: int = 1,  # Run planner every N steps
    ):
        super().__init__(
            task=task,
            llm=llm,
            browser=browser,
            browser_context=browser_context,
            controller=controller,
            use_vision=use_vision,
            use_vision_for_planner=use_vision_for_planner,
            save_conversation_path=save_conversation_path,
            save_conversation_path_encoding=save_conversation_path_encoding,
            max_failures=max_failures,
            retry_delay=retry_delay,
            system_prompt_class=system_prompt_class,
            max_input_tokens=max_input_tokens,
            validate_output=validate_output,
            message_context=message_context,
            generate_gif=generate_gif,
            sensitive_data=sensitive_data,
            available_file_paths=available_file_paths,
            include_attributes=include_attributes,
            max_error_length=max_error_length,
            max_actions_per_step=max_actions_per_step,
            tool_call_in_content=tool_call_in_content,
            initial_actions=initial_actions,
            register_new_step_callback=register_new_step_callback,
            register_done_callback=register_done_callback,
            tool_calling_method=tool_calling_method,
            planner_llm=planner_llm,
            planner_interval=planner_interval
        )
        if self.model_name in ["deepseek-reasoner"] or "deepseek-r1" in self.model_name:
            # deepseek-reasoner does not support function calling
            self.use_deepseek_r1 = True
            # deepseek-reasoner only support 64000 context
            self.max_input_tokens = 64000
        else:
            self.use_deepseek_r1 = False

        # record last actions
        self._last_actions = None
        # record extract content
        self.extracted_content = ""
        # custom new info
        self.add_infos = add_infos

        self.agent_prompt_class = agent_prompt_class
        self.message_manager = CustomMessageManager(
            llm=self.llm,
            task=self.task,
            action_descriptions=self.controller.registry.get_prompt_description(),
            system_prompt_class=self.system_prompt_class,
            agent_prompt_class=agent_prompt_class,
            max_input_tokens=self.max_input_tokens,
            include_attributes=self.include_attributes,
            max_error_length=self.max_error_length,
            max_actions_per_step=self.max_actions_per_step,
            message_context=self.message_context,
            sensitive_data=self.sensitive_data
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
        logger.info(f"⏳ Task Progress: \n{response.current_state.task_progress}")
        logger.info(f"📋 Future Plans: \n{response.current_state.future_plans}")
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

        task_progress = model_output.current_state.task_progress
        if task_progress and "None" not in task_progress:
            step_info.task_progress = task_progress

        future_plans = model_output.current_state.future_plans
        if future_plans and "None" not in future_plans:
            step_info.future_plans = future_plans

        logger.info(f"🧠 All Memory: \n{step_info.memory}")

    @time_execution_async("--get_next_action")
    async def get_next_action(self, input_messages: list[BaseMessage]) -> AgentOutput:
        """Get next action from LLM based on current state"""

        ai_message = self.llm.invoke(input_messages)
        self.message_manager._add_message_with_tokens(ai_message)

        if hasattr(ai_message, "reasoning_content"):
            logger.info("🤯 Start Deep Thinking: ")
            logger.info(ai_message.reasoning_content)
            logger.info("🤯 End Deep Thinking")

        if isinstance(ai_message.content, list):
            ai_content = ai_message.content[0]
        else:
            ai_content = ai_message.content

        ai_content = ai_content.replace("```json", "").replace("```", "")
        ai_content = repair_json(ai_content)
        parsed_json = json.loads(ai_content)
        parsed: AgentOutput = self.AgentOutput(**parsed_json)

        if parsed is None:
            logger.debug(ai_message.content)
            raise ValueError('Could not parse response.')

        # Limit actions to maximum allowed per step
        parsed.action = parsed.action[: self.max_actions_per_step]
        self._log_response(parsed)
        self.n_steps += 1

        return parsed

    async def _run_planner(self) -> Optional[str]:
        """Run the planner to analyze state and suggest next steps"""
        # Skip planning if no planner_llm is set
        if not self.planner_llm:
            return None

        # Create planner message history using full message history
        planner_messages = [
            PlannerPrompt(self.action_descriptions).get_system_message(),
            *self.message_manager.get_messages()[1:],  # Use full message history except the first
        ]

        if not self.use_vision_for_planner and self.use_vision:
            last_state_message = planner_messages[-1]
            # remove image from last state message
            new_msg = ''
            if isinstance(last_state_message.content, list):
                for msg in last_state_message.content:
                    if msg['type'] == 'text':
                        new_msg += msg['text']
                    elif msg['type'] == 'image_url':
                        continue
            else:
                new_msg = last_state_message.content

            planner_messages[-1] = HumanMessage(content=new_msg)

        # Get planner output
        response = await self.planner_llm.ainvoke(planner_messages)
        plan = response.content
        last_state_message = planner_messages[-1]
        # remove image from last state message
        if isinstance(last_state_message.content, list):
            for msg in last_state_message.content:
                if msg['type'] == 'text':
                    msg['text'] += f"\nPlanning Agent outputs plans:\n {plan}\n"
        else:
            last_state_message.content += f"\nPlanning Agent outputs plans:\n {plan}\n "

        try:
            plan_json = json.loads(plan.replace("```json", "").replace("```", ""))
            logger.info(f'📋 Plans:\n{json.dumps(plan_json, indent=4)}')

            if hasattr(response, "reasoning_content"):
                logger.info("🤯 Start Planning Deep Thinking: ")
                logger.info(response.reasoning_content)
                logger.info("🤯 End Planning Deep Thinking")

        except json.JSONDecodeError:
            logger.info(f'📋 Plans:\n{plan}')
        except Exception as e:
            logger.debug(f'Error parsing planning analysis: {e}')
            logger.info(f'📋 Plans: {plan}')

    @time_execution_async("--step")
    async def step(self, step_info: Optional[CustomAgentStepInfo] = None) -> None:
        """Execute one step of the task"""
        logger.info(f"\n📍 Step {self.n_steps}")
        state = None
        model_output = None
        result: list[ActionResult] = []
        actions: list[ActionModel] = []

        try:
            state = await self.browser_context.get_state()
            self._check_if_stopped_or_paused()

            self.message_manager.add_state_message(state, self._last_actions, self._last_result, step_info,
                                                   self.use_vision)

            # Run planner at specified intervals if planner is configured
            if self.planner_llm and self.n_steps % self.planning_interval == 0:
                await self._run_planner()
            input_messages = self.message_manager.get_messages()
            self._check_if_stopped_or_paused()
            try:
                model_output = await self.get_next_action(input_messages)
                if self.register_new_step_callback:
                    self.register_new_step_callback(state, model_output, self.n_steps)
                self.update_step_info(model_output, step_info)
                self._save_conversation(input_messages, model_output)
                if self.model_name != "deepseek-reasoner":
                    # remove prev message
                    self.message_manager._remove_state_message_by_index(-1)
                self._check_if_stopped_or_paused()
            except Exception as e:
                # model call failed, remove last state message from history
                self.message_manager._remove_state_message_by_index(-1)
                raise e

            actions: list[ActionModel] = model_output.action
            result: list[ActionResult] = await self.controller.multi_act(
                actions,
                self.browser_context,
                page_extraction_llm=self.page_extraction_llm,
                sensitive_data=self.sensitive_data,
                check_break_if_paused=lambda: self._check_if_stopped_or_paused(),
                available_file_paths=self.available_file_paths,
            )
            if len(result) != len(actions):
                # I think something changes, such information should let LLM know
                for ri in range(len(result), len(actions)):
                    result.append(ActionResult(extracted_content=None,
                                               include_in_memory=True,
                                               error=f"{actions[ri].model_dump_json(exclude_unset=True)} is Failed to execute. \
                                                    Something new appeared after action {actions[len(result) - 1].model_dump_json(exclude_unset=True)}",
                                               is_done=False))
            for ret_ in result:
                if ret_.extracted_content and "Extracted page" in ret_.extracted_content:
                    # record every extracted page
                    self.extracted_content += ret_.extracted_content
            self._last_result = result
            self._last_actions = actions
            if len(result) > 0 and result[-1].is_done:
                if not self.extracted_content:
                    self.extracted_content = step_info.memory
                result[-1].extracted_content = self.extracted_content
                logger.info(f"📄 Result: {result[-1].extracted_content}")

            self.consecutive_failures = 0

        except Exception as e:
            result = await self._handle_step_error(e)
            self._last_result = result

        finally:
            actions = [a.model_dump(exclude_unset=True) for a in model_output.action] if model_output else []
            self.telemetry.capture(
                AgentStepTelemetryEvent(
                    agent_id=self.agent_id,
                    step=self.n_steps,
                    actions=actions,
                    consecutive_failures=self.consecutive_failures,
                    step_error=[r.error for r in result if r.error] if result else ['No result'],
                )
            )
            if not result:
                return

            if state:
                self._make_history_item(model_output, state, result)

    async def run(self, max_steps: int = 100) -> AgentHistoryList:
        """Execute the task with maximum number of steps"""
        try:
            self._log_agent_run()

            # Execute initial actions if provided
            if self.initial_actions:
                result = await self.controller.multi_act(
                    self.initial_actions,
                    self.browser_context,
                    check_for_new_elements=False,
                    page_extraction_llm=self.page_extraction_llm,
                    check_break_if_paused=lambda: self._check_if_stopped_or_paused(),
                    available_file_paths=self.available_file_paths,
                )
                self._last_result = result

            step_info = CustomAgentStepInfo(
                task=self.task,
                add_infos=self.add_infos,
                step_number=1,
                max_steps=max_steps,
                memory="",
                task_progress="",
                future_plans=""
            )

            for step in range(max_steps):
                if self._too_many_failures():
                    break

                # 3) Do the step
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
                if not self.extracted_content:
                    self.history.history[-1].result[-1].extracted_content = step_info.memory
                else:
                    self.history.history[-1].result[-1].extracted_content = self.extracted_content

            return self.history

        finally:
            self.telemetry.capture(
                AgentEndTelemetryEvent(
                    agent_id=self.agent_id,
                    success=self.history.is_done(),
                    steps=self.n_steps,
                    max_steps_reached=self.n_steps >= max_steps,
                    errors=self.history.errors(),
                )
            )

            if not self.injected_browser_context:
                await self.browser_context.close()

            if not self.injected_browser and self.browser:
                await self.browser.close()

            if self.generate_gif:
                output_path: str = 'agent_history.gif'
                if isinstance(self.generate_gif, str):
                    output_path = self.generate_gif

                self.create_history_gif(output_path=output_path)

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
                    if platform.system() == 'Windows':
                        # Need to specify the abs font path on Windows
                        font_name = os.path.join(os.getenv('WIN_FONT_DIR', 'C:\\Windows\\Fonts'), font_name + '.ttf')
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
