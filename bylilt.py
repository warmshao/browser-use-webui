import pdb
import logging
from dotenv import load_dotenv
load_dotenv()
import os
import glob
import asyncio
import argparse
import re
import yaml
import subprocess
import requests
import gradio as gr
from playwright.async_api import async_playwright
from browser_use.agent.service import Agent
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContextConfig, BrowserContextWindowSize
from langchain_ollama import ChatOllama
from src.utils.agent_state import AgentState
from src.utils import utils
from src.agent.custom_agent import CustomAgent
from src.browser.custom_browser import CustomBrowser
from src.agent.custom_prompts import CustomSystemPrompt, CustomAgentMessagePrompt
from src.browser.custom_context import BrowserContextConfig as CustomBrowserContextConfig, CustomBrowserContext
from src.controller.custom_controller import CustomController
from gradio.themes import Citrus, Default, Glass, Monochrome, Ocean, Origin, Soft, Base
from src.utils.default_config_settings import default_config, load_config_from_file, save_config_to_file, save_current_config, update_ui_from_config
from src.utils.utils import update_model_dropdown, get_latest_files, capture_screenshot

logger = logging.getLogger(__name__)

# Global variables
_global_browser = None
_global_browser_context = None
_global_agent = None
_global_agent_state = AgentState()

# llms.txt handling
def fetch_llms_txt(prompt):
    url_match = re.search(r'https?://[^\s]+/llms.txt', prompt)
    if url_match:
        url = url_match.group(0)
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    elif os.path.exists('llms.txt'):
        with open('llms.txt', 'r') as file:
            return file.read()
    else:
        raise FileNotFoundError("llms.txt not found locally and no URL provided in prompt.")

def parse_llms_txt(content):
    data = yaml.safe_load(content)
    return data['actions']

def pre_evaluate_prompt(prompt):
    try:
        content = fetch_llms_txt(prompt)  # llms.txtの内容を取得
        actions = parse_llms_txt(content)  # YAMLをパース
        for action in actions:
            if 'name' in action and action['name'] in prompt:  # 'name'キーの存在を確認
                return action
            else:
                print(f"アクションに 'name' キーが見つかりません: {action}")
        return None
    except Exception as e:
        print(f"pre_evaluate_prompt でエラー: {str(e)}")
        return None

def extract_params(prompt, param_names):
    params = {}
    if not param_names:
        return params
    for param in param_names.split(','):
        param = param.strip()
        match = re.search(rf'{param}=(\S+)', prompt)
        if match:
            params[param] = match.group(1)
    return params

async def run_script(script_info, params, headless=False, save_recording_path=None):
    script_path = os.path.join('tmp', 'myscript', script_info['script'])
    if not os.path.exists(script_path):
        logger.error(f"Script not found: {script_path}")
        raise FileNotFoundError(f"Script not found: {script_path}")
    
    # Pytestコマンドを構築
    command = ['pytest', script_path]
    if not headless:
        command.append('--headed')
    
    # パラメータを追加
    if 'query' in params:
        command.extend(['--query', params['query']])
    
    # スローモーションを追加（llms.txtから取得）
    slowmo = script_info.get('slowmo')  # slowmoがなければNone
    if slowmo is not None:
        try:
            slowmo_ms = int(slowmo)  # 整数に変換
            command.extend(['--slowmo', str(slowmo_ms)])
            logger.info(f"Slow motion enabled with {slowmo_ms}ms delay")
        except ValueError:
            logger.warning(f"Invalid slowmo value: {slowmo}, ignoring")

    # レコーディングが有効なら、録画パスを指定
    if save_recording_path:
        os.makedirs(save_recording_path, exist_ok=True)
        logger.info(f"Recording enabled, saving to: {save_recording_path}")

    # コマンド実行前にログ出力
    logger.info(f"Executing command: {' '.join(command)}")
    
    # 非同期でコマンドを実行
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    
    # 実行結果をログに記録
    if process.returncode != 0:
        error_msg = f"Script execution failed: {stderr.decode()}"
        logger.error(error_msg)
        return error_msg, None
    else:
        success_msg = f"Script executed successfully: {stdout.decode()}"
        logger.info(success_msg)
        return success_msg, script_path

def resolve_sensitive_env_variables(text):
    if not text:
        return text
    env_vars = re.findall(r'\$SENSITIVE_[A-Za-z0-9_]*', text)
    result = text
    for var in env_vars:
        env_name = var[1:]
        env_value = os.getenv(env_name)
        if env_value is not None:
            result = result.replace(var, env_value)
    return result

async def stop_agent():
    global _global_agent_state, _global_browser_context, _global_browser, _global_agent
    try:
        _global_agent.stop()
        message = "Stop requested - the agent will halt at the next safe point"
        logger.info(f"🛑 {message}")
        return (
            message,
            gr.update(value="Stopping...", interactive=False),
            gr.update(interactive=False),
        )
    except Exception as e:
        error_msg = f"Error during stop: {str(e)}"
        logger.error(error_msg)
        return (
            error_msg,
            gr.update(value="Stop", interactive=True),
            gr.update(interactive=True)
        )

async def stop_research_agent():
    global _global_agent_state, _global_browser_context, _global_browser
    try:
        _global_agent_state.request_stop()
        message = "Stop requested - the agent will halt at the next safe point"
        logger.info(f"🛑 {message}")
        return (
            gr.update(value="Stopping...", interactive=False),
            gr.update(interactive=False),
        )
    except Exception as e:
        error_msg = f"Error during stop: {str(e)}"
        logger.error(error_msg)
        return (
            gr.update(value="Stop", interactive=True),
            gr.update(interactive=True)
        )

async def run_browser_agent(
    agent_type, llm_provider, llm_model_name, llm_num_ctx, llm_temperature, llm_base_url, llm_api_key,
    use_own_browser, keep_browser_open, headless, disable_security, window_w, window_h,
    save_recording_path, save_agent_history_path, save_trace_path, enable_recording, task, add_infos,
    max_steps, use_vision, max_actions_per_step, tool_calling_method
):
    global _global_agent_state
    _global_agent_state.clear_stop()

    try:
        # Pre-evaluate prompt
        script_info = pre_evaluate_prompt(task)
        if script_info:
            params = extract_params(task, script_info.get('params', ''))
            # Browser Settingsを反映してスクリプト実行
            script_output, script_path = await run_script(script_info, params, headless=headless, save_recording_path=save_recording_path if enable_recording else None)
            return (
                script_output,
                "",
                f"Executed script: {script_path}",
                "",
                script_path if enable_recording else None,  # レコーディングパスを返す
                None,
                None,
                gr.update(value="Stop", interactive=True),
                gr.update(interactive=True)
            )

        if not enable_recording:
            save_recording_path = None
        if save_recording_path:
            os.makedirs(save_recording_path, exist_ok=True)
        existing_videos = set()
        if save_recording_path:
            existing_videos = set(glob.glob(os.path.join(save_recording_path, "*.[mM][pP]4")) + glob.glob(os.path.join(save_recording_path, "*.[wW][eE][bB][mM]")))

        task = resolve_sensitive_env_variables(task)
        llm = utils.get_llm_model(
            provider=llm_provider, model_name=llm_model_name, num_ctx=llm_num_ctx,
            temperature=llm_temperature, base_url=llm_base_url, api_key=llm_api_key,
        )
        if agent_type == "org":
            final_result, errors, model_actions, model_thoughts, trace_file, history_file = await run_org_agent(
                llm=llm, use_own_browser=use_own_browser, keep_browser_open=keep_browser_open, headless=headless,
                disable_security=disable_security, window_w=window_w, window_h=window_h,
                save_recording_path=save_recording_path, save_agent_history_path=save_agent_history_path,
                save_trace_path=save_trace_path, task=task, max_steps=max_steps, use_vision=use_vision,
                max_actions_per_step=max_actions_per_step, tool_calling_method=tool_calling_method
            )
        elif agent_type == "custom":
            final_result, errors, model_actions, model_thoughts, trace_file, history_file = await run_custom_agent(
                llm=llm, use_own_browser=use_own_browser, keep_browser_open=keep_browser_open, headless=headless,
                disable_security=disable_security, window_w=window_w, window_h=window_h,
                save_recording_path=save_recording_path, save_agent_history_path=save_agent_history_path,
                save_trace_path=save_trace_path, task=task, add_infos=add_infos, max_steps=max_steps,
                use_vision=use_vision, max_actions_per_step=max_actions_per_step, tool_calling_method=tool_calling_method
            )
        else:
            raise ValueError(f"Invalid agent type: {agent_type}")

        latest_video = None
        if save_recording_path:
            new_videos = set(glob.glob(os.path.join(save_recording_path, "*.[mM][pP]4")) + glob.glob(os.path.join(save_recording_path, "*.[wW][eE][bB][mM]")))
            if new_videos - existing_videos:
                latest_video = list(new_videos - existing_videos)[0]

        return (
            final_result, errors, model_actions, model_thoughts, latest_video, trace_file, history_file,
            gr.update(value="Stop", interactive=True), gr.update(interactive=True)
        )
    except gr.Error:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        errors = str(e) + "\n" + traceback.format_exc()
        logger.error(errors)
        return (
            '', errors, '', '', None, None, None,
            gr.update(value="Stop", interactive=True), gr.update(interactive=True)
        )

async def run_org_agent(
        llm, use_own_browser, keep_browser_open, headless, disable_security, window_w, window_h,
        save_recording_path, save_agent_history_path, save_trace_path, task, max_steps, use_vision,
        max_actions_per_step, tool_calling_method
):
    global _global_browser, _global_browser_context, _global_agent_state, _global_agent
    _global_agent_state.clear_stop()

    try:
        extra_chromium_args = [f"--window-size={window_w},{window_h}"]
        if use_own_browser:
            chrome_path = os.getenv("CHROME_PATH", None)
            if chrome_path == "":
                chrome_path = None
            chrome_user_data = os.getenv("CHROME_USER_DATA", None)
            if chrome_user_data:
                extra_chromium_args += [f"--user-data-dir={chrome_user_data}"]
        else:
            chrome_path = None

        if _global_browser is None:
            _global_browser = Browser(
                config=BrowserConfig(
                    headless=headless, disable_security=disable_security, chrome_instance_path=chrome_path,
                    extra_chromium_args=extra_chromium_args,
                )
            )

        if _global_browser_context is None:
            _global_browser_context = await _global_browser.new_context(
                config=BrowserContextConfig(
                    trace_path=save_trace_path if save_trace_path else None,
                    save_recording_path=save_recording_path if save_recording_path else None,
                    no_viewport=False, browser_window_size=BrowserContextWindowSize(width=window_w, height=window_h),
                )
            )

        if _global_agent is None:
            _global_agent = Agent(
                task=task, llm=llm, use_vision=use_vision, browser=_global_browser,
                browser_context=_global_browser_context, max_actions_per_step=max_actions_per_step,
                tool_calling_method=tool_calling_method
            )
        history = await _global_agent.run(max_steps=max_steps)
        history_file = os.path.join(save_agent_history_path, f"{_global_agent.agent_id}.json")
        _global_agent.save_history(history_file)

        final_result = history.final_result()
        errors = history.errors()
        model_actions = history.model_actions()
        model_thoughts = history.model_thoughts()
        trace_file = get_latest_files(save_trace_path)

        return final_result, errors, model_actions, model_thoughts, trace_file.get('.zip'), history_file
    except Exception as e:
        import traceback
        traceback.print_exc()
        errors = str(e) + "\n" + traceback.format_exc()
        return '', errors, '', '', None, None
    finally:
        _global_agent = None
        if not keep_browser_open:
            if _global_browser_context:
                await _global_browser_context.close()
                _global_browser_context = None
            if _global_browser:
                await _global_browser.close()
                _global_browser = None

async def run_custom_agent(
        llm, use_own_browser, keep_browser_open, headless, disable_security, window_w, window_h,
        save_recording_path, save_agent_history_path, save_trace_path, task, add_infos, max_steps,
        use_vision, max_actions_per_step, tool_calling_method
):
    global _global_browser, _global_browser_context, _global_agent_state, _global_agent
    _global_agent_state.clear_stop()

    try:
        extra_chromium_args = [f"--window-size={window_w},{window_h}"]
        if use_own_browser:
            chrome_path = os.getenv("CHROME_PATH", None)
            if chrome_path == "":
                chrome_path = None
            chrome_user_data = os.getenv("CHROME_USER_DATA", None)
            if chrome_user_data:
                extra_chromium_args += [f"--user-data-dir={chrome_user_data}"]
        else:
            chrome_path = None

        controller = CustomController()
        if _global_browser is None:
            _global_browser = CustomBrowser(
                config=BrowserConfig(
                    headless=headless, disable_security=disable_security, chrome_instance_path=chrome_path,
                    extra_chromium_args=extra_chromium_args,
                )
            )

        if _global_browser_context is None:
            _global_browser_context = await _global_browser.new_context(
                config=CustomBrowserContextConfig(
                    trace_path=save_trace_path if save_trace_path else None,
                    save_recording_path=save_recording_path if save_recording_path else None,
                    no_viewport=False, browser_window_size=BrowserContextWindowSize(width=window_w, height=window_h),
                )
            )

        if _global_agent is None:
            _global_agent = CustomAgent(
                task=task, add_infos=add_infos, use_vision=use_vision, llm=llm, browser=_global_browser,
                browser_context=_global_browser_context, controller=controller, system_prompt_class=CustomSystemPrompt,
                agent_prompt_class=CustomAgentMessagePrompt, max_actions_per_step=max_actions_per_step,
                tool_calling_method=tool_calling_method
            )
        history = await _global_agent.run(max_steps=max_steps)
        history_file = os.path.join(save_agent_history_path, f"{_global_agent.agent_id}.json")
        _global_agent.save_history(history_file)

        final_result = history.final_result()
        errors = history.errors()
        model_actions = history.model_actions()
        model_thoughts = history.model_thoughts()
        trace_file = get_latest_files(save_trace_path)

        return final_result, errors, model_actions, model_thoughts, trace_file.get('.zip'), history_file
    except Exception as e:
        import traceback
        traceback.print_exc()
        errors = str(e) + "\n" + traceback.format_exc()
        return '', errors, '', '', None, None
    finally:
        _global_agent = None
        if not keep_browser_open:
            if _global_browser_context:
                await _global_browser_context.close()
                _global_browser_context = None
            if _global_browser:
                await _global_browser.close()
                _global_browser = None

async def run_with_stream(
    agent_type, llm_provider, llm_model_name, llm_num_ctx, llm_temperature, llm_base_url, llm_api_key,
    use_own_browser, keep_browser_open, headless, disable_security, window_w, window_h,
    save_recording_path, save_agent_history_path, save_trace_path, enable_recording, task, add_infos,
    max_steps, use_vision, max_actions_per_step, tool_calling_method
):
    global _global_agent_state
    stream_vw = 80
    stream_vh = int(80 * window_h // window_w)
    if not headless:
        result = await run_browser_agent(
            agent_type=agent_type, llm_provider=llm_provider, llm_model_name=llm_model_name, llm_num_ctx=llm_num_ctx,
            llm_temperature=llm_temperature, llm_base_url=llm_base_url, llm_api_key=llm_api_key,
            use_own_browser=use_own_browser, keep_browser_open=keep_browser_open, headless=headless,
            disable_security=disable_security, window_w=window_w, window_h=window_h,
            save_recording_path=save_recording_path, save_agent_history_path=save_agent_history_path,
            save_trace_path=save_trace_path, enable_recording=enable_recording, task=task, add_infos=add_infos,
            max_steps=max_steps, use_vision=use_vision, max_actions_per_step=max_actions_per_step,
            tool_calling_method=tool_calling_method
        )
        html_content = f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Using browser...</h1>"
        yield [html_content] + list(result)
    else:
        try:
            _global_agent_state.clear_stop()
            agent_task = asyncio.create_task(
                run_browser_agent(
                    agent_type=agent_type, llm_provider=llm_provider, llm_model_name=llm_model_name,
                    llm_num_ctx=llm_num_ctx, llm_temperature=llm_temperature, llm_base_url=llm_base_url,
                    llm_api_key=llm_api_key, use_own_browser=use_own_browser, keep_browser_open=keep_browser_open,
                    headless=headless, disable_security=disable_security, window_w=window_w, window_h=window_h,
                    save_recording_path=save_recording_path, save_agent_history_path=save_agent_history_path,
                    save_trace_path=save_trace_path, enable_recording=enable_recording, task=task, add_infos=add_infos,
                    max_steps=max_steps, use_vision=use_vision, max_actions_per_step=max_actions_per_step,
                    tool_calling_method=tool_calling_method
                )
            )

            html_content = f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Using browser...</h1>"
            final_result = errors = model_actions = model_thoughts = ""
            latest_videos = trace = history_file = None

            while not agent_task.done():
                try:
                    encoded_screenshot = await capture_screenshot(_global_browser_context)
                    if encoded_screenshot is not None:
                        html_content = f'<img src="data:image/jpeg;base64,{encoded_screenshot}" style="width:{stream_vw}vw; height:{stream_vh}vh ; border:1px solid #ccc;">'
                    else:
                        html_content = f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Waiting for browser session...</h1>"
                except Exception as e:
                    html_content = f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Waiting for browser session...</h1>"

                if _global_agent_state and _global_agent_state.is_stop_requested():
                    yield [
                        html_content, final_result, errors, model_actions, model_thoughts, latest_videos, trace, history_file,
                        gr.update(value="Stopping...", interactive=False), gr.update(interactive=False),
                    ]
                    break
                else:
                    yield [
                        html_content, final_result, errors, model_actions, model_thoughts, latest_videos, trace, history_file,
                        gr.update(value="Stop", interactive=True), gr.update(interactive=True)
                    ]
                await asyncio.sleep(0.05)

            try:
                result = await agent_task
                final_result, errors, model_actions, model_thoughts, latest_videos, trace, history_file, stop_button, run_button = result
            except gr.Error:
                final_result = ""
                model_actions = ""
                model_thoughts = ""
                latest_videos = trace = history_file = None
            except Exception as e:
                errors = f"Agent error: {str(e)}"

            yield [
                html_content, final_result, errors, model_actions, model_thoughts, latest_videos, trace, history_file,
                stop_button, run_button
            ]
        except Exception as e:
            import traceback
            yield [
                f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Waiting for browser session...</h1>",
                "", f"Error: {str(e)}\n{traceback.format_exc()}", "", "", None, None, None,
                gr.update(value="Stop", interactive=True), gr.update(interactive=True)
            ]

theme_map = {
    "Default": Default(), "Soft": Soft(), "Monochrome": Monochrome(), "Glass": Glass(),
    "Origin": Origin(), "Citrus": Citrus(), "Ocean": Ocean(), "Base": Base()
}

async def close_global_browser():
    global _global_browser, _global_browser_context
    if _global_browser_context:
        await _global_browser_context.close()
        _global_browser_context = None
    if _global_browser:
        await _global_browser.close()
        _global_browser = None

async def run_deep_search(research_task, max_search_iteration_input, max_query_per_iter_input, llm_provider, llm_model_name, llm_num_ctx, llm_temperature, llm_base_url, llm_api_key, use_vision, use_own_browser, headless):
    from src.utils.deep_research import deep_research
    global _global_agent_state
    _global_agent_state.clear_stop()
    
    llm = utils.get_llm_model(
        provider=llm_provider, model_name=llm_model_name, num_ctx=llm_num_ctx,
        temperature=llm_temperature, base_url=llm_base_url, api_key=llm_api_key,
    )
    markdown_content, file_path = await deep_research(research_task, llm, _global_agent_state,
                                                      max_search_iterations=max_search_iteration_input,
                                                      max_query_num=max_query_per_iter_input,
                                                      use_vision=use_vision, headless=headless,
                                                      use_own_browser=use_own_browser)
    return markdown_content, file_path, gr.update(value="Stop", interactive=True), gr.update(interactive=True)

def create_ui(config, theme_name="Ocean"):
    css = """
    .gradio-container { max-width: 1200px !important; margin: auto !important; padding-top: 20px !important; }
    .header-text { text-align: center; margin-bottom: 30px; }
    .theme-section { margin-bottom: 20px; padding: 15px; border-radius: 10px; }
    """

    with gr.Blocks(title="Bykilt", theme=theme_map[theme_name], css=css) as demo:
        with gr.Row():
            gr.Markdown("# 🌐 Bykilt\n### Enhanced Browser Control with AI and human, because for you", elem_classes=["header-text"])

        with gr.Tabs() as tabs:
            with gr.TabItem("⚙️ Agent Settings", id=1):
                with gr.Group():
                    agent_type = gr.Radio(["org", "custom"], label="Agent Type", value=config['agent_type'], info="Select the type of agent to use")
                    with gr.Column():
                        max_steps = gr.Slider(minimum=1, maximum=200, value=config['max_steps'], step=1, label="Max Run Steps", info="Maximum number of steps the agent will take")
                        max_actions_per_step = gr.Slider(minimum=1, maximum=20, value=config['max_actions_per_step'], step=1, label="Max Actions per Step", info="Maximum number of actions the agent will take per step")
                    with gr.Column():
                        use_vision = gr.Checkbox(label="Use Vision", value=config['use_vision'], info="Enable visual processing capabilities")
                        tool_calling_method = gr.Dropdown(label="Tool Calling Method", value=config['tool_calling_method'], interactive=True, allow_custom_value=True, choices=["auto", "json_schema", "function_calling"], info="Tool Calls Function Name", visible=False)

            with gr.TabItem("🔧 LLM Configuration", id=2):
                with gr.Group():
                    llm_provider = gr.Dropdown(choices=[provider for provider, model in utils.model_names.items()], label="LLM Provider", value=config['llm_provider'], info="Select your preferred language model provider")
                    llm_model_name = gr.Dropdown(label="Model Name", choices=utils.model_names['openai'], value=config['llm_model_name'], interactive=True, allow_custom_value=True, info="Select a model from the dropdown or type a custom model name")
                    llm_num_ctx = gr.Slider(minimum=2**8, maximum=2**16, value=config['llm_num_ctx'], step=1, label="Max Context Length", info="Controls max context length model needs to handle (less = faster)", visible=config['llm_provider'] == "ollama")
                    llm_temperature = gr.Slider(minimum=0.0, maximum=2.0, value=config['llm_temperature'], step=0.1, label="Temperature", info="Controls randomness in model outputs")
                    with gr.Row():
                        llm_base_url = gr.Textbox(label="Base URL", value=config['llm_base_url'], info="API endpoint URL (if required)")
                        llm_api_key = gr.Textbox(label="API Key", type="password", value=config['llm_api_key'], info="Your API key (leave blank to use .env)")

                    llm_provider.change(fn=lambda provider: gr.update(visible=provider == "ollama"), inputs=llm_provider, outputs=llm_num_ctx)

            with gr.TabItem("🌐 Browser Settings", id=3):
                with gr.Group():
                    with gr.Row():
                        use_own_browser = gr.Checkbox(label="Use Own Browser", value=config['use_own_browser'], info="Use your existing browser instance")
                        keep_browser_open = gr.Checkbox(label="Keep Browser Open", value=config['keep_browser_open'], info="Keep Browser Open between Tasks")
                        headless = gr.Checkbox(label="Headless Mode", value=config['headless'], info="Run browser without GUI")
                        disable_security = gr.Checkbox(label="Disable Security", value=config['disable_security'], info="Disable browser security features")
                        enable_recording = gr.Checkbox(label="Enable Recording", value=config['enable_recording'], info="Enable saving browser recordings")
                    with gr.Row():
                        window_w = gr.Number(label="Window Width", value=config['window_w'], info="Browser window width")
                        window_h = gr.Number(label="Window Height", value=config['window_h'], info="Browser window height")
                    save_recording_path = gr.Textbox(label="Recording Path", placeholder="e.g. ./tmp/record_videos", value=config['save_recording_path'], info="Path to save browser recordings", interactive=True)
                    save_trace_path = gr.Textbox(label="Trace Path", placeholder="e.g. ./tmp/traces", value=config['save_trace_path'], info="Path to save Agent traces", interactive=True)
                    save_agent_history_path = gr.Textbox(label="Agent History Save Path", placeholder="e.g., ./tmp/agent_history", value=config['save_agent_history_path'], info="Specify the directory where agent history should be saved.", interactive=True)

            with gr.TabItem("🤖 Run Agent", id=4):
                task = gr.Textbox(label="Task Description", lines=4, placeholder="Enter your task or script name (e.g., search-for-something query=python)", value=config['task'], info="Describe what you want the agent to do or specify a script from llms.txt")
                add_infos = gr.Textbox(label="Additional Information", lines=3, placeholder="Add any helpful context or instructions...", info="Optional hints to help the LLM complete the task")
                with gr.Row():
                    run_button = gr.Button("▶️ Run Agent", variant="primary", scale=2)
                    stop_button = gr.Button("⏹️ Stop", variant="stop", scale=1)
                with gr.Row():
                    browser_view = gr.HTML(value="<h1 style='width:80vw; height:50vh'>Waiting for browser session...</h1>", label="Live Browser View")

            with gr.TabItem("🧐 Deep Research", id=5):
                research_task_input = gr.Textbox(label="Research Task", lines=5, value="Compose a report on the use of Reinforcement Learning for training Large Language Models, encompassing its origins, current advancements, and future prospects, substantiated with examples of relevant models and techniques. The report should reflect original insights and analysis, moving beyond mere summarization of existing literature.")
                with gr.Row():
                    max_search_iteration_input = gr.Number(label="Max Search Iteration", value=3, precision=0)
                    max_query_per_iter_input = gr.Number(label="Max Query per Iteration", value=1, precision=0)
                with gr.Row():
                    research_button = gr.Button("▶️ Run Deep Research", variant="primary", scale=2)
                    stop_research_button = gr.Button("⏹️ Stop", variant="stop", scale=1)
                markdown_output_display = gr.Markdown(label="Research Report")
                markdown_download = gr.File(label="Download Research Report")

            with gr.TabItem("📊 Results", id=6):
                with gr.Group():
                    recording_display = gr.Video(label="Latest Recording")
                    gr.Markdown("### Results")
                    with gr.Row():
                        with gr.Column():
                            final_result_output = gr.Textbox(label="Final Result", lines=3, show_label=True)
                        with gr.Column():
                            errors_output = gr.Textbox(label="Errors", lines=3, show_label=True)
                    with gr.Row():
                        with gr.Column():
                            model_actions_output = gr.Textbox(label="Model Actions", lines=3, show_label=True)
                        with gr.Column():
                            model_thoughts_output = gr.Textbox(label="Model Thoughts", lines=3, show_label=True)
                    trace_file = gr.File(label="Trace File")
                    agent_history_file = gr.File(label="Agent History")

                    stop_button.click(fn=stop_agent, inputs=[], outputs=[errors_output, stop_button, run_button])
                    run_button.click(
                        fn=run_with_stream,
                        inputs=[
                            agent_type, llm_provider, llm_model_name, llm_num_ctx, llm_temperature, llm_base_url, llm_api_key,
                            use_own_browser, keep_browser_open, headless, disable_security, window_w, window_h,
                            save_recording_path, save_agent_history_path, save_trace_path, enable_recording, task, add_infos,
                            max_steps, use_vision, max_actions_per_step, tool_calling_method
                        ],
                        outputs=[
                            browser_view, final_result_output, errors_output, model_actions_output, model_thoughts_output,
                            recording_display, trace_file, agent_history_file, stop_button, run_button
                        ],
                    )
                    research_button.click(
                        fn=run_deep_search,
                        inputs=[research_task_input, max_search_iteration_input, max_query_per_iter_input, llm_provider, llm_model_name, llm_num_ctx, llm_temperature, llm_base_url, llm_api_key, use_vision, use_own_browser, headless],
                        outputs=[markdown_output_display, markdown_download, stop_research_button, research_button]
                    )
                    stop_research_button.click(fn=stop_research_agent, inputs=[], outputs=[stop_research_button, research_button])

            with gr.TabItem("🎥 Recordings", id=7):
                def list_recordings(save_recording_path):
                    if not os.path.exists(save_recording_path):
                        return []
                    recordings = glob.glob(os.path.join(save_recording_path, "*.[mM][pP]4")) + glob.glob(os.path.join(save_recording_path, "*.[wW][eE][bB][mM]"))
                    recordings.sort(key=os.path.getctime)
                    numbered_recordings = [(recording, f"{idx}. {os.path.basename(recording)}") for idx, recording in enumerate(recordings, start=1)]
                    return numbered_recordings

                recordings_gallery = gr.Gallery(label="Recordings", value=list_recordings(config['save_recording_path']), columns=3, height="auto", object_fit="contain")
                refresh_button = gr.Button("🔄 Refresh Recordings", variant="secondary")
                refresh_button.click(fn=list_recordings, inputs=save_recording_path, outputs=recordings_gallery)

            with gr.TabItem("📁 Configuration", id=8):
                with gr.Group():
                    config_file_input = gr.File(label="Load Config File", file_types=[".pkl"], interactive=True)
                    load_config_button = gr.Button("Load Existing Config From File", variant="primary")
                    save_config_button = gr.Button("Save Current Config", variant="primary")
                    config_status = gr.Textbox(label="Status", lines=2, interactive=False)

                    load_config_button.click(
                        fn=update_ui_from_config,
                        inputs=[config_file_input],
                        outputs=[
                            agent_type, max_steps, max_actions_per_step, use_vision, tool_calling_method,
                            llm_provider, llm_model_name, llm_num_ctx, llm_temperature, llm_base_url, llm_api_key,
                            use_own_browser, keep_browser_open, headless, disable_security, enable_recording,
                            window_w, window_h, save_recording_path, save_trace_path, save_agent_history_path,
                            task, config_status
                        ]
                    )
                    save_config_button.click(
                        fn=save_current_config,
                        inputs=[
                            agent_type, max_steps, max_actions_per_step, use_vision, tool_calling_method,
                            llm_provider, llm_model_name, llm_num_ctx, llm_temperature, llm_base_url, llm_api_key,
                            use_own_browser, keep_browser_open, headless, disable_security,
                            enable_recording, window_w, window_h, save_recording_path, save_trace_path,
                            save_agent_history_path, task,
                        ],
                        outputs=[config_status]
                    )

        llm_provider.change(lambda provider, api_key, base_url: update_model_dropdown(provider, api_key, base_url), inputs=[llm_provider, llm_api_key, llm_base_url], outputs=llm_model_name)
        enable_recording.change(lambda enabled: gr.update(interactive=enabled), inputs=enable_recording, outputs=save_recording_path)
        use_own_browser.change(fn=close_global_browser)
        keep_browser_open.change(fn=close_global_browser)

    return demo

def main():
    parser = argparse.ArgumentParser(description="Gradio UI for Bykilt Agent")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="IP address to bind to")
    parser.add_argument("--port", type=int, default=7788, help="Port to listen on")
    parser.add_argument("--theme", type=str, default="Ocean", choices=theme_map.keys(), help="Theme to use for the UI")
    parser.add_argument("--dark-mode", action="store_true", help="Enable dark mode")
    args = parser.parse_args()

    config_dict = default_config()
    demo = create_ui(config_dict, theme_name=args.theme)
    demo.launch(server_name=args.ip, server_port=args.port)

if __name__ == '__main__':
    main()