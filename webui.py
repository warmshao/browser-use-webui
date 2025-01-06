# -*- coding: utf-8 -*-
# @Time    : 2025/1/1
# @Author  : wenshao
# @Email   : wenshaoguo1026@gmail.com
# @Project : browser-use-webui
# @FileName: webui.py
import pdb

from dotenv import load_dotenv

load_dotenv()
import argparse

import asyncio

import gradio as gr
import asyncio
import os
from pprint import pprint
from typing import List, Dict, Any

from playwright.async_api import async_playwright
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import (
    BrowserContext,
    BrowserContextConfig,
    BrowserContextWindowSize,
)
from browser_use.agent.service import Agent

from src.browser.custom_browser import CustomBrowser, BrowserConfig
from src.browser.custom_context import BrowserContext, BrowserContextConfig
from src.controller.custom_controller import CustomController
from src.agent.custom_agent import CustomAgent
from src.agent.custom_prompts import CustomSystemPrompt

from src.utils import utils


async def run_browser_agent(
        agent_type,
        llm_provider,
        llm_model_name,
        llm_temperature,
        llm_base_url,
        llm_api_key,
        use_own_browser,
        headless,
        disable_security,
        window_w,
        window_h,
        save_recording_path,
        task,
        add_infos,
        max_steps,
        use_vision
):
    """
    Runs the browser agent based on user configurations.
    """

    llm = utils.get_llm_model(
        provider=llm_provider,
        model_name=llm_model_name,
        temperature=llm_temperature,
        base_url=llm_base_url,
        api_key=llm_api_key
    )
    if agent_type == "org":
        return await run_org_agent(
            llm=llm,
            headless=headless,
            disable_security=disable_security,
            window_w=window_w,
            window_h=window_h,
            save_recording_path=save_recording_path,
            task=task,
            max_steps=max_steps,
            use_vision=use_vision
        )
    elif agent_type == "custom":
        return await run_custom_agent(
            llm=llm,
            use_own_browser=use_own_browser,
            headless=headless,
            disable_security=disable_security,
            window_w=window_w,
            window_h=window_h,
            save_recording_path=save_recording_path,
            task=task,
            add_infos=add_infos,
            max_steps=max_steps,
            use_vision=use_vision
        )
    else:
        raise ValueError(f"Invalid agent type: {agent_type}")


async def run_org_agent(
        llm,
        headless,
        disable_security,
        window_w,
        window_h,
        save_recording_path,
        task,
        max_steps,
        use_vision
):
    browser = Browser(
        config=BrowserConfig(
            headless=headless,
            disable_security=disable_security,
            extra_chromium_args=[f'--window-size={window_w},{window_h}'],
        )
    )
    async with await browser.new_context(
            config=BrowserContextConfig(
                trace_path='./tmp/traces',
                save_recording_path=save_recording_path if save_recording_path else None,
                no_viewport=False,
                browser_window_size=BrowserContextWindowSize(width=window_w, height=window_h),
            )
    ) as browser_context:
        agent = Agent(
            task=task,
            llm=llm,
            use_vision=use_vision,
            browser_context=browser_context,
        )
        history = await agent.run(max_steps=max_steps)

        final_result = history.final_result()
        errors = history.errors()
        model_actions = history.model_actions()
        model_thoughts = history.model_thoughts()
    await browser.close()
    return final_result, errors, model_actions, model_thoughts


async def run_custom_agent(
        llm,
        use_own_browser,
        headless,
        disable_security,
        window_w,
        window_h,
        save_recording_path,
        task,
        add_infos,
        max_steps,
        use_vision
):
    controller = CustomController()
    playwright = None
    browser_context_ = None
    try:
        if use_own_browser:
            playwright = await async_playwright().start()
            chrome_exe = os.getenv("CHROME_PATH", "")
            chrome_use_data = os.getenv("CHROME_USER_DATA", "")
            browser_context_ = await playwright.chromium.launch_persistent_context(
                user_data_dir=chrome_use_data,
                executable_path=chrome_exe,
                no_viewport=False,
                headless=headless,  # 保持浏览器窗口可见
                user_agent=(
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                    '(KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36'
                ),
                java_script_enabled=True,
                bypass_csp=disable_security,
                ignore_https_errors=disable_security,
                record_video_dir=save_recording_path if save_recording_path else None,
                record_video_size={'width': window_w, 'height': window_h}
            )
        else:
            browser_context_ = None

        browser = CustomBrowser(
            config=BrowserConfig(
                headless=headless,
                disable_security=disable_security,
                extra_chromium_args=[f'--window-size={window_w},{window_h}'],
            )
        )
        async with await browser.new_context(
                config=BrowserContextConfig(
                    trace_path='./tmp/result_processing',
                    save_recording_path=save_recording_path if save_recording_path else None,
                    no_viewport=False,
                    browser_window_size=BrowserContextWindowSize(width=window_w, height=window_h),
                ),

                context=browser_context_
        ) as browser_context:
            agent = CustomAgent(
                task=task,
                add_infos=add_infos,
                use_vision=use_vision,
                llm=llm,
                browser_context=browser_context,
                controller=controller,
                system_prompt_class=CustomSystemPrompt
            )
            history = await agent.run(max_steps=max_steps)

            final_result = history.final_result()
            errors = history.errors()
            model_actions = history.model_actions()
            model_thoughts = history.model_thoughts()

    except Exception as e:
        import traceback
        traceback.print_exc()
        final_result = ""
        errors = str(e) + "\n" + traceback.format_exc()
        model_actions = ""
        model_thoughts = ""
    finally:
        # 显式关闭持久化上下文
        persistence_enabled = (
            os.getenv("CHROME_PERSISTENT_SESSION", "").lower() == "true"
        )
        if not persistence_enabled:
            if browser_context_:
                await browser_context_.close()

            # 关闭 Playwright 对象
            if playwright:
                await playwright.stop()
            await browser.close()
    return final_result, errors, model_actions, model_thoughts


def main():
    parser = argparse.ArgumentParser(description="Gradio UI for Browser Agent")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="IP address to bind to")
    parser.add_argument("--port", type=int, default=7788, help="Port to listen on")
    args = parser.parse_args()

    js_func = """
        function refresh() {
            const url = new URL(window.location);

            if (url.searchParams.get('__theme') !== 'dark') {
                url.searchParams.set('__theme', 'dark');
                window.location.href = url.href;
            }
        }
        """

    # Gradio UI setup
    with gr.Blocks(title="Browser Use WebUI", theme=gr.themes.Soft(font=[gr.themes.GoogleFont("Plus Jakarta Sans")]),
                   js=js_func) as demo:
        gr.Markdown("<center><h1>Browser Use WebUI</h1></center>")
        with gr.Row():
            agent_type = gr.Radio(["org", "custom"], label="Agent Type", value="custom")
            max_steps = gr.Number(label="max run steps", value=100)
            use_vision = gr.Checkbox(label="use vision", value=True)
        with gr.Row():
            llm_provider = gr.Dropdown(
                ["anthropic", "openai", "gemini", "azure_openai", "deepseek", "ollama"], label="LLM Provider",
                value="gemini"
            )
            llm_model_name = gr.Textbox(label="LLM Model Name", value="gemini-2.0-flash-exp")
            llm_temperature = gr.Number(label="LLM Temperature", value=1.0)
        with gr.Row():
            llm_base_url = gr.Textbox(label="LLM Base URL")
            llm_api_key = gr.Textbox(label="LLM API Key", type="password")

        with gr.Accordion("Browser Settings", open=False):
            use_own_browser = gr.Checkbox(label="Use Own Browser", value=False)
            headless = gr.Checkbox(label="Headless", value=False)
            disable_security = gr.Checkbox(label="Disable Security", value=True)
            with gr.Row():
                window_w = gr.Number(label="Window Width", value=1920)
                window_h = gr.Number(label="Window Height", value=1080)
            save_recording_path = gr.Textbox(label="Save Recording Path", placeholder="e.g. ./tmp/record_videos",
                                             value="./tmp/record_videos")
        with gr.Accordion("Task Settings", open=True):
            task = gr.Textbox(label="Task", lines=10,
                              value="go to google.com and type 'OpenAI' click search and give me the first url")
            add_infos = gr.Textbox(label="Additional Infos(Optional): Hints to help LLM complete Task", lines=5)

        run_button = gr.Button("Run Agent", variant="primary")
        with gr.Column():
            final_result_output = gr.Textbox(label="Final Result", lines=5)
            errors_output = gr.Textbox(label="Errors", lines=5, )
            model_actions_output = gr.Textbox(label="Model Actions", lines=5)
            model_thoughts_output = gr.Textbox(label="Model Thoughts", lines=5)

        run_button.click(
            fn=run_browser_agent,
            inputs=[
                agent_type,
                llm_provider,
                llm_model_name,
                llm_temperature,
                llm_base_url,
                llm_api_key,
                use_own_browser,
                headless,
                disable_security,
                window_w,
                window_h,
                save_recording_path,
                task,
                add_infos,
                max_steps,
                use_vision
            ],
            outputs=[final_result_output, errors_output, model_actions_output, model_thoughts_output],
        )

    demo.launch(server_name=args.ip, server_port=args.port)


if __name__ == '__main__':
    main()
