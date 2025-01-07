# -*- coding: utf-8 -*-
# @Time    : 2025/1/1
# @Author  : wenshao
# @Email   : wenshaoguo1026@gmail.com
# @Project : browser-use-webui
# @FileName: webui.py
import pdb
import logging
import sys
import os

# Create logs directory if it doesn't exist
logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(logs_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(logs_dir, 'webui.log'), encoding='utf-8', mode='a')
    ]
)

logger = logging.getLogger(__name__)

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
    browser = None
    try:
        if use_own_browser:
            playwright = await async_playwright().start()
            chrome_exe = os.getenv("CHROME_PATH", "")
            chrome_use_data = os.getenv("CHROME_USER_DATA", "")

            if not chrome_exe or not chrome_use_data:
                raise ValueError("CHROME_PATH and CHROME_USER_DATA environment variables must be set when use_own_browser=True")

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
        try:
            # Clean up in reverse order of creation
            if browser:
                await browser.close()
            if browser_context_:
                await browser_context_.close()
            if playwright:
                await playwright.stop()
        except Exception as cleanup_error:
            logger = logging.getLogger(__name__)
            logger.error(f"Error during cleanup: {cleanup_error}")
    return final_result, errors, model_actions, model_thoughts


def main():
    parser = argparse.ArgumentParser(description="Gradio UI for Browser Agent")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="IP address to bind to")
    parser.add_argument("--port", type=int, default=7788, help="Port to listen on")
    args = parser.parse_args()

    css = """
    /* Modern UI Styles */
    :root {
        --primary-color: #2563eb;
        --secondary-color: #1e40af;
        --success-color: #059669;
        --error-color: #dc2626;
        --background-light: #f8fafc;
        --text-primary: #1e293b;
        --text-secondary: #64748b;
        --border-color: #e2e8f0;
        --radius-md: 8px;
        --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
        --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }

    /* Hide footer */
    footer {display: none !important;}

    /* Container styles */
    .container {
        max-width: 1200px;
        margin: auto;
        padding: 2rem;
    }

    /* Header styles */
    .header {
        text-align: center;
        margin-bottom: 2.5rem;
        padding: 2rem;
        background: linear-gradient(to right, #2563eb, #1e40af);
        border-radius: var(--radius-md);
        color: white;
        box-shadow: var(--shadow-md);
    }
    .header h1 {
        margin-bottom: 1rem;
        font-size: 2.5rem;
        font-weight: bold;
    }
    .header p {
        color: #e2e8f0;
        font-size: 1.1rem;
    }

    /* Tab styles */
    .tabs {
        margin-top: 1rem;
    }
    .tab-nav {
        background: var(--background-light);
        border-radius: var(--radius-md);
        padding: 0.5rem;
    }
    .tab-nav button {
        font-weight: 500;
        padding: 0.75rem 1.5rem;
    }
    .tab-nav button.selected {
        background: var(--primary-color);
        color: white;
    }

    /* Form elements */
    .gr-form {
        border: 1px solid var(--border-color);
        padding: 1.5rem;
        border-radius: var(--radius-md);
        background: white;
        box-shadow: var(--shadow-sm);
        margin-bottom: 1rem;
    }
    .gr-form:hover {
        box-shadow: var(--shadow-md);
        transition: box-shadow 0.2s;
    }
    .gr-input, .gr-select, .gr-checkbox {
        border: 1px solid var(--border-color);
        border-radius: var(--radius-md);
        padding: 0.5rem;
    }
    .gr-input:focus, .gr-select:focus {
        border-color: var(--primary-color);
        outline: none;
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.1);
    }
    .gr-button {
        border-radius: var(--radius-md);
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    .gr-button:hover {
        transform: translateY(-1px);
    }

    /* Section headers */
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 1.5rem 0 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--border-color);
    }

    /* Status indicators */
    .status-indicator {
        padding: 0.5rem 1rem;
        border-radius: var(--radius-md);
        display: inline-block;
        margin: 0.5rem;
        font-weight: 500;
        box-shadow: var(--shadow-sm);
    }
    .status-success {
        background: #ecfdf5;
        color: var(--success-color);
        border: 1px solid #a7f3d0;
    }
    .status-error {
        background: #fef2f2;
        color: var(--error-color);
        border: 1px solid #fecaca;
    }

    /* Output panel */
    .output-panel {
        margin-top: 2rem;
        padding: 1.5rem;
        background: white;
        border-radius: var(--radius-md);
        box-shadow: var(--shadow-md);
    }
    .output-panel .gr-textarea {
        border: 1px solid var(--border-color);
        border-radius: var(--radius-md);
        padding: 1rem;
        font-family: monospace;
        background: var(--background-light);
    }
    .copy-button {
        position: absolute;
        top: 0.5rem;
        right: 0.5rem;
        padding: 0.25rem 0.5rem;
        background: white;
        border: 1px solid var(--border-color);
        border-radius: var(--radius-md);
        cursor: pointer;
    }
    .copy-button:hover {
        background: var(--background-light);
    }

    /* Tooltips */
    .gr-form label span.info-icon {
        color: var(--text-secondary);
        font-size: 0.875rem;
        margin-left: 0.5rem;
    }
    """

    with gr.Blocks(
        title="Browser Use WebUI",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="indigo",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"]
        ),
        css=css
    ) as demo:
        gr.Markdown(
            """
            <div class="header">
                <h1>🌐 Browser Use WebUI</h1>
                <p>A powerful and intuitive tool for browser automation and task execution</p>
                <p style="font-size: 0.9rem; margin-top: 0.5rem;">Configure your agent, set up your browser, and automate your tasks with ease</p>
            </div>
            """
        )

        with gr.Tabs(elem_classes="tabs"):
            with gr.TabItem("🤖 Task Configuration", elem_classes="tab-content"):
                with gr.Group(elem_classes="gr-form"):
                    gr.Markdown("### Agent Configuration")
                    with gr.Row():
                        agent_type = gr.Radio(
                            ["org", "custom"],
                            label="Agent Type",
                            value="custom",
                            info="Select the type of agent to use for task execution",
                            elem_classes="agent-select"
                        )
                        max_steps = gr.Slider(
                            minimum=1,
                            maximum=200,
                            value=100,
                            step=1,
                            label="Maximum Steps",
                            info="Maximum number of steps the agent can take to complete the task",
                            elem_classes="step-slider"
                        )
                        use_vision = gr.Checkbox(
                            label="Enable Vision",
                            value=True,
                            info="Allow agent to process and understand visual information from the browser",
                            elem_classes="vision-toggle"
                        )

                with gr.Group(elem_classes="gr-form"):
                    gr.Markdown("### 🧠 Language Model Configuration")
                    with gr.Row():
                        llm_provider = gr.Dropdown(
                            ["anthropic", "openai", "gemini", "azure_openai", "deepseek", "ollama"],
                            label="LLM Provider",
                            value="gemini",
                            info="Choose your preferred Language Model provider",
                            elem_classes="llm-select"
                        )
                        llm_model_name = gr.Textbox(
                            label="Model Name",
                            value="gemini-2.0-flash-exp",
                            info="Specify the model version/name to use",
                            elem_classes="model-input"
                        )
                        llm_temperature = gr.Slider(
                            minimum=0.0,
                            maximum=2.0,
                            value=1.0,
                            step=0.1,
                            label="Temperature",
                            info="Adjust creativity vs determinism (higher = more creative)",
                            elem_classes="temp-slider"
                        )
                    
                    with gr.Row():
                        llm_base_url = gr.Textbox(
                            label="API Base URL",
                            placeholder="Enter your API endpoint URL (optional)",
                            info="Custom API endpoint for your LLM provider",
                            elem_classes="url-input"
                        )
                        llm_api_key = gr.Textbox(
                            label="API Key",
                            type="password",
                            placeholder="Enter your API key here",
                            info="Your LLM provider API key (stored securely)",
                            elem_classes="key-input"
                        )

            with gr.TabItem("🔧 Browser Settings", elem_classes="tab-content"):
                with gr.Group(elem_classes="gr-form"):
                    gr.Markdown("### Browser Configuration")
                    with gr.Row():
                        with gr.Column(scale=2):
                            use_own_browser = gr.Checkbox(
                                label="Use Own Browser",
                                value=False,
                                info="Use your local browser instance instead of a new one",
                                elem_classes="browser-toggle"
                            )
                            headless = gr.Checkbox(
                                label="Headless Mode",
                                value=False,
                                info="Run browser without visible window (faster execution)",
                                elem_classes="headless-toggle"
                            )
                            disable_security = gr.Checkbox(
                                label="Disable Security",
                                value=True,
                                info="⚠️ Disable browser security features (use with caution)",
                                elem_classes="security-toggle"
                            )
                        
                        with gr.Column(scale=3):
                            gr.Markdown("### Window Dimensions")
                            with gr.Row():
                                window_w = gr.Number(
                                    label="Width (px)",
                                    value=1920,
                                    info="Browser window width in pixels",
                                    elem_classes="dimension-input"
                                )
                                window_h = gr.Number(
                                    label="Height (px)",
                                    value=1080,
                                    info="Browser window height in pixels",
                                    elem_classes="dimension-input"
                                )
                    
                    save_recording_path = gr.Textbox(
                        label="Recording Path",
                        placeholder="e.g. ./tmp/record_videos",
                        value="./tmp/record_videos",
                        info="Directory path to save browser session recordings",
                        elem_classes="path-input"
                    )

            with gr.TabItem("📝 Task Definition", elem_classes="tab-content"):
                with gr.Group(elem_classes="gr-form"):
                    task = gr.Textbox(
                        label="Task Description",
                        lines=5,
                        placeholder="Describe what you want the agent to do...\nExample: Go to google.com, search for 'OpenAI', and return the first result URL",
                        value="go to google.com and type 'OpenAI' click search and give me the first url",
                        info="Provide clear instructions for what you want the agent to accomplish",
                        elem_classes="task-input"
                    )
                    add_infos = gr.Textbox(
                        label="Additional Information",
                        lines=3,
                        placeholder="Add any helpful context or hints for the agent...",
                        info="Optional: Provide extra context or specific instructions to help the agent",
                        elem_classes="hints-input"
                    )

        with gr.Row(elem_classes="action-buttons"):
            run_button = gr.Button("▶️ Start Task", variant="primary", scale=2, elem_classes="run-button")
            stop_button = gr.Button("⏹️ Stop Task", variant="stop", scale=1, elem_classes="stop-button")

        with gr.Group(elem_classes="output-panel"):
            gr.Markdown("### 📊 Task Results")
            with gr.Row():
                with gr.Column():
                    final_result_output = gr.TextArea(
                        label="Final Result",
                        lines=4,
                        show_copy_button=True,
                        elem_classes="result-output"
                    )
                    model_thoughts_output = gr.TextArea(
                        label="Agent Thoughts",
                        lines=4,
                        show_copy_button=True,
                        elem_classes="thoughts-output"
                    )
                with gr.Column():
                    model_actions_output = gr.TextArea(
                        label="Actions Taken",
                        lines=4,
                        show_copy_button=True,
                        elem_classes="actions-output"
                    )
                    errors_output = gr.TextArea(
                        label="Errors & Warnings",
                        lines=4,
                        show_copy_button=True,
                        elem_classes="errors-output"
                    )

            with gr.Row():
                status_output = gr.HTML(
                    value='<div class="status-indicator">Ready to start</div>',
                    label="Status",
                    elem_classes="status-display"
                )

        # Event handlers
        def update_status(is_running=True):
            if is_running:
                return '<div class="status-indicator status-success">Running...</div>'
            return '<div class="status-indicator">Ready</div>'

        async def run_task(*args):
            try:
                status = update_status(True)
                logger.info("Starting browser agent task")
                result = await run_browser_agent(*args)
                logger.info("Browser agent task completed")
                return [*result, update_status(False)]
            except Exception as e:
                import traceback
                error_msg = str(e) + "\n" + traceback.format_exc()
                logger.error(f"Error in browser agent task: {error_msg}")
                return ["", error_msg, "", "", '<div class="status-indicator status-error">Error</div>']

        # Connect the run button
        run_button.click(
            fn=run_task,
            inputs=[
                agent_type, llm_provider, llm_model_name, llm_temperature,
                llm_base_url, llm_api_key, use_own_browser, headless,
                disable_security, window_w, window_h, save_recording_path,
                task, add_infos, max_steps, use_vision
            ],
            outputs=[
                final_result_output, errors_output, 
                model_actions_output, model_thoughts_output,
                status_output
            ],
            api_name="run_agent"  # Enable API access
        )

        # Simple status update for stop button
        stop_button.click(
            fn=lambda: '<div class="status-indicator status-error">Stopped</div>',
            outputs=[status_output],
            api_name=False  # Disable API access for stop button
        )

    # Launch with queue enabled for async operation
    demo.queue().launch(
        server_name=args.ip,
        server_port=args.port,
        show_api=False,
        share=False
    )


if __name__ == '__main__':
    main()
