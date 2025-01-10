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
import os

import gradio as gr
import argparse


from gradio.themes import Base, Default, Soft, Monochrome, Glass, Origin, Citrus, Ocean
import asyncio
import os, glob
from browser_use.agent.service import Agent
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import (
    BrowserContextConfig,
    BrowserContextWindowSize,
)
from playwright.async_api import async_playwright

from src.agent.custom_agent import CustomAgent
from src.agent.custom_prompts import CustomSystemPrompt
from src.agent.custom_massage_manager import CustomMassageManager
from src.browser.custom_browser import CustomBrowser
from src.browser.custom_context import BrowserContextConfig
from src.controller.custom_controller import CustomController
from src.utils import utils
from src.utils.utils import update_model_dropdown

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
        save_trace_path,
        enable_recording,
        task,
        add_infos,
        max_steps,
        use_vision,
        max_actions_per_step,
        tool_call_in_content,
        use_proxy
):
    # Configure proxy settings if enabled
    if use_proxy:
        proxy_url = os.getenv("PROXY_URL", "")
        if proxy_url:
            os.environ["http_proxy"] = proxy_url
            os.environ["https_proxy"] = proxy_url
    else:
        # Clear proxy settings if disabled
        os.environ.pop("http_proxy", None)
        os.environ.pop("https_proxy", None)

    # Disable recording if the checkbox is unchecked
    if not enable_recording:
        save_recording_path = None

    # Ensure the recording directory exists if recording is enabled
    if save_recording_path:
        os.makedirs(save_recording_path, exist_ok=True)

    # Get the list of existing videos before the agent runs
    existing_videos = set()
    if save_recording_path:
        existing_videos = set(
            glob.glob(os.path.join(save_recording_path, "*.[mM][pP]4"))
            + glob.glob(os.path.join(save_recording_path, "*.[wW][eE][bB][mM]"))
        )

    # Run the agent
    llm = utils.get_llm_model(
        provider=llm_provider,
        model_name=llm_model_name,
        temperature=llm_temperature,
        base_url=llm_base_url,
        api_key=llm_api_key,
    )
    if agent_type == "org":
        final_result, errors, model_actions, model_thoughts = await run_org_agent(
            llm=llm,
            headless=headless,
            disable_security=disable_security,
            window_w=window_w,
            window_h=window_h,
            save_recording_path=save_recording_path,
            save_trace_path=save_trace_path,
            task=task,
            max_steps=max_steps,
            use_vision=use_vision,
            max_actions_per_step=max_actions_per_step,
            tool_call_in_content=tool_call_in_content
        )
    elif agent_type == "custom":
        final_result, errors, model_actions, model_thoughts = await run_custom_agent(
            llm=llm,
            use_own_browser=use_own_browser,
            headless=headless,
            disable_security=disable_security,
            window_w=window_w,
            window_h=window_h,
            save_recording_path=save_recording_path,
            save_trace_path=save_trace_path,
            task=task,
            add_infos=add_infos,
            max_steps=max_steps,
            use_vision=use_vision,
            max_actions_per_step=max_actions_per_step,
            tool_call_in_content=tool_call_in_content,
            use_proxy=use_proxy
        )
    else:
        raise ValueError(f"Invalid agent type: {agent_type}")

    # Get the list of videos after the agent runs (if recording is enabled)
    latest_video = None
    if save_recording_path:
        new_videos = set(
            glob.glob(os.path.join(save_recording_path, "*.[mM][pP]4"))
            + glob.glob(os.path.join(save_recording_path, "*.[wW][eE][bB][mM]"))
        )
        if new_videos - existing_videos:
            latest_video = list(new_videos - existing_videos)[0]  # Get the first new video

    return final_result, errors, model_actions, model_thoughts, latest_video


async def run_org_agent(
        llm,
        headless,
        disable_security,
        window_w,
        window_h,
        save_recording_path,
        save_trace_path,
        task,
        max_steps,
        use_vision,
        max_actions_per_step,
        tool_call_in_content

):
    browser = Browser(
        config=BrowserConfig(
            headless=headless,
            disable_security=disable_security,
            extra_chromium_args=[f"--window-size={window_w},{window_h}"],
        )
    )
    async with await browser.new_context(
            config=BrowserContextConfig(
                trace_path=save_trace_path if save_trace_path else None,
                save_recording_path=save_recording_path if save_recording_path else None,
                no_viewport=False,
                browser_window_size=BrowserContextWindowSize(
                    width=window_w, height=window_h
                ),
            )
    ) as browser_context:
        agent = Agent(
            task=task,
            llm=llm,
            use_vision=use_vision,
            browser_context=browser_context,
            max_actions_per_step=max_actions_per_step,
            tool_call_in_content=tool_call_in_content
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
        save_trace_path,
        task,
        add_infos,
        max_steps,
        use_vision,
        max_actions_per_step,
        tool_call_in_content,
        use_proxy
):
    controller = CustomController()
    playwright = None
    browser_context_ = None
    try:
        if use_own_browser:
            playwright = await async_playwright().start()
            chrome_exe = os.getenv("CHROME_PATH", "")
            chrome_use_data = os.getenv("CHROME_USER_DATA", "")

            if chrome_exe == "":
                chrome_exe = None
            elif not os.path.exists(chrome_exe):
                raise ValueError(f"Chrome executable not found at {chrome_exe}")
            
            if chrome_use_data == "":
                chrome_use_data = None

            # Configure launch arguments with proxy support
            launch_args = {
                "user_data_dir": chrome_use_data,
                "executable_path": chrome_exe,
                "no_viewport": False,
                "headless": headless,
                "user_agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36"
                ),
                "java_script_enabled": True,
                "bypass_csp": disable_security,
                "ignore_https_errors": disable_security,
                "record_video_dir": save_recording_path if save_recording_path else None,
                "record_video_size": {"width": window_w, "height": window_h},
            }

            # Add proxy configuration if enabled
            if use_proxy:
                proxy_url = os.getenv("PROXY_URL", "")
                if proxy_url:
                    launch_args["proxy"] = {
                        "server": proxy_url
                    }

            browser_context_ = await playwright.chromium.launch_persistent_context(**launch_args)
        else:
            browser_context_ = None

        browser = CustomBrowser(
            config=BrowserConfig(
                headless=headless,
                disable_security=disable_security,
                extra_chromium_args=[f"--window-size={window_w},{window_h}"],
            )
        )
        async with await browser.new_context(
                config=BrowserContextConfig(
                    trace_path=save_trace_path if save_trace_path else None,
                    save_recording_path=save_recording_path
                    if save_recording_path
                    else None,
                    no_viewport=False,
                    browser_window_size=BrowserContextWindowSize(
                        width=window_w, height=window_h
                    ),
                ),
                context=browser_context_,
        ) as browser_context:
            # Create message manager instance
            message_manager = CustomMassageManager(
                llm=llm,
                task=task,
                action_descriptions=controller.registry.get_prompt_description(),
                system_prompt_class=CustomSystemPrompt,
                max_actions_per_step=max_actions_per_step,
                tool_call_in_content=tool_call_in_content
            )
            
            agent = CustomAgent(
                task=task,
                add_infos=add_infos,
                use_vision=use_vision,
                llm=llm,
                browser_context=browser_context,
                controller=controller,
                system_prompt_class=CustomSystemPrompt,
                max_actions_per_step=max_actions_per_step,
                message_manager=message_manager
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
        # Explicitly close the persistent context
        if browser_context_:
            await browser_context_.close()

        # Close the Playwright instance
        if playwright:
            await playwright.stop()
        await browser.close()
    return final_result, errors, model_actions, model_thoughts

# Define the theme map globally
theme_map = {
    "Default": Default(),
    "Soft": Soft(),
    "Monochrome": Monochrome(),
    "Glass": Glass(),
    "Origin": Origin(),
    "Citrus": Citrus(),
    "Ocean": Ocean(),
    "Base": Base()
}


def create_ui(theme_name="Ocean"):
    css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
        padding-top: 20px !important;
    }
    .header-text {
        text-align: center;
        margin-bottom: 30px;
    }
    .theme-section {
        margin-bottom: 20px;
        padding: 15px;
        border-radius: 10px;
    }
    """

    js = """
    function refresh() {
        const url = new URL(window.location);
        if (url.searchParams.get('__theme') !== 'dark') {
            url.searchParams.set('__theme', 'dark');
            window.location.href = url.href;
        }
    }
    """

    with gr.Blocks(
            title="Browser Use WebUI", theme=theme_map[theme_name], css=css, js=js
    ) as demo:
        with gr.Row():
            gr.Markdown(
                """
                # 🌐 Browser Use WebUI
                ### Control your browser with AI assistance
                """,
                elem_classes=["header-text"],
            )

        with gr.Tabs() as tabs:
            with gr.TabItem("⚙️ Agent Settings", id=1):
                with gr.Group():
                    agent_type = gr.Radio(
                        ["org", "custom"],
                        label="Agent Type",
                        value="custom",
                        info="Select the type of agent to use",
                    )
                    max_steps = gr.Slider(
                        minimum=1,
                        maximum=200,
                        value=100,
                        step=1,
                        label="Max Run Steps",
                        info="Maximum number of steps the agent will take",
                    )
                    max_actions_per_step = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=10,
                        step=1,
                        label="Max Actions per Step",
                        info="Maximum number of actions the agent will take per step",
                    )
                    use_vision = gr.Checkbox(
                        label="Use Vision",
                        value=True,
                        info="Enable visual processing capabilities",
                    )
                    tool_call_in_content = gr.Checkbox(
                        label="Use Tool Calls in Content",
                        value=True,
                        info="Enable Tool Calls in content",
                    )

            with gr.TabItem("🔧 LLM Configuration", id=2):
                with gr.Group():
                    llm_provider = gr.Dropdown(
                        ["anthropic", "openai", "deepseek", "gemini", "ollama", "azure_openai"],
                        label="LLM Provider",
                        value="",
                        info="Select your preferred language model provider"
                    )
                    llm_model_name = gr.Dropdown(
                        label="Model Name",
                        value="",
                        interactive=True,
                        allow_custom_value=True,  # Allow users to input custom model names
                        info="Select a model from the dropdown or type a custom model name"
                    )
                    llm_temperature = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Temperature",
                        info="Controls randomness in model outputs"
                    )
                    with gr.Row():
                        llm_base_url = gr.Textbox(
                            label="Base URL",
                            value=os.getenv(f"{llm_provider.value.upper()}_BASE_URL ", ""),  # Default to .env value
                            info="API endpoint URL (if required)"
                        )
                        llm_api_key = gr.Textbox(
                            label="API Key",
                            type="password",
                            value=os.getenv(f"{llm_provider.value.upper()}_API_KEY", ""),  # Default to .env value
                            info="Your API key (leave blank to use .env)"
                        )
                    
                    # Proxy configuration
                    use_proxy = gr.Checkbox(
                        label="Use Proxy",
                        value=os.getenv("USE_PROXY", "false").lower() == "true",
                        info="Enable proxy for API requests and browser (proxy URL is configured in .env file)"
                    )

            with gr.TabItem("🌐 Browser Settings", id=3):
                with gr.Group():
                    with gr.Row():
                        use_own_browser = gr.Checkbox(
                            label="Use Own Browser",
                            value=False,
                            info="Use your existing browser instance",
                        )
                        headless = gr.Checkbox(
                            label="Headless Mode",
                            value=False,
                            info="Run browser without GUI",
                        )
                        disable_security = gr.Checkbox(
                            label="Disable Security",
                            value=True,
                            info="Disable browser security features",
                        )
                        enable_recording = gr.Checkbox(
                            label="Enable Recording",
                            value=True,
                            info="Enable saving browser recordings",
                        )

                    with gr.Row():
                        window_w = gr.Number(
                            label="Window Width",
                            value=1280,
                            info="Browser window width",
                        )
                        window_h = gr.Number(
                            label="Window Height",
                            value=1100,
                            info="Browser window height",
                        )

                    save_recording_path = gr.Textbox(
                        label="Recording Path",
                        placeholder="e.g. ./tmp/record_videos",
                        value="./tmp/record_videos",
                        info="Path to save browser recordings",
                        interactive=True,  # Allow editing only if recording is enabled
                    )

                    save_trace_path = gr.Textbox(
                        label="Trace Path",
                        placeholder="e.g. ./tmp/traces",
                        value="./tmp/traces",
                        info="Path to save Agent traces",
                        interactive=True,
                    )

            with gr.TabItem("🤖 Run Agent", id=4):
                task = gr.Textbox(
                    label="Task Description",
                    lines=4,
                    placeholder="Enter your task here...",
                    value="go to google.com and type 'OpenAI' click search and give me the first url",
                    info="Describe what you want the agent to do",
                )
                add_infos = gr.Textbox(
                    label="Additional Information",
                    lines=3,
                    placeholder="Add any helpful context or instructions...",
                    info="Optional hints to help the LLM complete the task",
                )

                with gr.Row():
                    run_button = gr.Button("▶️ Run Agent", variant="primary", scale=2)
                    stop_button = gr.Button("⏹️ Stop", variant="stop", scale=1)

            with gr.TabItem("📊 Results", id=5):
                recording_display = gr.Video(label="Latest Recording")

                with gr.Group():
                    gr.Markdown("### Results")
                    with gr.Row():
                        with gr.Column():
                            final_result_output = gr.Textbox(
                                label="Final Result", lines=3, show_label=True
                            )
                        with gr.Column():
                            errors_output = gr.Textbox(
                                label="Errors", lines=3, show_label=True
                            )
                    with gr.Row():
                        with gr.Column():
                            model_actions_output = gr.Textbox(
                                label="Model Actions", lines=3, show_label=True
                            )
                        with gr.Column():
                            model_thoughts_output = gr.Textbox(
                                label="Model Thoughts", lines=3, show_label=True
                            )
            
            with gr.TabItem("🎥 Recordings", id=6):
                def list_recordings(save_recording_path):
                    if not os.path.exists(save_recording_path):
                        return []
                    
                    # Get all video files
                    recordings = glob.glob(os.path.join(save_recording_path, "*.[mM][pP]4")) + glob.glob(os.path.join(save_recording_path, "*.[wW][eE][bB][mM]"))
                    
                    # Sort recordings by creation time (oldest first)
                    recordings.sort(key=os.path.getctime)
                    
                    # Add numbering to the recordings
                    numbered_recordings = []
                    for idx, recording in enumerate(recordings, start=1):
                        filename = os.path.basename(recording)
                        numbered_recordings.append((recording, f"{idx}. {filename}"))
                    
                    return numbered_recordings

                recordings_gallery = gr.Gallery(
                    label="Recordings",
                    value=list_recordings("./tmp/record_videos"),
                    columns=3,
                    height="auto",
                    object_fit="contain"
                )

                refresh_button = gr.Button("🔄 Refresh Recordings", variant="secondary")
                refresh_button.click(
                    fn=list_recordings,
                    inputs=save_recording_path,
                    outputs=recordings_gallery
                )

        # Attach the callback to the LLM provider dropdown
        llm_provider.change(
            lambda provider, api_key, base_url: update_model_dropdown(provider, api_key, base_url),
            inputs=[llm_provider, llm_api_key, llm_base_url],
            outputs=llm_model_name
        )  

        # Add this after defining the components
        enable_recording.change(
            lambda enabled: gr.update(interactive=enabled),
            inputs=enable_recording,
            outputs=save_recording_path
        )
          
        # Run button click handler
        run_button.click(
            fn=run_browser_agent,
            inputs=[
                agent_type, llm_provider, llm_model_name, llm_temperature, llm_base_url, llm_api_key,
                use_own_browser, headless, disable_security, window_w, window_h, save_recording_path, save_trace_path,
                enable_recording, task, add_infos, max_steps, use_vision, max_actions_per_step, tool_call_in_content,
                use_proxy
            ],
            outputs=[final_result_output, errors_output, model_actions_output, model_thoughts_output, recording_display],
        )

    return demo

def main():
    parser = argparse.ArgumentParser(description="Gradio UI for Browser Agent")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="IP address to bind to")
    parser.add_argument("--port", type=int, default=7788, help="Port to listen on")
    parser.add_argument("--theme", type=str, default="Ocean", choices=theme_map.keys(), help="Theme to use for the UI")
    parser.add_argument("--dark-mode", action="store_true", help="Enable dark mode")
    args = parser.parse_args()

    demo = create_ui(theme_name=args.theme)
    demo.launch(server_name=args.ip, server_port=args.port)

if __name__ == '__main__':
    main()
