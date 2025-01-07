# Browser-Use WebUI

## Background

This project builds upon the foundation of the [browser-use](https://github.com/browser-use/browser-use), which is designed to make websites accessible for AI agents. We have enhanced the original capabilities by providing:

1.  **A Brand New WebUI:** We offer a comprehensive web interface that supports a wide range of `browser-use` functionalities. This UI is designed to be user-friendly and enables easy interaction with the browser agent.

2.  **Expanded LLM Support:** We've integrated support for various Large Language Models (LLMs), including: Gemini, OpenAI, Azure OpenAI, Anthropic, DeepSeek, Ollama etc. And we plan to add support for even more models in the future.

3.  **Custom Browser Support:** You can use your own browser with our tool, eliminating the need to re-login to sites or deal with other authentication challenges. This feature also supports high-definition screen recording.

4.  **Customized Agent:** We've implemented a custom agent that enhances `browser-use` with Optimized prompts.

<video src="assets/examples/Recording_compressed.mp4" controls="controls" width="500" height="300" >Your browser does not support playing this video!</video>

**Changelog**
- [x] **2025/01/06:** Thanks to @richard-devbot, a New and Well-Designed WebUI is released. [Video tutorial demo](https://github.com/warmshao/browser-use-webui/issues/1#issuecomment-2573393113).


## Environment Installation

1.  **Python Version:** Ensure you have Python 3.11 or higher installed.
2.  **Install `browser-use`:**
    ```bash
    pip install browser-use
    ```
3.  **Install Playwright:**
    ```bash
    playwright install
    ```
4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
5.  **Configure Environment Variables:**
    - Copy `.env.example` to `.env` and set your environment variables, including API keys for the LLM.
    - **If using your own browser:**
      - Set `CHROME_PATH` to the executable path of your browser (e.g., `C:\Program Files\Google\Chrome\Application\chrome.exe` on Windows).
      - Set `CHROME_USER_DATA` to the user data directory of your browser (e.g.,`C:\Users\<YourUsername>\AppData\Local\Google\Chrome\User Data`).

## Usage

1.  **Run the WebUI:**
    ```bash
    python webui.py --ip 127.0.0.1 --port 7788
    ```
2.  **Access the WebUI:** Open your web browser and navigate to `http://127.0.0.1:7788`.
3.  **Using Your Own Browser:**
    - Close all chrome windows
    - Open the WebUI in a non-Chrome browser, such as Firefox or Edge. This is important because the persistent browser context will use the Chrome data when running the agent.
    - Check the "Use Own Browser" option within the Browser Settings.
