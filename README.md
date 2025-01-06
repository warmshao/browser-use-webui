# Browser-Use WebUI

## Background

This project builds upon the foundation of the [browser-use](https://github.com/browser-use/browser-use), which is designed to make websites accessible for AI agents. We have enhanced the original capabilities by providing:

1.  **A Brand New WebUI:** We offer a comprehensive web interface that supports a wide range of `browser-use` functionalities. This UI is designed to be user-friendly and enables easy interaction with the browser agent.

2.  **Expanded LLM Support:** We've integrated support for various Large Language Models (LLMs), including: Gemini, OpenAI, Azure OpenAI, Anthropic, DeepSeek, Ollama etc. And we plan to add support for even more models in the future.

3.  **Custom Browser Support:** You can use your own browser with our tool, eliminating the need to re-login to sites or deal with other authentication challenges. This feature also supports high-definition screen recording.

4.  **Customized Agent:** We've implemented a custom agent that enhances `browser-use` with Optimized prompts.

5.  **Persistent Browser Sessions:** You can choose to keep the browser window open between AI tasks, allowing you to see the complete history and state of AI interactions.

<video src="https://github.com/user-attachments/assets/58c0f59e-02b4-4413-aba8-6184616bf181" controls="controls" width="500" height="300" >Your browser does not support playing this video!</video>

**Changelog**
- [x] **2025/01/06:** Thanks to @richard-devbot, a New and Well-Designed WebUI is released. [Video tutorial demo](https://github.com/warmshao/browser-use-webui/issues/1#issuecomment-2573393113).


## Installation Options

### Option 1: Docker Installation (Recommended)

1. **Prerequisites:**
   - Docker and Docker Compose installed on your system
   - Git to clone the repository

2. **Setup:**
   ```bash
   # Clone the repository
   git clone <repository-url>
   cd browser-use-webui

   # Copy and configure environment variables
   cp .env.example .env
   # Edit .env with your preferred text editor and add your API keys
   ```

3. **Run with Docker:**
   ```bash
   # Build and start the container with default settings (browser closes after AI tasks)
   docker compose up --build

   # Or run with persistent browser (browser stays open between AI tasks)
   CHROME_PERSISTENT_SESSION=true docker compose up --build
   ```

4. **Access the Application:**
   - WebUI: `http://localhost:7788`
   - VNC Viewer (to see browser interactions): `http://localhost:6080/vnc.html`
   
   Default VNC password is "vncpassword". You can change it by setting the `VNC_PASSWORD` environment variable in your `.env` file.

### Option 2: Local Installation

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

### Docker Setup
1. **Environment Variables:**
   - All configuration is done through the `.env` file
   - Available environment variables:
     ```
     # LLM API Keys
     OPENAI_API_KEY=your_key_here
     ANTHROPIC_API_KEY=your_key_here
     GOOGLE_API_KEY=your_key_here

     # Browser Settings
     CHROME_PERSISTENT_SESSION=true   # Set to true to keep browser open between AI tasks
     RESOLUTION=1920x1080x24         # Custom resolution format: WIDTHxHEIGHTxDEPTH
     RESOLUTION_WIDTH=1920           # Custom width in pixels
     RESOLUTION_HEIGHT=1080          # Custom height in pixels

     # VNC Settings
     VNC_PASSWORD=your_vnc_password  # Optional, defaults to "vncpassword"
     ```

2. **Browser Persistence Modes:**
   - **Default Mode (CHROME_PERSISTENT_SESSION=false):**
     - Browser opens and closes with each AI task
     - Clean state for each interaction
     - Lower resource usage

   - **Persistent Mode (CHROME_PERSISTENT_SESSION=true):**
     - Browser stays open between AI tasks
     - Maintains history and state
     - Allows viewing previous AI interactions
     - Set in `.env` file or via environment variable when starting container

3. **Viewing Browser Interactions:**
   - Access the noVNC viewer at `http://localhost:6080/vnc.html`
   - Enter the VNC password (default: "vncpassword" or what you set in VNC_PASSWORD)
   - You can now see all browser interactions in real-time

4. **Container Management:**
   ```bash
   # Start with persistent browser
   CHROME_PERSISTENT_SESSION=true docker compose up -d

   # Start with default mode (browser closes after tasks)
   docker compose up -d

   # View logs
   docker compose logs -f

   # Stop the container
   docker compose down
   ```

### Local Setup
1.  **Run the WebUI:**
    ```bash
    python webui.py --ip 127.0.0.1 --port 7788
    ```
2.  **Access the WebUI:** Open your web browser and navigate to `http://127.0.0.1:7788`.
3.  **Using Your Own Browser:**
    - Close all chrome windows
    - Open the WebUI in a non-Chrome browser, such as Firefox or Edge. This is important because the persistent browser context will use the Chrome data when running the agent.
    - Check the "Use Own Browser" option within the Browser Settings.