import pdb
from dotenv import load_dotenv

load_dotenv()


def test_connect_browser():
    import os
    from playwright.sync_api import sync_playwright

    chrome_exe = os.getenv("CHROME_PATH", "")
    chrome_use_data = os.getenv("CHROME_USER_DATA", "")

    with sync_playwright() as p:
        browser = p.chromium.launch_persistent_context(
            user_data_dir=chrome_use_data,
            executable_path=chrome_exe,
            headless=False  # Keep browser window visible
        )

        page = browser.new_page()
        page.goto("https://mail.google.com/mail/u/0/#inbox")
        page.wait_for_load_state()

        input("Press the Enter key to close the browser...")

        browser.close()


if __name__ == '__main__':
    test_connect_browser()
