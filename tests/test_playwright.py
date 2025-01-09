# -*- coding: utf-8 -*-
# @Time    : 2025/1/2
# @Author  : wenshao
# @Email   : wenshaoguo1026@gmail.com
# @Project : browser-use-webui
# @FileName: test_playwright.py
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
            headless=False  # 保持浏览器窗口可见
        )

        page = browser.new_page()
        page.goto("https://mail.google.com/mail/u/0/#inbox")
        page.wait_for_load_state()

        input("按下回车键以关闭浏览器...")

        browser.close()


if __name__ == '__main__':
    test_connect_browser()
