# -*- coding: utf-8 -*-
# @Time    : 2025/1/6
# @Author  : wenshao
# @ProjectName: browser-use-webui
# @FileName: config.py

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class BrowserPersistenceConfig:
    """Configuration for browser persistence"""

    persistent_session: bool = False
    user_data_dir: Optional[str] = None
    debugging_port: Optional[int] = None
    debugging_host: Optional[str] = None

    @classmethod
    def from_env(cls) -> "BrowserPersistenceConfig":
        """Create config from environment variables"""
        return cls(
            persistent_session=os.getenv("CHROME_PERSISTENT_SESSION", "").lower()
            == "true",
            user_data_dir=os.getenv("CHROME_USER_DATA"),
            debugging_port=int(os.getenv("CHROME_DEBUGGING_PORT", "9222")),
            debugging_host=os.getenv("CHROME_DEBUGGING_HOST", "localhost"),
        )