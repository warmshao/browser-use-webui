from pydantic import BaseModel, Field
from typing import Literal

class CoordinatesAction(BaseModel):
    x: int
    y: int

class ClickAction(BaseModel):
    x: int = Field(..., description="X coordinate on the page")
    y: int = Field(..., description="Y coordinate on the page")
    button: Literal["left", "right", "middle"] = "left"
