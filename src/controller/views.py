from pydantic import BaseModel

class MouseMoveAction(BaseModel):
    x: int
    y: int