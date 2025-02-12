from pydantic import BaseModel

class CoordinatesAction(BaseModel):
    x: int
    y: int

