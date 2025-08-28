from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class MyCustomToolInput(BaseModel):
    """Input schema for MyCustomTool."""

    arg1: str = Field(description="The first number to sum.")
    arg2: str = Field(description="The second number to sum.")


class MyCustomTool(BaseTool):
    name: str = "Sum"
    description: str = (
        "Given two numbers, {arg1} and {arg2}, return their sum."
    )
    args_schema: Type[BaseModel] = MyCustomToolInput

    def _run(self, arg1: str, arg2: str) -> str:
        # Implementation goes here
        return str(int(arg1) + int(arg2))
