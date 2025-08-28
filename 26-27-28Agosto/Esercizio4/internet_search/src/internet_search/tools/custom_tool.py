from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from ddgs import DDGS


class MyCustomToolInput(BaseModel):
    """Input schema for MyCustomTool."""

    argument: str = Field(..., description="Description of the argument.")
    n: int = Field(3, description="Number of results to return.")

class MyCustomToolOutput(BaseModel):
    """Output schema for MyCustomTool."""
    results: list[dict] = Field(..., description="List of search results.")

class MyCustomTool(BaseTool):
    name: str = "DuckDuckGo Search"
    description: str = (
        "Given a text, search on internet using DuckDuckGo"
    )
    args_schema: Type[BaseModel] = MyCustomToolInput

    def _run(self, argument: str, n: int) -> str:
        with DDGS(verify=False) as ddgs:
            results = list(ddgs.text(argument, region="it-it", safesearch="off", max_results=n))
            for i, result in enumerate(results):
                print(f"Result {i + 1}: {result['body']}")
            return MyCustomToolOutput(results=results).json()

            
