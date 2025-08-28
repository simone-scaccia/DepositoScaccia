from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from rag_med.tools import rag_faiss_lmstudio


class MyCustomToolInput(BaseModel):
    """Input schema for MyCustomTool."""

    question: str = Field(..., description="Description of the question to ask.")


class MyCustomTool(BaseTool):
    name: str = "RAG"
    description: str = (
        "RAG tool containing relevant information for retrieval-augmented generation."
    )
    args_schema: Type[BaseModel] = MyCustomToolInput
    

    def _run(self, question: str) -> str:
        # Implementation goes here
        return rag_faiss_lmstudio.rag_answer(question, rag_faiss_lmstudio.setup())
