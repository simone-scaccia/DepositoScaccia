#!/usr/bin/env python
"""Application entry-point and flow definition for the RagMed demo.

This module defines the `RagMed` flow which routes a user query to one of two
crews based on an LLM's classification: a web search summarizer crew or a RAG
creator crew. It also exposes convenience functions `kickoff` and `plot` to
run and visualize the flow.

Examples:
    Create the flow and access its state class:

    >>> from rag_med.main import RagMed, SearchState  # doctest: +SKIP
    >>> flow = RagMed()  # doctest: +SKIP
    >>> isinstance(flow.state, SearchState)  # doctest: +SKIP
    True
"""
import json

from pydantic import BaseModel, Field

from crewai.flow import Flow, listen, start, router, or_
from crewai import LLM

from rag_med.crews.search_summarize.search_summarize import SearchSummarizeCrew
from rag_med.crews.rag_creator.rag_creator import RagCreatorCrew


class LLMResponseTask(BaseModel):
    """Structured response from the LLM menu classifier.

    Attributes:
        response (str): One word among "Ricerca", "Rag", or "Uscita".

    Examples:
        >>> LLMResponseTask(response="Ricerca").response
        'Ricerca'
    """
    response: str = Field(description="LLM response as a word: Ricerca, Rag o Uscita")

class SearchState(BaseModel):
    """State container for the `RagMed` flow.

    Attributes:
        search_query (str): The current search query typed by the user.
        response (LLMResponseTask | None): The last LLM classification result.
        question (str): The original user question or topic.

    Examples:
        >>> state = SearchState(question="Cos'e' RAG?")
        >>> state.question
        "Cos'e' RAG?"
    """
    search_query: str = ""
    response: LLMResponseTask = None
    question: str = ""


class RagMed(Flow[SearchState]):
    """Interactive flow that routes to search or RAG based on LLM output.

    The flow starts by asking the user a question, then calls an LLM to decide
    whether to invoke internet search, a RAG demo, or exit.
    """

    @start("restart_flow")
    def init_menu(self):
        """Start node that asks the user for a topic and classifies intent.

        Prompts for input on the console, calls the configured LLM to classify
        the intent into one of "Ricerca", "Rag", or "Uscita", and stores the
        result in the flow state.

        Raises:
            ValueError: If the LLM returns an unexpected payload.
        """

        question = "Quale argomento vuoi cercare?"
        self.state.question = input(question)

        # Initialize the LLM
        llm = LLM(model="azure/gpt-4o", response_format=LLMResponseTask)

        # Create the messages for the outline
        messages = [
            {"role": "system", "content": "Sei un assistente utile progettato per restituire una sola parola tra: Ricerca, Rag, Uscita."},
            {"role": "user", "content": f"""
                Considerando l'argomento fornito dall'utente: "{self.state.question}", comprendi l'obiettivo dell'utente e scegli una delle seguenti opzioni:

                1 - Rag: Se l'utente sta facendo domande sugli agenti RAG (Retrieval-Augmented Generation)
                2 - Uscita: Se l'utente vuole uscire dall'applicazione
                3 - Ricerca: Se l'utente sta facendo domande su altri argomenti
                Rispondi con **una sola parola**: Rag, Uscita oppure Ricerca.
            """}
        ]


        # Make the LLM call with JSON response format
        response = llm.call(messages=messages)
        response = json.loads(response)
        self.state.response = LLMResponseTask(**response)
        print("Risposta LLM: ", self.state.question)


    @router(init_menu)
    def route_init_menu(self):
        """Router that decides the next step based on LLM classification.

        Returns:
            str: The name of the next node: "search_internet", "search_rag", or
            "exit".
        """
        if self.state.response.response == "Ricerca":
            return "search_internet"
        elif self.state.response.response == "Rag":
            return "search_rag"
        elif self.state.response.response == "Uscita":
            print("Uscita...")
            return "exit"

    @start("search_rag")
    def rag_crew(self):
        """Starts the RAG creator crew with the current question.

        Returns:
            None
        """
        RagCreatorCrew().crew().kickoff(inputs={
            "question": self.state.question
        })

    @listen("search_internet")
    def search_crew(self):
        """Starts the internet search+summarize crew with the current topic.

        Returns:
            None
        """
        print("Searching the internet with DuckDuckGo")
        SearchSummarizeCrew().crew().kickoff(inputs={
            "topic": self.state.question
        })

    @router(or_(search_crew, rag_crew))
    def restart(self):
        """Router that loops the flow back to the start.

        Returns:
            str: The name of the start node.
        """
        return "restart_flow"


def kickoff():
    """Run the `RagMed` flow end-to-end in interactive mode.

    Returns:
        None

    Examples:
        >>> # Launch the interactive flow (requires user input)  
        >>> # doctest: +SKIP
        >>> kickoff()
    """
    flow = RagMed()
    flow.kickoff()
    plot()


def plot():
    """Render the flow graph using the built-in plotting utility.

    Returns:
        None

    Examples:
        >>> # Generate a diagram of the flow  
        >>> # doctest: +SKIP
        >>> plot()
    """
    flow = RagMed()
    flow.plot()


if __name__ == "__main__":
    kickoff()
