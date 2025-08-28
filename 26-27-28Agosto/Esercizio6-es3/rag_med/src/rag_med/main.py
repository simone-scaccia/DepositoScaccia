#!/usr/bin/env python
import json

from pydantic import BaseModel, Field

from crewai.flow import Flow, listen, start, router, or_
from crewai import LLM

from rag_med.crews.search_summarize.search_summarize import SearchSummarizeCrew
from rag_med.crews.rag_creator.rag_creator import RagCreatorCrew


class LLMResponseTask(BaseModel):
    response: str = Field(description="LLM response as a word: Ricerca, Rag o Uscita")

class SearchState(BaseModel):
    search_query: str = ""
    response: LLMResponseTask = None
    question: str = ""


class RagMed(Flow[SearchState]):

    @start("restart_flow")
    def init_menu(self):

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
        if self.state.response.response == "Ricerca":
            return "search_internet"
        elif self.state.response.response == "Rag":
            return "search_rag"
        elif self.state.response.response == "Uscita":
            print("Uscita...")
            return "exit"

    @start("search_rag")
    def rag_crew(self):
        RagCreatorCrew().crew().kickoff(inputs={
            "question": self.state.question
        })

    @listen("search_internet")
    def search_crew(self):
        print("Searching the internet with DuckDuckGo")
        SearchSummarizeCrew().crew().kickoff(inputs={
            "topic": self.state.question
        })

    @router(or_(search_crew, rag_crew))
    def restart(self):
        return "restart_flow"


def kickoff():
    flow = RagMed()
    flow.kickoff()
    plot()


def plot():
    flow = RagMed()
    flow.plot()


if __name__ == "__main__":
    kickoff()
