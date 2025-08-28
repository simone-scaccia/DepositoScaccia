#!/usr/bin/env python
import json
from random import randint

from pydantic import BaseModel, Field

from crewai.flow import Flow, listen, start, router, or_
from crewai import LLM

from ricerca_o_calcolo.crews.search_summarize.search_summarize import SearchSummarizeCrew
from ricerca_o_calcolo.crews.sum.sum import SumCrew

class LLMResponseNumbers(BaseModel):
    numbers: list[int] = Field(description="List of numbers to sum")

class LLMResponseTask(BaseModel):
    response: str = Field(description="LLM response as a word: Ricerca, Somma o Uscita")

class SearchState(BaseModel):
    search_query: str = ""
    task: str = ""
    response: LLMResponseTask = None
    numbers: LLMResponseNumbers = None


class RicercaOCalcoloFlow(Flow[SearchState]):

    @start("restart_flow")
    def init_menu(self):

        question = "Vuoi fare una ricerca web o una somma?"
        user_response = input(question)

        # Initialize the LLM
        llm = LLM(model="azure/gpt-4o", response_format=LLMResponseTask)

        # Create the messages for the outline
        messages = [
            {"role": "system", "content": "You are a helpful assistant designed to output a single word between: Ricerca, Somma, Uscita, Richiedi."},
            {"role": "user", "content": f"""
                Resume the user response: "{user_response}" in one word between: Ricerca, Somma, Uscita, Richiedi.

                The meaning of the response is the following:
                1 - Ricerca: Search the web
                2 - Somma: Sum two numbers
                3 - Uscita: Exit the application
                4 - Richiedi: The user response is not coherent with the question
            """}
        ]

        # Make the LLM call with JSON response format
        response = llm.call(messages=messages)
        response = json.loads(response)
        self.state.response = LLMResponseTask(**response)


    @router(init_menu)
    def route_init_menu(self):
        if self.state.response.response == "Ricerca":
            self.state.task = "1"
            return "search_internet"
        elif self.state.response.response == "Somma":
            self.state.task = "2"
            return "sum_numbers"
        elif self.state.response.response == "Richiedi":
            print("Richiedi...")
            return "restart_flow"
        elif self.state.response.response == "Uscita":
            print("Uscita...")
            return "exit"

    @listen("search_internet")
    def collect_search_query(self):
        self.state.search_query = input("Enter your search query: ")
        return self.state.search_query

    @listen(collect_search_query)
    def search_crew(self):
        print("Searching the internet with DuckDuckGo")
        SearchSummarizeCrew().crew().kickoff(inputs={
            "topic": self.state.search_query
        })

    @listen("sum_numbers")
    def sum_two_numbers(self):
        question = "Quali numeri vuoi sommare?"
        user_response = input(question)

        # Initialize the LLM
        llm = LLM(model="azure/gpt-4o", response_format=LLMResponseNumbers)

        # Create the messages for the outline
        messages = [
            {"role": "system", "content": "You are a helpful assistant designed to output the numbers to sum."},
            {"role": "user", "content": f"""
                Extract the numbers to sum from the user response: "{user_response}" as a list.

                The output should be a list of numbers.
            """}
        ]

        # Make the LLM call with JSON response format
        response = llm.call(messages=messages)
        response = json.loads(response).get("numbers", [])
        self.state.numbers = LLMResponseNumbers(numbers=response)

    @listen(sum_two_numbers)
    def sum_two_numbers_crews(self):
        print("Summing numbers:", self.state.numbers.numbers[0], "+", self.state.numbers.numbers[1])
        SumCrew().crew().kickoff(inputs={
            "arg1": self.state.numbers.numbers[0],
            "arg2": self.state.numbers.numbers[1]
        })

    @router(or_(sum_two_numbers_crews, search_crew))
    def restart(self):
        self.state = SearchState()
        return "restart_flow"


def kickoff():
    flow = RicercaOCalcoloFlow()
    flow.kickoff()
    plot()


def plot():
    flow = RicercaOCalcoloFlow()
    flow.plot()


if __name__ == "__main__":
    kickoff()
