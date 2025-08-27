#!/usr/bin/env python
from random import randint

from pydantic import BaseModel

from crewai.flow import Flow, listen, start

from ricerca_o_calcolo.crews.search_summarize.search_summarize import SearchSummarizeCrew


class SearchState(BaseModel):
    search_query: str = ""


class RicercaOCalcoloFlow(Flow[SearchState]):

    @start()
    def init_menu(self):
        self.state.num_task = input("1 - Ricerca web\n2 - Somma\nScegli un'opzione: ")
        if self.state.num_task == "1":
            return "search_internet"
        elif self.state.num_task == "2":
            return "sum_numbers"
        else:
            print("Opzione non valida. Riprova.")
            self.init_menu()

    @listen("search_internet")
    def collect_search_query(self):
        self.state.search_query = input("Enter your search query: ")
        return self.state.search_query

    @listen(collect_search_query)
    def search_internet(self):
        print("Searching the internet with DuckDuckGo")
        results = SearchSummarizeCrew().crew().kickoff(inputs={
            "topic": self.state.search_query
        })
        print("Search results:")
        # Accessing the crew output
        print(results)

    @listen("sum_numbers")
    def sum_two_numbers(self):
        pass

def kickoff():
    poem_flow = RicercaOCalcoloFlow()
    poem_flow.kickoff()
    plot()


def plot():
    poem_flow = RicercaOCalcoloFlow()
    poem_flow.plot()


if __name__ == "__main__":
    kickoff()
