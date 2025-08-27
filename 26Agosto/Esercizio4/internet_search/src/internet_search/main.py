#!/usr/bin/env python
from random import randint

from pydantic import BaseModel

from crewai.flow import Flow, listen, start

from internet_search.crews.duckduckgo_search_crew.duckduckgo_search_crew import DuckduckgoSearchCrew


class SearchState(BaseModel):
    search_query: str = ""


class InternetSearchFlow(Flow[SearchState]):

    @start()
    def collect_search_query(self):
        self.state.search_query = input("Enter your search query: ")
        return self.state.search_query

    @listen(collect_search_query)
    def search_internet(self):
        print("Searching the internet with DuckDuckGo")
        result = DuckduckgoSearchCrew().crew().kickoff({
            "user_query": self.state.search_query
        })
        print("Search results:", result)


def kickoff():
    poem_flow = InternetSearchFlow()
    poem_flow.kickoff()
    plot()


def plot():
    poem_flow = InternetSearchFlow()
    poem_flow.plot()


if __name__ == "__main__":
    kickoff()
