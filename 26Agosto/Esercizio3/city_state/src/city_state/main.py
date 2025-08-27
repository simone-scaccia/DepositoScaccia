from crewai.flow.flow import Flow, listen, start, router
from dotenv import load_dotenv
from litellm import completion

class ExampleFlow(Flow):
    model = "azure/gpt-4o"

    @start()
    def generate_city_or_country(self):
        print("Starting flow")
        print(f"Flow State ID: {self.state['id']}")  # ID univoco della run

        response = completion(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": "Return the name of a random city in the world OR a random country.",
                },
            ],
        )

        random_city_or_country = response["choices"][0]["message"]["content"]

        # Salvo la città nello stato del flow
        self.state["city_or_country"] = random_city_or_country
        print(f"Random City or Country: {random_city_or_country}")
        return random_city_or_country

    @listen(generate_city_or_country)
    def discriminator(self, random_city_or_country):
        response = completion(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": f"Is {random_city_or_country} a city or a country? 1 for city, 0 for country. Return only 0 or 1",
                },
            ],
        )

        print(f"Discriminator Response: {response['choices'][0]['message']['content']}")
        is_city = response["choices"][0]["message"]["content"] == "1"
        self.state["is_city"] = is_city
        return is_city

    @router(discriminator)
    def route_discriminator(self, is_city):
        if is_city:
            return "city"
        else:
            return "country"

    @listen("city")
    def generate_fun_fact(self):
        random_city = self.state.get("city_or_country")
        response = completion(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": f"Tell me a fun fact about {random_city}",
                },
            ],
        )

        fun_fact = response["choices"][0]["message"]["content"]
        self.state["fun_fact"] = fun_fact
        return fun_fact

    @listen("country")
    def generate_country_neighborhood(self):
        random_country = self.state.get("city_or_country")
        response = completion(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": f"Tell me the name of neighbors of {random_country}.",
                },
            ],
        )

        neighborhood_fact = response["choices"][0]["message"]["content"]
        self.state["neighborhood_fact"] = neighborhood_fact
        return neighborhood_fact


def kickoff():
    result = ExampleFlow().kickoff()
    print(f"{result}")
    plot()

def plot():
    flow = ExampleFlow()
    flow.plot()

if __name__ == "__main__":
    kickoff()