from dotenv import load_dotenv
import os
from openai import AzureOpenAI
from tenacity import retry, wait_exponential, stop_after_attempt

# Carica le variabili d'ambiente dal file .env
load_dotenv(dotenv_path=".env")

# Recupera le credenziali
api_key = os.getenv("AZURE_API_KEY")
endpoint = os.getenv("AZURE_ENDPOINT")
deployment = os.getenv("DEPLOYMENT")

client = AzureOpenAI(
    api_version="2025-01-01-preview",
    azure_endpoint=endpoint,
    api_key=api_key,
)

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=10),  # 2s, 4s, 8s, max 10s
    stop=stop_after_attempt(5)  # massimo 5 tentativi
)
def ask():
    return client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": "I am going to Paris, what should I see?",
            }
        ],
        max_tokens=4096,
        temperature=1.0,
        top_p=1.0,
        model=deployment
    )

response = ask()

print(response.choices[0].message.content)

