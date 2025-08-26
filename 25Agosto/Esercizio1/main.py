from dotenv import load_dotenv
import os
from openai import AzureOpenAI

# Carica le variabili d'ambiente dal file .env
load_dotenv(dotenv_path=".env")

# Recupera le credenziali
api_key = os.getenv("AZURE_API_KEY")
endpoint = os.getenv("AZURE_ENDPOINT")
deployment = os.getenv("DEPLOYMENT")

print("Deployment:", deployment)

client = AzureOpenAI(
    api_version="2024-02-01",
    azure_endpoint=endpoint,
    api_key=api_key,
)

response = client.embeddings.create(
    input=["Hello World"],
    model=deployment
)

for item in response.data:
    length = len(item.embedding)
    print(
        f"data[{item.index}]: length={length}, "
        f"[{item.embedding[0]}, {item.embedding[1]}, "
        f"..., {item.embedding[length-2]}, {item.embedding[length-1]}]"
    )
print(response.usage)

