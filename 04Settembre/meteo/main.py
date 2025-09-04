import requests
import urllib3

# Disabilita gli avvisi SSL (solo per test)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def main():
    url = "https://api.open-meteo.com/v1/forecast?latitude=52.52&longitude=13.41&hourly=temperature_2m"
    # Disabilita verifica SSL temporaneamente
    response = requests.get(url, verify=False)
    print("Richiesta effettuata con successo")
    print("Status Code", response.status_code)
    print("Contenuto della risposta JSON:", response.json())

main()