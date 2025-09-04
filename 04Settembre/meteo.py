import requests

# URL api per ottenere latitudine e longitudine di una città. Placeholder {city} da sostituire con il nome della città
URL_API_LONG_LAT = "https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=it&format=json"
URL_API_WEATHER = "https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&hourly=temperature_2m{settings}&forecast_days={days}"

PRECIPITATION_SETTINGS = ",precipitation_probability"
WIND_SETTINGS = ",wind_speed_10m"

QA = [
    {
     "q": "Inserisci il numero di giorni per la previsione tra 1, 3, 7 (o 'exit' per uscire): ",
     "choices": ['1', '3', '7'],
     "error": "Numero di giorni non valido. Riprova.",
     "action": lambda x: int(x)
     },
    {
     "q": "Includere la probabilità di precipitazione? (s/n) (o 'exit' per uscire): ",
     "choices": ['s', 'n'],
     "error": "Risposta non valida. Riprova.",
     "action": lambda x: PRECIPITATION_SETTINGS if x.lower() == 's' else ''
    },
    {
     "q": "Includere la velocità del vento? (s/n) (o 'exit' per uscire): ",
     "choices": ['s', 'n'],
     "error": "Risposta non valida. Riprova.",
     "action": lambda x: WIND_SETTINGS if x.lower() == 's' else ''
    }
]

def check_exit(input_str):
    if input_str.lower() == 'exit':
        print("Uscita dal programma.")
        exit()

def get_weather_data(city):
    # Ottieni latitudine e longitudine
    response = requests.get(URL_API_LONG_LAT.format(city=city), verify=False)
    data = response.json()
    if response.status_code != 200:
        print("Errore nella richiesta di latitudine e longitudine. Errore:", response.status_code)
        return
    if 'results' not in data or len(data['results']) == 0:
        print(f"Nessun risultato trovato per la città: {city}")
        return    
    
    name = data['results'][0]['name']
    latitude = data['results'][0]['latitude']
    longitude = data['results'][0]['longitude']    
    print(f"City: {name}, Latitudine: {latitude}, Longitudine: {longitude}")

    # Ottieni dati meteo
    settings = ''.join(QA[1]['a'] + QA[2]['a'])
    days = QA[0]['a']
    weather_response = requests.get(URL_API_WEATHER.format(latitude=latitude, longitude=longitude, settings=settings, days=days), verify=False)
    if weather_response.status_code != 200:
        print("Errore nella richiesta di dati meteo. Errore:", weather_response.status_code)
        return
    weather_data = weather_response.json()
    print(f"Dati meteo per {city} per i prossimi {days} giorni:")
    print("Temperatura (2m):", weather_data['hourly']['temperature_2m'])
    if QA[1]['a'] != '':
        print("Probabilità di precipitazione:", weather_data['hourly'].get('precipitation_probability', 'N/A'))
    if QA[2]['a'] != '':
        print("Velocità del vento (10m):", weather_data['hourly'].get('wind_speed_10m', 'N/A'))

def main():
    while True:
        # inserisci la città da cercare o 'exit' per uscire
        city = input("Inserisci il nome della città (o 'exit' per uscire): ")
        check_exit(city)

        for item in QA:
            answer = None
            while answer not in item["choices"]:
                answer = input(item["q"])
                check_exit(answer)
                if answer not in item["choices"]:
                    print(item["error"])
                    continue
            item["a"] = item["action"](answer)

        get_weather_data(city)

        
if __name__ == "__main__":
   main()
