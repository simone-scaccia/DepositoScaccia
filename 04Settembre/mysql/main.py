import mysql.connector

if __name__ == "__main__":
    # Setup: crea la tabella utenti se non esiste
    mydb = mysql.connector.connect(
        host="localhost",
        user="myuser",
        password="mypassword",
        port=3306,
        database="mydb"
    )
    mycursor = mydb.cursor()

    mycursor.execute("CREATE TABLE IF NOT EXISTS utenti (id INT AUTO_INCREMENT PRIMARY KEY, nome VARCHAR(255))")

    while True:
        # Inserisci 0 per inserire un nuovo utente, 1 per visualizzare tutti gli utenti, o 'exit' per uscire
        print("Comandi:")
        print("0 - Inserisci un nuovo utente")
        print("1 - Visualizza tutti gli utenti")
        print("exit - Esci dal programma")
        action = input("Seleziona un'azione: ")
        if action == 'exit':
            print("Uscita dal programma.")
            break
        elif action == '0':
            nome = input("Inserisci il nome dell'utente: ")
            sql = "INSERT INTO utenti (nome) VALUES (%s)"
            val = (nome,)
            mycursor.execute(sql, val)
            mydb.commit()
            print(f"Utente {nome} inserito con ID {mycursor.lastrowid}.")
        elif action == '1':
            mycursor.execute("SELECT * FROM utenti")
            myresult = mycursor.fetchall()
            for x in myresult:
                print(x)