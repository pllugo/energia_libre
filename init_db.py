import sqlite3



connection = sqlite3.connect('database.db')


with open('schema.sql') as f:
    connection.executescript(f.read())

cur = connection.cursor()

cur.execute("INSERT INTO covs (formula, tipo, kcl, koh, logkCl, logkOH, enlace) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ('0', '0', '0', '0', '0', '0', '0')
            )


connection.commit()
connection.close()
