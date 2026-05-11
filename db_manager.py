import sqlite3
from datetime import datetime

class DBManager:
    def __init__(self, db_name="history.db"):
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.create_table()

    def create_table(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                result VARCHAR(255) NOT NULL,
                confidence FLOAT,
                input_mode VARCHAR(10) NOT NULL,
                recognize_mode VARCHAR(20) NOT NULL,
                time DATETIME NOT NULL
            )
        ''')
        self.conn.commit()

    def insert_record(self, result, conf, in_mode, rec_mode):
        cursor = self.conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute('''
            INSERT INTO history (result, confidence, input_mode, recognize_mode, time)
            VALUES (?, ?, ?, ?, ?)
        ''', (result, conf, in_mode, rec_mode, now))
        self.conn.commit()

    def fetch_all(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM history ORDER BY time DESC")
        return cursor.fetchall()

    def delete_record(self, row_id):
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM history WHERE id=?", (row_id,))
        self.conn.commit()