import sqlite3
from datetime import datetime

DATABASE_FILE = "fall_events.db"


def create_connection():
    """Create a database connection to the SQLite database."""
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        return conn
    except sqlite3.Error as e:
        print(f"[DB] Error connecting to database: {e}")
        return None


def create_table():
    """Create the fall_events table with a schema that matches the JSON data."""
    conn = create_connection()
    if conn:
        try:
            sql_create_table = """
            CREATE TABLE IF NOT EXISTS fall_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                device_id TEXT,
                fall_detected INTEGER,
                latitude REAL,
                longitude REAL,
                has_gps_fix INTEGER,
                status TEXT NOT NULL
            );
            """
            cursor = conn.cursor()
            cursor.execute(sql_create_table)
            conn.commit()
            print("[DB] Database and table created successfully.")
        except sqlite3.Error as e:
            print(f"[DB] Error creating table: {e}")
        finally:
            conn.close()


def insert_fall_event(json_data):
    """
    Insert a new fall event from a JSON object.

    Args:
        json_data (dict): A dictionary containing the fall event data.

    Returns:
        int: The ID of the newly inserted row, or None on failure.
    """
    conn = create_connection()
    if conn:
        try:
            # Extract data from the JSON object
            timestamp = datetime.fromtimestamp(
                json_data.get("timestamp", 0)
            ).isoformat()
            device_id = json_data.get("device_id")
            fall_detected = (
                1 if json_data.get("fall_detected", False) else 0
            )  # SQLite stores boolean as INTEGER
            latitude = json_data.get("latitude")
            longitude = json_data.get("longitude")
            has_gps_fix = 1 if json_data.get("has_gps_fix", False) else 0
            status = "pending"

            sql = """
            INSERT INTO fall_events(timestamp, device_id, fall_detected, latitude, longitude, has_gps_fix, status)
            VALUES (?, ?, ?, ?, ?, ?, ?);
            """
            cursor = conn.cursor()
            cursor.execute(
                sql,
                (
                    timestamp,
                    device_id,
                    fall_detected,
                    latitude,
                    longitude,
                    has_gps_fix,
                    status,
                ),
            )
            conn.commit()
            print(f"[DB] Fall event recorded with ID: {cursor.lastrowid}")
            return cursor.lastrowid
        except sqlite3.Error as e:
            print(f"[DB] Error inserting data: {e}")
            return None
        finally:
            conn.close()


def update_alert_status(event_id, new_status):
    """Update the status of a fall alert by its ID."""
    conn = create_connection()
    if conn:
        try:
            sql = """
            UPDATE fall_events
            SET status = ?
            WHERE id = ?;
            """
            cursor = conn.cursor()
            cursor.execute(sql, (new_status, event_id))
            conn.commit()
            print(f"[DB] Alert ID {event_id} status updated to '{new_status}'.")
        except sqlite3.Error as e:
            print(f"[DB] Error updating status: {e}")
        finally:
            conn.close()


def get_all_alerts():
    """Retrieve all fall alerts from the database."""
    conn = create_connection()
    alerts = []
    if conn:
        try:
            sql = "SELECT id, timestamp, device_id, fall_detected, latitude, longitude, has_gps_fix, status FROM fall_events ORDER BY timestamp DESC;"
            cursor = conn.cursor()
            cursor.execute(sql)
            alerts = cursor.fetchall()
        except sqlite3.Error as e:
            print(f"[DB] Error retrieving alerts: {e}")
        finally:
            conn.close()
    return alerts
