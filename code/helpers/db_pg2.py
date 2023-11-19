import base64
import psycopg2
from config.settings import settings
from helpers.logger import logger

pg2_conn = None
pg2_cursor = None


def pg2_connect():
    global pg2_conn, pg2_cursor

    try:
        # Établir la connexion à la base de données PostgreSQL
        pg2_conn = psycopg2.connect(
            host=settings["PG_HOST"],
            port=settings["PG_PORT"],
            database=settings["PG_DATABASE"],
            user=settings["PG_USERNAME"],
            password=base64.b64decode(settings["PG_PASSWORD"]).decode("utf-8"),
            connect_timeout=settings["PG_CONNECT_TIMEOUT"],
        )
        # Créer un curseur
        pg2_cursor = pg2_conn.cursor()
    except:
        logger.critical(f"Impossible de se connecter à la base de données")
        return False

    return True


def pg2_disconnect():
    global pg2_conn, pg2_cursor

    if pg2_cursor is not None:
        pg2_cursor.close()
        pg2_cursor = None
    if pg2_conn is not None:
        pg2_conn.close()
        pg2_conn = None


if pg2_conn is None:
    pg2_connect()
