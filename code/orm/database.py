from contextlib import contextmanager

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker

from config.settings import settings

default_language = "fr"
url = URL.create(
    drivername="postgresql",
    host=settings("DB_HOST"),
    username=settings("DB_USER"),
    password=settings("DB_PASSWORD"),
    database=settings("DB_NAME"),
    port=settings("DB_PORT"),
)
engine = create_engine(url)
session = sessionmaker(bind=engine)
db_session = session()
Base = declarative_base()


@contextmanager
def get_db():
    try:
        yield db_session
    finally:
        db_session.close()


def insert_exel_in_dbb(file_name, sheet_name, request):
    file_content = pd.read_excel(file_name, sheet_name=sheet_name, skiprows=4)
    with get_db() as connection:
        for row in file_content.itertuples(index=False):
            connection.execute(request, tuple(row))
