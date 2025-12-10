# db.py
import os
from dotenv import load_dotenv

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

load_dotenv()  # charge le .env

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://myuser:mypassword@192.168.1.208:5432/mydatabase",
)

engine = create_engine(
    DATABASE_URL,
    echo=True,        # log les requêtes SQL (utile en dev)
    future=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# Dependency pour FastAPI
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
