from sqlmodel import SQLModel
from core.db import engine

def init_db() -> None:
    SQLModel.metadata.create_all(engine)
