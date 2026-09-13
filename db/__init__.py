"""Database extension objects and the ORM base class."""
from __future__ import annotations

from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import MetaData, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase

# Deterministic constraint names keep Alembic migrations identical on
# PostgreSQL and SQLite.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


db = SQLAlchemy(model_class=Base)
migrate = Migrate(render_as_batch=True)


@event.listens_for(Engine, "connect")
def _enable_sqlite_foreign_keys(dbapi_connection, connection_record):
    # SQLite ignores foreign keys unless asked, which would hide locally the
    # integrity errors PostgreSQL raises in production.
    if type(dbapi_connection).__module__.startswith("sqlite3"):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


from db import models  # noqa: E402,F401  (registers the tables on the metadata)
