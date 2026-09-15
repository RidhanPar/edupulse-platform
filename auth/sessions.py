"""Server-side sessions in the migration-managed user_sessions table.

Built on Flask-Session's ServerSideSessionInterface, which handles the cookie,
serialisation, regeneration and expiry clean-up. This replaces Flask-Session's
bundled SQLAlchemy backend (0.8.0), which creates its table at application start-up
outside Alembic, redefines its model on every create_app(), and commits the app's
own db.session whenever it saves a session, so a view's uncommitted changes would be
committed as a side effect. Here every session read and write runs on its own short
connection and transaction.
"""
from __future__ import annotations

import uuid
from datetime import timedelta
from typing import Optional

from flask import Flask
from flask_session.base import ServerSideSession, ServerSideSessionInterface
from sqlalchemy import delete, insert, select, update

from db import db
from db.models import UserSession, utcnow

_TABLE = UserSession.__table__


def _session_user_id(session: ServerSideSession) -> uuid.UUID | None:
    try:
        return uuid.UUID(str(dict(session).get("_user_id")))
    except ValueError:
        return None


class DatabaseSessionInterface(ServerSideSessionInterface):
    session_class = ServerSideSession
    ttl = False  # the database does not expire rows itself; see _delete_expired_sessions

    def __init__(self, app: Flask) -> None:
        super().__init__(
            app,
            permanent=True,
            cleanup_n_requests=app.config.get("SESSION_CLEANUP_N_REQUESTS"),
        )

    def _retrieve_session_data(self, store_id: str) -> Optional[dict]:
        with db.engine.connect() as connection:
            data = connection.execute(
                select(_TABLE.c.data).where(_TABLE.c.session_id == store_id, _TABLE.c.expiry > utcnow())
            ).scalar_one_or_none()
        return self.serializer.decode(data) if data is not None else None

    def _delete_session(self, store_id: str) -> None:
        with db.engine.begin() as connection:
            connection.execute(delete(_TABLE).where(_TABLE.c.session_id == store_id))

    def _upsert_session(self, session_lifetime: timedelta, session: ServerSideSession, store_id: str) -> None:
        values = {
            "data": self.serializer.encode(session),
            "expiry": utcnow() + session_lifetime,
            "user_id": _session_user_id(session),
        }
        with db.engine.begin() as connection:
            updated = connection.execute(
                update(_TABLE).where(_TABLE.c.session_id == store_id).values(**values)
            ).rowcount
            if not updated:
                connection.execute(insert(_TABLE).values(session_id=store_id, **values))

    def _delete_expired_sessions(self) -> None:
        with db.engine.begin() as connection:
            connection.execute(delete(_TABLE).where(_TABLE.c.expiry <= utcnow()))

    def regenerate(self, session: ServerSideSession) -> None:
        """Give the session a new id and discard the old one.

        Unlike the base class, this also rotates an empty session: skipping those would
        let an id planted before login carry over into the authenticated session.
        """
        self._delete_session(self._get_store_id(session.sid))
        session.sid = self._generate_sid(self.sid_length)
        session.modified = True

    def revoke_user_sessions(self, user_id) -> None:
        """End every stored session belonging to `user_id`."""
        with db.engine.begin() as connection:
            connection.execute(delete(_TABLE).where(_TABLE.c.user_id == uuid.UUID(str(user_id))))
