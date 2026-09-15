"""Argon2id password hashing via passlib."""
from __future__ import annotations

import secrets
from functools import lru_cache

from flask import current_app
from passlib.context import CryptContext


@lru_cache(maxsize=8)
def _context(memory_cost: int, time_cost: int, parallelism: int) -> CryptContext:
    return CryptContext(
        schemes=["argon2"],
        deprecated="auto",
        argon2__type="ID",
        argon2__memory_cost=memory_cost,
        argon2__time_cost=time_cost,
        argon2__parallelism=parallelism,
    )


def _params() -> tuple[int, int, int]:
    params = current_app.config["ARGON2_PARAMS"]
    return params["memory_cost"], params["time_cost"], params["parallelism"]


def hash_password(password: str) -> str:
    return _context(*_params()).hash(password)


def verify_password(password: str, password_hash: str | None) -> bool:
    if not password_hash:
        return False
    try:
        return _context(*_params()).verify(password, password_hash)
    except (TypeError, ValueError):
        return False


def needs_rehash(password_hash: str | None) -> bool:
    """True when a stored hash uses weaker or different parameters than the current ones."""
    return bool(password_hash) and _context(*_params()).needs_update(password_hash)


@lru_cache(maxsize=8)
def _dummy_hash(memory_cost: int, time_cost: int, parallelism: int) -> str:
    return _context(memory_cost, time_cost, parallelism).hash(secrets.token_urlsafe(32))


def dummy_hash() -> str:
    """A hash of a random secret, verified when there is no real hash to check.

    Failing logins for unregistered addresses then cost the same Argon2 work as real
    ones, so response time does not reveal which addresses have accounts.
    """
    return _dummy_hash(*_params())
