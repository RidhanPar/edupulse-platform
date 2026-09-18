"""Argon2id password hashing with argon2-cffi.

Hashes use the standard PHC string format, so hashes written earlier by passlib
(before this module switched libraries) still verify, and are upgraded to the
current parameters on the user's next login.
"""
from __future__ import annotations

import secrets
from functools import lru_cache

from argon2 import PasswordHasher, Type
from argon2.exceptions import InvalidHashError, VerificationError
from flask import current_app


@lru_cache(maxsize=8)
def _hasher(memory_cost: int, time_cost: int, parallelism: int) -> PasswordHasher:
    return PasswordHasher(
        time_cost=time_cost,
        memory_cost=memory_cost,
        parallelism=parallelism,
        hash_len=32,
        salt_len=16,
        type=Type.ID,
    )


def _params() -> tuple[int, int, int]:
    params = current_app.config["ARGON2_PARAMS"]
    return params["memory_cost"], params["time_cost"], params["parallelism"]


def hash_password(password: str) -> str:
    return _hasher(*_params()).hash(password)


def verify_password(password: str, password_hash: str | None) -> bool:
    if not password_hash:
        return False
    try:
        return _hasher(*_params()).verify(password_hash, password)
    except (VerificationError, InvalidHashError):
        return False


def needs_rehash(password_hash: str | None) -> bool:
    """True when a stored hash uses different parameters from the current ones."""
    if not password_hash:
        return False
    try:
        return _hasher(*_params()).check_needs_rehash(password_hash)
    except InvalidHashError:
        return True


@lru_cache(maxsize=8)
def _dummy_hash(memory_cost: int, time_cost: int, parallelism: int) -> str:
    return _hasher(memory_cost, time_cost, parallelism).hash(secrets.token_urlsafe(32))


def dummy_hash() -> str:
    """A hash of a random secret, verified when there is no real hash to check.

    Failing logins for unregistered addresses then cost the same Argon2 work as real
    ones, so response time does not reveal which addresses have accounts.
    """
    return _dummy_hash(*_params())
