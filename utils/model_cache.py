"""Process-local cache of deserialised models."""
from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Callable, Hashable


class ModelCache:
    """A small LRU of loaded models, so each worker unpickles a model once rather than per request.

    Keys identify an immutable artifact: (organisation id, ModelArtifact id). Retraining
    creates a new artifact id, so entries never go stale and need no invalidation. It
    holds loaded models only, never predictions.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.capacity = capacity
        self._entries: OrderedDict[Hashable, Any] = OrderedDict()
        self._lock = threading.Lock()

    def get_or_load(self, key: Hashable, loader: Callable[[], Any]) -> Any:
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                return self._entries[key]
        # Load outside the lock: two threads may occasionally load the same model, which is
        # harmless, rather than every request queueing behind one slow load.
        model = loader()
        with self._lock:
            self._entries[key] = model
            self._entries.move_to_end(key)
            while len(self._entries) > self.capacity:
                self._entries.popitem(last=False)
        return model

    def __contains__(self, key: Hashable) -> bool:
        with self._lock:
            return key in self._entries

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
