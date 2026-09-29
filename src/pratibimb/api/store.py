"""Twin repositories. Twins are event-sourced, so persisting the record is sufficient."""

from __future__ import annotations

import contextlib
import json
import os
import re
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from pratibimb.twin.engine import DigitalTwin, TwinRecord

_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


class TwinRepository(Protocol):
    def get(self, twin_id: str) -> DigitalTwin: ...
    def save(self, twin: DigitalTwin) -> None: ...
    def delete(self, twin_id: str) -> None: ...
    def list_ids(self) -> list[str]: ...


class InMemoryRepository:
    def __init__(self) -> None:
        self._twins: dict[str, DigitalTwin] = {}
        self._lock = threading.Lock()

    def get(self, twin_id: str) -> DigitalTwin:
        with self._lock:
            return self._twins[twin_id]

    def save(self, twin: DigitalTwin) -> None:
        with self._lock:
            self._twins[twin.twin_id] = twin

    def delete(self, twin_id: str) -> None:
        with self._lock:
            del self._twins[twin_id]

    def list_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._twins)


class JsonFileRepository(InMemoryRepository):
    """Write-through cache backed by one JSON event log per twin (atomic replace on save)."""

    def __init__(self, directory: Path) -> None:
        super().__init__()
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)

    def _path(self, twin_id: str) -> Path:
        if not _ID_RE.match(twin_id):
            raise KeyError(twin_id)
        return self.directory / f"{twin_id}.json"

    def get(self, twin_id: str) -> DigitalTwin:
        try:
            return super().get(twin_id)
        except KeyError:
            path = self._path(twin_id)
            if not path.exists():
                raise
            twin = DigitalTwin.from_record(TwinRecord(**json.loads(path.read_text(encoding="utf-8"))))
            super().save(twin)
            return twin

    def save(self, twin: DigitalTwin) -> None:
        super().save(twin)
        path = self._path(twin.twin_id)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(twin.to_record())), encoding="utf-8")
        os.replace(tmp, path)

    def delete(self, twin_id: str) -> None:
        with contextlib.suppress(KeyError):
            super().delete(twin_id)
        path = self._path(twin_id)
        if not path.exists():
            raise KeyError(twin_id)
        path.unlink()

    def list_ids(self) -> list[str]:
        return sorted(p.stem for p in self.directory.glob("*.json"))
