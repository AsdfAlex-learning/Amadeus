"""Runtime concurrency primitives for the owner-thread architecture.

See ``docs/adr/0003-runtime-threading-and-realtime-architecture.md``.

The rules encoded here come directly from the architecture review:

* ``AudioRing`` is single-producer / single-consumer, and the **producer never
  reads or writes consumer-owned state**. An earlier design had the writer
  advance the read index on overflow; that is a data race and defeats the point
  of an SPSC ring. Instead the producer overwrites and the consumer detects that
  it was lapped by comparing monotonic counters.
* ``Mailbox`` is latest-value-wins, for data where staleness is worse than loss
  (camera frames).
* ``BoundedDropOldest`` is a FIFO with explicit, countable loss, for data where
  order and completeness matter (ASR transcripts).

Counters are plain Python ints. CPython guarantees a single attribute read or
write is atomic under the GIL, which is sufficient for one reader and one writer.
The write path deliberately takes no lock so the real-time audio callback never
blocks on contention.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Generic, TypeVar

import numpy as np

T = TypeVar("T")


class Mailbox(Generic[T]):
    """Single-slot, latest-value-wins container.

    Memory is bounded to exactly one value no matter how fast the producer runs.
    Use this only where a stale value is worthless; it is **not** appropriate for
    audio or transcripts, where the whole sequence matters.
    """

    __slots__ = ("_value", "_has_value", "_lock")

    def __init__(self) -> None:
        self._value: T | None = None
        self._has_value = False
        self._lock = threading.Lock()

    def put(self, value: T) -> None:
        """Publish ``value``, discarding any previous unread value."""
        with self._lock:
            self._value = value
            self._has_value = True

    def get(self) -> T | None:
        """Consume and return the latest value, or ``None`` if empty."""
        with self._lock:
            if not self._has_value:
                return None
            value = self._value
            self._value = None
            self._has_value = False
            return value

    def peek(self) -> T | None:
        """Return the latest value **without** consuming it."""
        with self._lock:
            return self._value if self._has_value else None

    def clear(self) -> None:
        with self._lock:
            self._value = None
            self._has_value = False


class BoundedDropOldest(Generic[T]):
    """Bounded FIFO that drops the oldest item on overflow and counts the loss.

    Order is preserved because the items are not interchangeable. Loss is
    explicit (``dropped``) rather than silent, so a caller can log or surface it.
    """

    __slots__ = ("_items", "_capacity", "_dropped", "_lock")

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._items: deque[T] = deque()
        self._capacity = int(capacity)
        self._dropped = 0
        self._lock = threading.Lock()

    def put(self, item: T) -> None:
        with self._lock:
            if len(self._items) >= self._capacity:
                self._items.popleft()
                self._dropped += 1
            self._items.append(item)

    def get_nowait(self) -> T | None:
        with self._lock:
            return self._items.popleft() if self._items else None

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def dropped(self) -> int:
        """Number of items discarded because the queue was full."""
        with self._lock:
            return self._dropped


class AudioRing:
    """Single-producer / single-consumer ring of audio samples.

    The producer's only obligation is to copy samples and bump its own counter.
    It never blocks and never inspects consumer state, so an overflowing producer
    cannot corrupt the reader's cursor — the reader notices it was lapped and
    resyncs to the newest ``capacity`` samples, reporting the skipped count.
    """

    __slots__ = ("_buf", "_capacity", "_written_total", "_read_total", "_dropped")

    def __init__(self, capacity: int, dtype: type = np.float32) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = int(capacity)
        self._buf = np.zeros(self._capacity, dtype=dtype)
        self._written_total = 0  # producer-owned
        self._read_total = 0  # consumer-owned
        self._dropped = 0  # consumer-owned: samples skipped because of lapping

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def written_total(self) -> int:
        """Total samples ever written (monotonic)."""
        return self._written_total

    @property
    def available(self) -> int:
        """Samples currently readable, capped at ``capacity``."""
        lag = self._written_total - self._read_total
        if lag <= 0:
            return 0
        return min(lag, self._capacity)

    @property
    def dropped(self) -> int:
        """Samples this consumer had to skip because the producer lapped it."""
        return self._dropped

    def write(self, samples: np.ndarray) -> None:
        """Copy ``samples`` in, overwriting the oldest data when full.

        Never blocks. Never touches consumer state. If ``samples`` is longer than
        the ring, only its newest ``capacity`` samples can survive.
        """
        data = np.asarray(samples)
        count = int(data.size)
        if count <= 0:
            return
        if count > self._capacity:
            data = data[count - self._capacity :]
            count = self._capacity

        start = self._written_total % self._capacity
        first = min(count, self._capacity - start)
        self._buf[start : start + first] = data[:first]
        if first < count:
            self._buf[: count - first] = data[first:]
        self._written_total += count

    def read(self, count: int) -> np.ndarray:
        """Return up to ``count`` oldest available samples, in order.

        Resyncs to the newest data if this consumer was lapped, counting the
        missed samples in :attr:`dropped`.
        """
        count = int(count)
        if count <= 0:
            return np.zeros(0, dtype=self._buf.dtype)

        lag = self._written_total - self._read_total
        if lag > self._capacity:
            skipped = lag - self._capacity
            self._read_total += skipped
            self._dropped += skipped
            lag = self._capacity
        if lag <= 0:
            return np.zeros(0, dtype=self._buf.dtype)

        take = min(count, lag)
        start = self._read_total % self._capacity
        first = min(take, self._capacity - start)
        out = np.empty(take, dtype=self._buf.dtype)
        out[:first] = self._buf[start : start + first]
        if first < take:
            out[first:] = self._buf[: take - first]
        self._read_total += take
        return out


class FrameBuffer:
    """Thread-safe bounded buffer of timestamped frames, read newest-first.

    Written by the camera thread and read by the motion worker. A plain list
    cannot be shared between them: the writer pruned with ``pop(0)`` while the
    reader indexed a snapshot, so a read could raise ``IndexError`` or silently
    pair the wrong frames with the current audio.

    Frames are stamped at ingest and evicted by age, so a stalled camera does not
    leave the model pairing a two-second-old frame with audio from now. This is
    never on the real-time audio path, so a short-lived lock is acceptable.
    """

    __slots__ = ("_frames", "_capacity", "_max_age_s", "_dropped", "_lock")

    def __init__(self, capacity: int = 10, max_age_s: float = 2.5) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._frames: deque[tuple[float, np.ndarray]] = deque(maxlen=capacity)
        self._capacity = int(capacity)
        self._max_age_s = float(max_age_s)
        self._dropped = 0
        self._lock = threading.Lock()

    def put(self, frame: np.ndarray, timestamp: float | None = None) -> None:
        """Publish a frame, stamped at ingest."""
        now = time.monotonic() if timestamp is None else float(timestamp)
        with self._lock:
            if len(self._frames) >= self._capacity:
                self._dropped += 1
            self._frames.append((now, frame))
            self._evict_expired(now)

    def _evict_expired(self, now: float) -> None:
        while self._frames and (now - self._frames[0][0]) > self._max_age_s:
            self._frames.popleft()

    def snapshot(self, count: int, now: float | None = None) -> list[np.ndarray]:
        """Return the newest ``count`` frames, oldest first."""
        if count <= 0:
            return []
        current = time.monotonic() if now is None else float(now)
        with self._lock:
            self._evict_expired(current)
            frames = [frame for _, frame in self._frames]
        return frames[-count:]

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def dropped(self) -> int:
        """Frames discarded because the buffer was full."""
        with self._lock:
            return self._dropped
