"""Behavioural contracts for the runtime concurrency primitives.

These are the foundations of the owner-thread architecture in
``docs/adr/0003-runtime-threading-and-realtime-architecture.md``. Each primitive
exists because a *specific* reviewed mistake must be prevented, so the tests are
written against the failure modes rather than the implementation:

- ``AudioRing``   — the PortAudio callback's only output. Single producer,
  single consumer, and the writer must **never** touch consumer state: an
  earlier design had the writer advance the read index on overflow, which is a
  data race that defeats the whole point of an SPSC ring.
- ``Mailbox``     — single-slot latest-value-wins, for the camera frame. A stale
  frame is worthless, so staleness beats loss here.
- ``BoundedDropOldest`` — a bounded FIFO for transcripts, where loss must be
  explicit and countable, and order must be preserved. Latest-wins here would
  silently delete a user utterance.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from src.runtime.concurrency import AudioRing, BoundedDropOldest, Mailbox


# ── Mailbox: latest wins, non-consuming peek ─────────────────────────────────


def test_mailbox_returns_none_when_never_written():
    assert Mailbox().get() is None


def test_mailbox_keeps_only_the_latest_value():
    box: Mailbox[str] = Mailbox()
    box.put("first")
    box.put("second")
    assert box.get() == "second"


def test_mailbox_get_consumes_and_then_empties():
    box: Mailbox[int] = Mailbox()
    box.put(1)
    assert box.get() == 1
    assert box.get() is None


def test_mailbox_peek_does_not_consume():
    box: Mailbox[str] = Mailbox()
    box.put("frame")
    assert box.peek() == "frame"
    assert box.peek() == "frame"
    assert box.get() == "frame"
    assert box.peek() is None


def test_mailbox_never_grows():
    """Latest-wins must bound memory even under a fast producer."""
    box: Mailbox[np.ndarray] = Mailbox()
    for i in range(1000):
        box.put(np.full(1024, i, dtype=np.float32))
    latest = box.peek()
    assert latest is not None
    assert float(latest[0]) == 999.0


def test_mailbox_clear():
    box: Mailbox[int] = Mailbox()
    box.put(5)
    box.clear()
    assert box.peek() is None


# ── BoundedDropOldest: bounded FIFO, explicit and countable loss ─────────────


def test_bounded_queue_returns_none_when_empty():
    assert BoundedDropOldest(capacity=2).get_nowait() is None


def test_bounded_queue_preserves_fifo_order():
    """Order matters: transcripts are not interchangeable."""
    q: BoundedDropOldest[int] = BoundedDropOldest(capacity=3)
    for i in range(3):
        q.put(i)
    assert [q.get_nowait() for _ in range(3)] == [0, 1, 2]


def test_bounded_queue_drops_oldest_and_counts_it():
    q: BoundedDropOldest[int] = BoundedDropOldest(capacity=2)
    q.put(1)
    q.put(2)
    q.put(3)
    assert q.dropped == 1
    assert [q.get_nowait() for _ in range(2)] == [2, 3]


def test_bounded_queue_requires_positive_capacity():
    with pytest.raises(ValueError):
        BoundedDropOldest(capacity=0)


# ── AudioRing: SPSC, writer never touches consumer state ─────────────────────


def test_audio_ring_starts_empty():
    ring = AudioRing(capacity=8)
    assert ring.available == 0
    assert ring.read(4).size == 0


def test_audio_ring_round_trip_preserves_order():
    ring = AudioRing(capacity=8)
    ring.write(np.arange(4, dtype=np.float32))
    assert ring.available == 4
    assert list(ring.read(4)) == [0.0, 1.0, 2.0, 3.0]
    assert ring.available == 0


def test_audio_ring_reads_oldest_samples_first():
    ring = AudioRing(capacity=8)
    ring.write(np.arange(6, dtype=np.float32))
    assert list(ring.read(3)) == [0.0, 1.0, 2.0]
    assert list(ring.read(3)) == [3.0, 4.0, 5.0]


def test_audio_ring_read_is_clamped_to_available():
    ring = AudioRing(capacity=8)
    ring.write(np.arange(2, dtype=np.float32))
    assert list(ring.read(99)) == [0.0, 1.0]


def test_audio_ring_wraps_around_capacity():
    ring = AudioRing(capacity=4)
    ring.write(np.arange(4, dtype=np.float32))
    assert list(ring.read(4)) == [0.0, 1.0, 2.0, 3.0]
    ring.write(np.arange(4, 8, dtype=np.float32))
    assert list(ring.read(4)) == [4.0, 5.0, 6.0, 7.0]


def test_audio_ring_write_larger_than_capacity_keeps_the_tail():
    """The newest audio is what matters; the rest is necessarily lost."""
    ring = AudioRing(capacity=4)
    ring.write(np.arange(10, dtype=np.float32))
    assert ring.available == 4
    assert list(ring.read(4)) == [6.0, 7.0, 8.0, 9.0]


def test_audio_ring_writer_does_not_touch_consumer_state():
    """The writer must not advance the read cursor - that is the SPSC invariant.

    A design that had the writer advance the read index on overflow would be a
    data race. Observable consequence: a writer that overflows the ring must not
    cause the *consumer* to report skipped samples; only the consumer, on seeing
    a gap, may do that.
    """
    ring = AudioRing(capacity=4)
    ring.write(np.full(100, 1.0, dtype=np.float32))
    # The writer overflowed badly, but the consumer has not read yet.
    assert ring.dropped == 0
    assert ring.available == 4  # capped at capacity, not 100


def test_audio_ring_consumer_falling_behind_is_counted():
    """When the reader is lapped, it resyncs to the newest data and reports it."""
    ring = AudioRing(capacity=4)
    ring.write(np.arange(4, dtype=np.float32))   # consumer has not read
    ring.write(np.arange(4, 8, dtype=np.float32))  # lapped once
    assert list(ring.read(4)) == [4.0, 5.0, 6.0, 7.0]
    assert ring.dropped == 4


def test_audio_ring_empty_write_is_a_noop():
    ring = AudioRing(capacity=4)
    ring.write(np.zeros(0, dtype=np.float32))
    assert ring.available == 0
    assert ring.dropped == 0


def test_audio_ring_non_positive_read_is_a_noop():
    ring = AudioRing(capacity=4)
    ring.write(np.arange(4, dtype=np.float32))
    assert ring.read(0).size == 0
    assert ring.available == 4  # nothing consumed


def test_audio_ring_requires_positive_capacity():
    with pytest.raises(ValueError):
        AudioRing(capacity=0)


def test_audio_ring_write_does_not_block_when_full():
    """The real-time writer must never block, no matter how far behind the reader is."""
    ring = AudioRing(capacity=4)
    start = time.perf_counter()
    for _ in range(1000):
        ring.write(np.ones(1024, dtype=np.float32))
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0


def test_audio_ring_is_safe_under_concurrent_producer_and_consumer():
    """Smoke test for the SPSC claim: no exception, no corruption of ordering."""
    ring = AudioRing(capacity=1024)
    written = 20_000
    consumed: list[np.ndarray] = []
    stop = threading.Event()

    def producer():
        for start in range(0, written, 256):
            ring.write(np.arange(start, start + 256, dtype=np.float32))
        stop.set()

    def consumer():
        while not stop.is_set() or ring.available > 0:
            chunk = ring.read(256)
            if chunk.size:
                consumed.append(chunk)

    threads = [threading.Thread(target=producer), threading.Thread(target=consumer)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    assert all(not t.is_alive() for t in threads), "SPSC smoke test deadlocked"
    data = np.concatenate(consumed) if consumed else np.zeros(0, dtype=np.float32)
    # Whatever survived must be strictly increasing: the ring must never
    # reorder or duplicate samples.
    assert np.all(np.diff(data) > 0), "ring reordered or duplicated samples"
