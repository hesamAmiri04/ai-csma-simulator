"""
wireless_medium/medium.py
=========================
Virtual "air" channel for the AI-CSMA simulator.

Protocol (UDP, all messages are plain text):
  Node → Medium:
    SENSE <node_id>               → Medium replies FREE or BUSY
    TX_START <node_id> <duration_ms>  → begin a transmission slot
    TX_END   <node_id>            → finish the transmission slot

  Medium → Node:
    FREE                          → channel is idle
    BUSY                          → channel is occupied
    ACK  <node_id>                → successful delivery
    COLLISION <node_id>           → collision detected

The medium uses a single-slot model: if more than one TX_START arrives
within the same slot window, all overlapping transmissions are marked as
collisions.
"""

import socket
import threading
import time
import logging
import json
import os

# ── Configuration ────────────────────────────────────────────────────────────
MEDIUM_HOST   = "0.0.0.0"
MEDIUM_PORT   = int(os.environ.get("MEDIUM_PORT", 9000))
SLOT_DURATION = float(os.environ.get("SLOT_DURATION_MS", 10)) / 1000.0  # seconds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MEDIUM] %(levelname)s %(message)s",
)
log = logging.getLogger("medium")

# ── Shared state ─────────────────────────────────────────────────────────────
lock              = threading.Lock()
active_tx: dict   = {}   # node_id -> (addr, deadline)
stats             = {"total_tx": 0, "collisions": 0, "successes": 0}


def channel_is_busy() -> bool:
    now = time.time()
    with lock:
        # Prune expired transmissions
        expired = [nid for nid, (_, deadline) in active_tx.items() if now > deadline]
        for nid in expired:
            del active_tx[nid]
        return len(active_tx) > 0


def handle_sense(node_id: str, addr, sock: socket.socket):
    reply = b"BUSY" if channel_is_busy() else b"FREE"
    sock.sendto(reply, addr)
    log.debug("SENSE from %s → %s", node_id, reply.decode())


def handle_tx_start(node_id: str, duration_ms: float, addr, sock: socket.socket):
    now      = time.time()
    deadline = now + duration_ms / 1000.0

    with lock:
        active_tx[node_id] = (addr, deadline)
        occupants = list(active_tx.keys())

    stats["total_tx"] += 1
    log.info("TX_START from %s  |  channel occupants: %s", node_id, occupants)

    # Schedule resolution after the slot ends
    def resolve():
        time.sleep(duration_ms / 1000.0 + 0.001)
        with lock:
            current_occupants = [
                nid for nid, (a, dl) in list(active_tx.items())
                if dl >= now   # still inside the original slot window
            ]

        if len(current_occupants) > 1:
            # Collision
            stats["collisions"] += 1
            log.warning("COLLISION among %s", current_occupants)
            with lock:
                for nid in current_occupants:
                    if nid in active_tx:
                        node_addr = active_tx[nid][0]
                        sock.sendto(f"COLLISION {nid}".encode(), node_addr)
                        del active_tx[nid]
        else:
            # Clean delivery
            stats["successes"] += 1
            log.info("ACK → %s", node_id)
            with lock:
                if node_id in active_tx:
                    node_addr = active_tx[node_id][0]
                    sock.sendto(f"ACK {node_id}".encode(), node_addr)
                    del active_tx[node_id]

    threading.Thread(target=resolve, daemon=True).start()


def handle_tx_end(node_id: str, addr, sock: socket.socket):
    """Explicit early end (optional; resolution already scheduled via timer)."""
    with lock:
        active_tx.pop(node_id, None)
    log.debug("TX_END from %s", node_id)


def stats_reporter():
    """Log aggregate stats every 10 seconds."""
    while True:
        time.sleep(10)
        log.info(
            "STATS | total_tx=%d  successes=%d  collisions=%d  "
            "collision_rate=%.1f%%",
            stats["total_tx"],
            stats["successes"],
            stats["collisions"],
            100 * stats["collisions"] / max(stats["total_tx"], 1),
        )


def run():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((MEDIUM_HOST, MEDIUM_PORT))
    log.info("Wireless medium listening on %s:%d", MEDIUM_HOST, MEDIUM_PORT)

    threading.Thread(target=stats_reporter, daemon=True).start()

    while True:
        try:
            data, addr = sock.recvfrom(1024)
            msg        = data.decode().strip().split()
            if not msg:
                continue

            cmd = msg[0].upper()

            if cmd == "SENSE" and len(msg) >= 2:
                handle_sense(msg[1], addr, sock)

            elif cmd == "TX_START" and len(msg) >= 3:
                handle_tx_start(msg[1], float(msg[2]), addr, sock)

            elif cmd == "TX_END" and len(msg) >= 2:
                handle_tx_end(msg[1], addr, sock)

            else:
                log.warning("Unknown message from %s: %s", addr, data)

        except Exception as exc:
            log.error("Error handling packet: %s", exc)


if __name__ == "__main__":
    run()
