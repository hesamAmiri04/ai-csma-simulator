"""
ai_node/csma_mac.py
===================
CSMA/CA MAC-layer state machine.

States
------
  IDLE       → node has no packet to send
  SENSE      → node senses the channel (asks medium FREE/BUSY)
  BACKOFF    → channel was busy; waiting a random number of slots before re-sensing
  TRANSMIT   → node is transmitting (TX_START sent; waiting for ACK/COLLISION)
  WAIT_ACK   → transmission sent; awaiting medium feedback

Interaction with the wireless medium uses the same UDP protocol defined in
medium.py.  All socket operations use a short timeout so the state machine
never blocks indefinitely.
"""

import socket
import time
import random
import logging
import os

log = logging.getLogger("csma_mac")

# ── Configuration ──────────────────────────────────────────────────────────────
MEDIUM_HOST   = os.environ.get("MEDIUM_HOST",   "medium")
MEDIUM_PORT   = int(os.environ.get("MEDIUM_PORT", 9000))
SLOT_MS       = float(os.environ.get("SLOT_MS",   10))    # slot duration in ms
TX_DURATION   = float(os.environ.get("TX_DURATION_MS", 20))  # TX slot in ms
SOCK_TIMEOUT  = 0.5   # seconds – socket receive timeout


class CSMAResult:
    """Outcome of one CSMA transmission attempt."""
    SUCCESS   = "SUCCESS"
    COLLISION = "COLLISION"
    BUSY      = "BUSY"       # channel was busy for the whole CW (extreme congestion)


class CSMAMAC:
    """
    CSMA/CA MAC state machine.

    Parameters
    ----------
    node_id : str
        Unique identifier for this node (e.g. "node_3").
    """

    def __init__(self, node_id: str):
        self.node_id = node_id
        self._sock   = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(SOCK_TIMEOUT)
        self._medium = (MEDIUM_HOST, MEDIUM_PORT)
        log.info("[%s] MAC initialised → medium at %s:%d", node_id, MEDIUM_HOST, MEDIUM_PORT)

    # ── Low-level medium interaction ──────────────────────────────────────────
    def _send(self, msg: str) -> None:
        self._sock.sendto(msg.encode(), self._medium)

    def _recv(self) -> str | None:
        try:
            data, _ = self._sock.recvfrom(256)
            return data.decode().strip()
        except socket.timeout:
            return None

    # ── Channel sensing ───────────────────────────────────────────────────────
    def sense(self) -> bool:
        """Return True if channel is FREE, False if BUSY."""
        self._send(f"SENSE {self.node_id}")
        reply = self._recv()
        return reply == "FREE"

    # ── One complete CSMA attempt ─────────────────────────────────────────────
    def attempt_transmit(
        self,
        contention_window: int,
        history_window:    int = 10,
    ) -> tuple[str, dict]:
        """
        Run one full CSMA attempt using the given contention window size.

        Returns
        -------
        (result: str, metrics: dict)
            result  – CSMAResult constant
            metrics – busy_slots, backoff_slots, delay_ms
        """
        busy_slots    = 0
        backoff_slots = 0
        start_time    = time.time()

        # ── SENSE phase ───────────────────────────────────────────────────────
        if not self.sense():
            busy_slots += 1
            # ── BACKOFF phase ─────────────────────────────────────────────────
            backoff = random.randint(0, contention_window - 1)
            log.debug("[%s] channel BUSY → backoff=%d slots (CW=%d)",
                      self.node_id, backoff, contention_window)
            for _ in range(backoff):
                time.sleep(SLOT_MS / 1000.0)
                backoff_slots += 1
                if not self.sense():
                    busy_slots += 1
                # Re-sense after each backoff slot (802.11-style freeze)

        # ── TRANSMIT phase ────────────────────────────────────────────────────
        self._send(f"TX_START {self.node_id} {TX_DURATION}")

        # ── WAIT_ACK phase ────────────────────────────────────────────────────
        reply = self._recv()
        delay_ms = (time.time() - start_time) * 1000

        if reply and reply.startswith("ACK"):
            result = CSMAResult.SUCCESS
        elif reply and reply.startswith("COLLISION"):
            result = CSMAResult.COLLISION
        else:
            # Timeout or unknown → treat as collision / channel grab failure
            result = CSMAResult.COLLISION

        metrics = {
            "busy_slots":    busy_slots,
            "backoff_slots": backoff_slots,
            "delay_ms":      delay_ms,
            "result":        result,
        }
        log.debug("[%s] %s | busy=%d backoff=%d delay=%.1fms",
                  self.node_id, result, busy_slots, backoff_slots, delay_ms)
        return result, metrics

    def close(self):
        self._sock.close()
