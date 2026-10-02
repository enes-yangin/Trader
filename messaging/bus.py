"""
ZeroMQ PUB/SUB Message Bus for TraderAI v2.

Single-process (inproc://) pub/sub with optional tkinter UI callback
routing. The bus is the central nervous system: services publish events,
the UI subscribes to relevant topics.

Architecture:
  ┌──────────────┐     PUB     ┌──────────────┐
  │  DataService │ ──────────→ │              │
  └──────────────┘             │              │
                               │  MessageBus  │
  ┌──────────────────┐   PUB   │  inproc://   │  SUB  ┌─────────┐
  │ TrainingService  │ ──────→ │  traderai    │ ────→ │  App UI │
  └──────────────────┘         │  -pub        │       └─────────┘
                               │              │
  ┌──────────────────┐   PUB   │              │
  │ BacktestService  │ ──────→ │              │
  └──────────────────┘         └──────────────┘

Thread safety:
  - publish(): protected by threading.Lock (ZMQ sockets are not thread-safe)
  - subscribe(): spawns a daemon thread per subscription for recv loop
  - UiSubscription: wraps callbacks with root.after(0, ...) for tkinter

"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional

import zmq

from utils.logger import get_logger

log = get_logger("message_bus")

INPROC_ADDR = "inproc://traderai-pub"


class Subscription:
    """Handle returned by MessageBus.subscribe().

    Call unsubscribe() to stop the listener thread and close the socket.
    """

    def __init__(self, socket: zmq.Socket, thread: threading.Thread, topic: str):
        self._socket = socket
        self._thread = thread
        self.topic = topic
        self._active = True

    def unsubscribe(self):
        """Stop the listener thread and close the subscription socket."""
        if not self._active:
            return
        self._active = False
        self._socket.close()
        # Thread is daemon — will exit when socket.recv_multipart() unblocks
        # on context termination. We don't join() to avoid blocking the main thread.
        # For clean shutdown, call MessageBus.close() which terminates the context.


class UiSubscription(Subscription):
    """Subscription that marshals callbacks to the tkinter main thread.

    Every received message is wrapped in root.after(0, callback, payload)
    so UI updates happen on the main thread — tkinter is not thread-safe.
    """

    def __init__(self, socket: zmq.Socket, thread: threading.Thread,
                 topic: str, root: Any, callback: Callable):
        super().__init__(socket, thread, topic)
        self._root = root
        self._callback = callback

    def _dispatch(self, topic_str: str, payload: Dict[str, Any]):
        """Schedule the callback on the tkinter event loop."""
        try:
            self._root.after(0, self._callback, payload)
        except Exception:
            log.exception("UiSubscription dispatch failed for topic %s", topic_str)


class MessageBus:
    """Central pub/sub bus backed by ZeroMQ inproc:// transport.

    Usage:
        bus = MessageBus()
        bus.subscribe("ml.training.complete", on_training_done, ui_root=root)
        bus.publish("ui.status", {"text": "Ready", "level": "info"})
        bus.close()   # on app shutdown
    """

    def __init__(self):
        self._ctx = zmq.Context.instance()
        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.bind(INPROC_ADDR)
        self._pub_lock = threading.Lock()
        self._subscriptions: List[Subscription] = []
        self._closed = False
        log.debug("MessageBus bound to %s", INPROC_ADDR)

    # ── Publish ─────────────────────────────────────────────────────

    def publish(self, topic: str, payload: Dict[str, Any]):
        """Publish a message on a topic (thread-safe).

        topic:  dot-delimited topic string (e.g. "data.live.price.BTC/USDT")
        payload: dict of JSON-serializable values
        """
        if self._closed:
            log.warning("publish() on closed bus, topic=%s", topic)
            return
        try:
            json_bytes = json.dumps(payload, default=str).encode("utf-8")
            with self._pub_lock:
                self._pub.send_multipart(
                    [topic.encode("utf-8"), json_bytes],
                    flags=zmq.NOBLOCK,
                )
        except zmq.ZMQError as e:
            if e.errno == zmq.EAGAIN:
                log.debug("ZMQ EAGAIN on publish (no subscribers for %s)", topic)
            else:
                log.exception("ZMQ publish error on topic %s", topic)
        except Exception:
            log.exception("publish() failed for topic %s", topic)

    def publish_obj(self, topic: str, obj: Any):
        """Publish a dataclass instance — converts via asdict()."""
        if hasattr(obj, '__dataclass_fields__'):
            self.publish(topic, asdict(obj))
        elif isinstance(obj, dict):
            self.publish(topic, obj)
        else:
            self.publish(topic, {"data": str(obj)})

    # ── Subscribe ───────────────────────────────────────────────────

    def subscribe(self, topic: str, callback: Callable[[Dict[str, Any]], None],
                  ui_root: Any = None) -> Subscription:
        """Subscribe to a topic.

        topic:    ZMQ subscription filter (prefix match). "data.live.price"
                  matches "data.live.price.BTC/USDT", "data.live.price.ETH/USDT", etc.
        callback: called with the deserialized payload dict
        ui_root:  if provided (tkinter root widget), callbacks are dispatched
                  via root.after(0, ...) for main-thread safety

        Returns a Subscription handle; call .unsubscribe() to stop.
        """
        sock = self._ctx.socket(zmq.SUB)
        sock.connect(INPROC_ADDR)
        sock.setsockopt_string(zmq.SUBSCRIBE, topic)

        if ui_root is not None:
            sub = UiSubscription(sock, None, topic, ui_root, callback)

            def _recv_loop():
                while sub._active:
                    try:
                        raw = sock.recv_multipart()
                        if len(raw) >= 2:
                            topic_str = raw[0].decode("utf-8")
                            payload = json.loads(raw[1].decode("utf-8"))
                            sub._dispatch(topic_str, payload)
                    except zmq.ZMQError as e:
                        if e.errno == zmq.ETERM:
                            break  # context terminated — normal shutdown
                        if sub._active:
                            log.debug("ZMQ recv error on topic %s: %s", topic, e)
                    except Exception:
                        if sub._active:
                            log.exception("Subscription callback error for %s", topic)
        else:
            sub = Subscription(sock, None, topic)

            def _recv_loop():
                while sub._active:
                    try:
                        raw = sock.recv_multipart()
                        if len(raw) >= 2:
                            payload = json.loads(raw[1].decode("utf-8"))
                            callback(payload)
                    except zmq.ZMQError as e:
                        if e.errno == zmq.ETERM:
                            break
                        if sub._active:
                            log.debug("ZMQ recv error on topic %s: %s", topic, e)
                    except Exception:
                        if sub._active:
                            log.exception("Subscription callback error for %s", topic)

        thread = threading.Thread(
            target=_recv_loop,
            name=f"zmq-sub-{topic}",
            daemon=True,
        )
        sub._thread = thread
        thread.start()

        self._subscriptions.append(sub)
        log.debug("Subscribed to %s (ui_root=%s)", topic, ui_root is not None)
        return sub

    # ── Shutdown ────────────────────────────────────────────────────

    def close(self):
        """Close all sockets and terminate the ZMQ context.

        Safe to call from the main thread on app shutdown.
        """
        if self._closed:
            return
        self._closed = True

        # Unsubscribe all — closes SUB sockets so recv loops see ETERM
        for sub in list(self._subscriptions):
            try:
                sub.unsubscribe()
            except Exception:
                pass
        self._subscriptions.clear()

        # Close publisher
        try:
            with self._pub_lock:
                self._pub.close()
        except Exception:
            pass

        # Terminate context — this unblocks all remaining recv_multipart() calls
        try:
            self._ctx.term()
        except Exception:
            pass

        log.debug("MessageBus closed")
