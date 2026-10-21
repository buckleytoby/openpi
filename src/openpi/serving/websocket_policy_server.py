import asyncio
import http
import logging
import time
import traceback
from typing import Callable, Optional

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)

# Sentinel key: a received message containing this key is treated as a
# control command (e.g. "reload the policy") rather than an inference
# request. Regular inference obs dicts never contain this key, so this is
# fully backward-compatible with existing clients/servers that don't know
# about control messages at all.
CONTROL_KEY = "__control__"


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods, plus an optional
    control-message channel (see CONTROL_KEY / on_control) for hot-swapping
    the served policy without restarting the server process.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
        on_control: Optional[Callable[[dict], dict]] = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        # Optional callback: on_control(command_dict) -> response_dict.
        # Handles anything sent under CONTROL_KEY -- e.g. "list_checkpoints",
        # "load_checkpoint", "current_checkpoint" (see scripts/serve_policy.py's
        # dexnex-specific handler). If unset, control messages get an
        # "unsupported" error response rather than being run through
        # self._policy.infer() (which would fail confusingly on an obs dict
        # that isn't real observation data).
        self._on_control = on_control
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def set_on_control(self, on_control: Optional[Callable[[dict], dict]]) -> None:
        """Set (or clear, with None) the control-message handler after
        construction -- useful when the handler itself needs a reference to
        this server instance (e.g. to later call set_policy), which isn't
        available yet at __init__ time."""
        self._on_control = on_control

    def set_policy(self, policy: _base_policy.BasePolicy, metadata: dict | None = None) -> None:
        """Hot-swap the served policy. Safe to call from a control handler
        invoked by _handler -- asyncio's single-threaded event loop means
        this can only ever happen between (never during) a policy.infer()
        call, so in-flight inference is never disrupted by a concurrent
        swap. Only affects requests handled after this call returns."""
        self._policy = policy
        if metadata is not None:
            self._metadata = metadata

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
            # (dexnex) no keepalive pings: inference runs inside this event loop, so a long JAX compile (e.g. the first
            # call with a new __num_samples) can't answer them and the client would drop the connection
            ping_interval=None,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                if CONTROL_KEY in obs:
                    if self._on_control is not None:
                        response = self._on_control(obs[CONTROL_KEY])
                    else:
                        response = {"success": False, "message": "Server has no control handler configured."}
                    await websocket.send(packer.pack(response))
                    continue

                if self._policy is None:
                    # Briefly true mid-reload (see set_policy(None) callers,
                    # e.g. scripts/serve_policy.py's _ReloadHandler, which
                    # frees the old policy's GPU memory before building the
                    # new one) -- report a clean error instead of crashing
                    # with AttributeError on None.infer(...). Sent as a bare
                    # string (not packed), matching the existing "string
                    # response = error" convention websocket_client_policy.py's
                    # infer() already checks for (see the except-Exception
                    # branch at the bottom of this handler).
                    await websocket.send("Server is reloading its policy -- try again shortly.")
                    continue

                infer_time = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
