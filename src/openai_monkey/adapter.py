"""Runtime adapter that monkeypatches the official ``openai`` client.

The :func:`apply_adapter_patch` function swaps the HTTP layer used by the
official `openai` package so requests are transparently redirected to an
internal endpoint.  The helpers defined within the module focus on translating
between the public OpenAI API surface and the internal provider so callers do
not need to change their application code.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping, Sequence
from typing import Any, cast


def apply_adapter_patch(
    *,
    base_url: str,
    auth_type: str,
    token: str,
    path_map: dict[str, str],
    param_map: dict[str, str],
    drop_params: set[str],
    extra_allow: set[str],
    model_routes: dict[str, dict[str, Any]] | None,
    disable_streaming: bool,
    default_headers: dict[str, str] | None,
) -> bool:
    """Patch ``openai`` to issue requests through the configured adapter.

    The patch works by replacing the ``OpenAI``/``AsyncOpenAI`` client classes
    and selected resource methods with thin wrappers.  These wrappers translate
    keyword arguments, map request bodies to the internal schema, and route
    calls to the HTTP endpoints described by ``path_map`` and
    ``model_routes``.

    Args:
        base_url: Fully qualified base URL for the internal provider.
        auth_type: Authentication scheme used to build the ``Authorization``
            header (``"basic"`` or ``"bearer"``).
        token: Credential associated with ``auth_type``.
        path_map: Mapping of OpenAI REST paths to internal paths.  Streaming
            variants are suffixed with ``":stream"``.
        param_map: Mapping of keyword parameters that must be renamed when
            forwarding to the internal API.
        drop_params: Parameters that should be removed before forwarding the
            request.
        extra_allow: Parameters that should bypass validation when interacting
            with the internal API.
        model_routes: Overrides for particular model identifiers.  Regular
            expression keys can redirect traffic to custom paths.
        disable_streaming: When ``True`` the adapter converts streaming requests
            into synchronous calls.
        default_headers: Additional HTTP headers that should be attached to
            outgoing requests.

    Returns:
        ``True`` when the patch applied successfully.

    Examples:
        >>> apply_adapter_patch(
        ...     base_url="https://internal.example",  # doctest: +SKIP
        ...     auth_type="basic",
        ...     token="token",
        ...     path_map={"/responses": "/responses"},
        ...     param_map={},
        ...     drop_params=set(),
        ...     extra_allow=set(),
        ...     model_routes={},
        ...     disable_streaming=False,
        ...     default_headers=None,
        ... )
        True
    """
    import httpx
    import openai

    # Helper utilities -----------------------------------------------------
    # The following closures operate on the configuration above to bridge the
    # gap between the public ``openai`` API and the internal provider.

    def _mk_headers():
        """Construct the HTTP headers used for every proxied request."""

        scheme = "basic" if not auth_type else auth_type.lower()
        if scheme == "basic":
            value = f"Basic {token}"
        elif scheme == "bearer":
            value = f"Bearer {token}"
        else:
            raise ValueError(f"Unsupported auth type: {auth_type}")
        h = {"Authorization": value}
        if default_headers:
            h.update(default_headers)
        return h

    def _rewrite_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
        """Rewrite keyword arguments so the internal API understands them."""

        out: dict[str, Any] = {}
        for k, v in kwargs.items():
            if k in drop_params and k not in extra_allow:
                continue
            if k in extra_allow:
                out[k] = v
                continue
            mk = param_map.get(k, k)
            out[mk] = v
        return out

    def _stringify_message_content(content: Any) -> str:
        """Return a text representation for ``content``.

        The adapter only supports text-based chat payloads.  Multimodal
        structures are normalised by concatenating all ``"text"`` parts while
        raising an explicit error for unsupported entries.  This avoids leaking
        Python ``repr`` strings to the model when the upstream caller passes
        complex message objects.

        Args:
            content: Raw ``content`` value from the OpenAI chat message.

        Returns:
            The text contained within ``content`` joined by newlines when the
            payload uses the list-of-parts encoding.

        Raises:
            TypeError: If ``content`` contains non-textual elements that the
                adapter cannot translate.
        """

        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, Mapping):
            part_type = content.get("type")
            text = content.get("text")
            if part_type == "text" and isinstance(text, str):
                return text
            raise TypeError(
                f"Unsupported chat message content part type: {part_type!r}"
            )
        if isinstance(content, Sequence) and not isinstance(
            content, (str, bytes, bytearray)
        ):
            parts = [_stringify_message_content(item) for item in content]
            return "\n".join(part for part in parts if part)
        raise TypeError("Chat message content must be text or a sequence of text parts")

    def _messages_to_prompt(messages):
        """Flatten chat messages into the prompt format required internally."""

        lines = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            try:
                text = _stringify_message_content(content)
            except TypeError as exc:
                raise TypeError(
                    f"Unsupported chat message content for role {role!r}: {exc}"
                ) from exc
            lines.append(f"{role.upper()}: {text}")
        lines.append("ASSISTANT:")
        return "\n".join(lines)

    def _build_payload(model: str, content: Any, **kwargs) -> dict[str, Any]:
        """Compose the JSON body expected by the internal inference endpoint."""

        pay: dict[str, Any] = {"model": model, "input": content}
        if "temperature" in kwargs:
            pay[param_map.get("temperature", "temperature")] = kwargs["temperature"]
        filtered = {
            k: v for k, v in kwargs.items() if k not in {"stream", "temperature"}
        }
        pay.update(_rewrite_kwargs(filtered))
        return pay

    def _normalize_sync(data: dict[str, Any]) -> dict[str, Any]:
        """Normalize synchronous responses to mimic ``openai`` semantics."""

        result = data.get("result") or {}
        text = (
            result.get("text")
            or data.get("text")
            or (
                data.get("choices", [{}])[0].get("message", {}).get("content")
                if isinstance(data.get("choices"), list)
                else ""
            )
            or ""
        )
        usage = data.get("usage") or {}
        return {
            "id": data.get("id", f"resp-{int(time.time() * 1000)}"),
            "model": data.get("model"),
            "output_text": text,
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            },
        }

    def _normalize_stream_line(line: bytes | str):
        """Translate streaming payloads into OpenAI-compatible events."""

        if not line:
            return None
        if isinstance(line, bytes):
            try:
                line = line.decode("utf-8", "ignore")
            except Exception:
                return None
        raw = line.strip()
        if not raw:
            return None
        if raw.startswith("data:"):
            raw = raw[5:].strip()
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return None
        tpe = msg.get("type")
        if tpe == "delta":
            return {
                "type": "response.delta",
                "delta": {"output_text": msg.get("text", "")},
            }
        if tpe == "done":
            return {"type": "response.completed"}
        if "text" in msg:
            return {"type": "response.delta", "delta": {"output_text": msg["text"]}}
        return None

    def _route_for(model: str, base_path: str, *, stream: bool) -> str:
        """Choose the target path for *model* optionally considering streaming."""

        if model_routes:
            for pat, cfg in model_routes.items():
                try:
                    if re.fullmatch(pat, model):
                        p = cfg.get("path")
                        if p:
                            return p
                except re.error:
                    pass
        key = f"{base_path}:stream" if stream else base_path
        return path_map.get(key, base_path)

    # Patch constructors ----------------------------------------------------
    # We wrap the official ``OpenAI`` client classes so they default to using
    # our HTTP clients.  This ensures every consumer—synchronous or
    # asynchronous—transparently communicates with the internal base URL.
    _OrigOpenAI = cast(type[Any], openai.OpenAI)
    _OrigAsyncOpenAI = cast(type[Any] | None, getattr(openai, "AsyncOpenAI", None))

    def _mk_httpx_client():
        """Create the default ``httpx.Client`` used for patched calls."""

        return httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=_mk_headers(),
            timeout=httpx.Timeout(60.0),
        )

    class PatchedOpenAI(_OrigOpenAI):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            """Initialize the patched client with adapter-aware defaults."""

            kwargs.setdefault("base_url", base_url.rstrip("/"))
            kwargs.setdefault("api_key", "x")  # ignored
            if "http_client" not in kwargs or kwargs["http_client"] is None:
                kwargs["http_client"] = _mk_httpx_client()
            super().__init__(*args, **kwargs)

    PatchedAsyncOpenAI: type[Any] | None
    if _OrigAsyncOpenAI is not None:
        import httpx as _httpx_async

        class _PatchedAsyncOpenAI(_OrigAsyncOpenAI):  # type: ignore[misc, valid-type]
            def __init__(self, *args, **kwargs):
                """Initialize the async client to communicate via the adapter."""

                kwargs.setdefault("base_url", base_url.rstrip("/"))
                kwargs.setdefault("api_key", "x")
                if "http_client" not in kwargs or kwargs["http_client"] is None:
                    kwargs["http_client"] = _httpx_async.AsyncClient(
                        base_url=base_url.rstrip("/"),
                        headers=_mk_headers(),
                        timeout=_httpx_async.Timeout(60.0),
                    )
                super().__init__(*args, **kwargs)

        PatchedAsyncOpenAI = _PatchedAsyncOpenAI

    else:
        PatchedAsyncOpenAI = None

    setattr(openai, "OpenAI", PatchedOpenAI)
    if PatchedAsyncOpenAI is not None:
        setattr(openai, "AsyncOpenAI", PatchedAsyncOpenAI)

    # Patch resources -------------------------------------------------------
    # Resource-specific methods are monkeypatched so their HTTP layer uses the
    # helpers above.  The replacements keep the public method signatures while
    # quietly rewriting payloads and normalizing responses.
    from openai.resources.responses import Responses as _Responses

    _orig_resp_create = _Responses.create

    def _patched_resp_create(self, *args, **kwargs):
        """Proxy ``Responses.create`` through the internal HTTP client."""

        model = kwargs.pop("model")
        input_ = kwargs.pop("input")
        stream = bool(kwargs.pop("stream", False))
        if disable_streaming:
            stream = False
        payload = _build_payload(model, input_, **kwargs)
        path = _route_for(model, "/responses", stream=stream)

        client = getattr(self, "_client", None) or getattr(self, "client", None)
        http = getattr(client, "_client", None)
        if not http:
            return _orig_resp_create(self, *args, **kwargs)

        if stream:

            def _event_iterator():
                """Yield streaming events using the internal HTTP endpoint."""

                with http.stream("POST", path, json=payload) as r:
                    r.raise_for_status()
                    for line in r.iter_lines():
                        ev = _normalize_stream_line(line)
                        if ev:
                            yield ev

            return _event_iterator()

        r = http.post(path, json=payload)
        r.raise_for_status()
        return _normalize_sync(r.json())

    setattr(_Responses, "create", _patched_resp_create)

    try:
        from openai.resources.chat.completions import (
            Completions as _ImportedChatCompletions,
        )
    except Exception:
        _ChatCompletions: type[Any] | None = None
    else:
        _ChatCompletions = cast(type[Any], _ImportedChatCompletions)

    if _ChatCompletions is not None:
        _orig_chat_create = _ChatCompletions.create

        def _patched_chat_create(self, *args, **kwargs):
            """Proxy ``ChatCompletions.create`` to the internal HTTP endpoint."""

            model = kwargs.pop("model")
            messages = kwargs.pop("messages")
            stream = bool(kwargs.pop("stream", False))
            if disable_streaming:
                stream = False
            input_ = _messages_to_prompt(messages)
            payload = _build_payload(model, input_, **kwargs)
            path = _route_for(model, "/chat/completions", stream=stream)

            client = getattr(self, "_client", None) or getattr(self, "client", None)
            http = getattr(client, "_client", None)
            if not http:
                return _orig_chat_create(self, *args, **kwargs)

            if stream:
                with http.stream("POST", path, json=payload) as r:
                    r.raise_for_status()
                    for line in r.iter_lines():
                        ev = _normalize_stream_line(line)
                        if ev:
                            yield ev
            else:
                r = http.post(path, json=payload)
                r.raise_for_status()
                return _normalize_sync(r.json())

        setattr(_ChatCompletions, "create", _patched_chat_create)

    return True
