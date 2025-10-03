from __future__ import annotations
import json, re, time
from typing import Any, Dict, Optional, Set

def apply_adapter_patch(
    *,
    base_url: str,
    basic_token: str,
    path_map: Dict[str, str],
    param_map: Dict[str, str],
    drop_params: Set[str],
    extra_allow: Set[str],
    model_routes: Dict[str, Dict[str, Any]] | None,
    disable_streaming: bool,
    default_headers: Dict[str, str] | None,
) -> bool:
    import httpx
    import openai

    def _mk_headers():
        # Basic token is used AS-IS after the word 'Basic '
        h = {"Authorization": f"Basic {basic_token}"}
        if default_headers:
            h.update(default_headers)
        return h

    def _rewrite_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in kwargs.items():
            if k in drop_params:
                continue
            mk = param_map.get(k, k)
            out[mk] = v
        return out

    def _messages_to_prompt(messages):
        lines = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            lines.append(f"{role.upper()}: {content}")
        lines.append("ASSISTANT:")
        return "\n".join(lines)

    def _build_payload(model: str, content: Any, **kwargs) -> Dict[str, Any]:
        pay: Dict[str, Any] = {"model": model, "input": content}
        if "temperature" in kwargs:
            pay[param_map.get("temperature", "temperature")] = kwargs["temperature"]
        filtered = {k: v for k, v in kwargs.items() if k not in {"stream", "temperature"}}
        pay.update(_rewrite_kwargs(filtered))
        return pay

    def _normalize_sync(data: Dict[str, Any]) -> Dict[str, Any]:
        result = data.get("result") or {}
        text = (
            result.get("text")
            or data.get("text")
            or (data.get("choices", [{}])[0].get("message", {}).get("content") if isinstance(data.get("choices"), list) else "")
            or ""
        )
        usage = data.get("usage") or {}
        return {
            "id": data.get("id", f"resp-{int(time.time()*1000)}"),
            "model": data.get("model"),
            "output_text": text,
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            },
        }

    def _normalize_stream_line(line: bytes | str):
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
            return {"type": "response.delta", "delta": {"output_text": msg.get("text", "")}}
        if tpe == "done":
            return {"type": "response.completed"}
        if "text" in msg:
            return {"type": "response.delta", "delta": {"output_text": msg["text"]}}
        return None

    def _route_for(model: str, base_path: str, *, stream: bool) -> str:
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

    # Patch constructors
    _OrigOpenAI = openai.OpenAI
    _OrigAsyncOpenAI = getattr(openai, "AsyncOpenAI", None)

    def _mk_httpx_client():
        return httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=_mk_headers(),
            timeout=httpx.Timeout(60.0),
        )

    class PatchedOpenAI(_OrigOpenAI):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("base_url", base_url.rstrip("/"))
            kwargs.setdefault("api_key", "x")  # ignored
            if "http_client" not in kwargs or kwargs["http_client"] is None:
                kwargs["http_client"] = _mk_httpx_client()
            super().__init__(*args, **kwargs)

    if _OrigAsyncOpenAI:
        import httpx as _httpx_async
        class PatchedAsyncOpenAI(_OrigAsyncOpenAI):
            def __init__(self, *args, **kwargs):
                kwargs.setdefault("base_url", base_url.rstrip("/"))
                kwargs.setdefault("api_key", "x")
                if "http_client" not in kwargs or kwargs["http_client"] is None:
                    kwargs["http_client"] = _httpx_async.AsyncClient(
                        base_url=base_url.rstrip("/"),
                        headers=_mk_headers(),
                        timeout=_httpx_async.Timeout(60.0),
                    )
                super().__init__(*args, **kwargs)
    else:
        PatchedAsyncOpenAI = None

    openai.OpenAI = PatchedOpenAI
    if PatchedAsyncOpenAI:
        openai.AsyncOpenAI = PatchedAsyncOpenAI

    # Patch resources
    from openai.resources.responses import Responses as _Responses
    _orig_resp_create = _Responses.create

    def _patched_resp_create(self, *args, **kwargs):
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

    _Responses.create = _patched_resp_create

    try:
        from openai.resources.chat.completions import Completions as _ChatCompletions
    except Exception:
        _ChatCompletions = None

    if _ChatCompletions:
        _orig_chat_create = _ChatCompletions.create
        def _patched_chat_create(self, *args, **kwargs):
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

        _ChatCompletions.create = _patched_chat_create

    return True
