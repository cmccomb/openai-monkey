from __future__ import annotations
import os, sys
from .config import load_config
from .adapter import apply_adapter_patch

CFG = load_config()

_SUPPORTED_AUTH_TYPES = {"basic", "bearer"}
if CFG["auth_type"] not in _SUPPORTED_AUTH_TYPES:
    raise ValueError(
        f"Unsupported OPENAI_AUTH_TYPE '{CFG['auth_type']}'. Supported types: {sorted(_SUPPORTED_AUTH_TYPES)}"
    )

apply_adapter_patch(
    base_url=CFG["base_url"],
    auth_type=CFG["auth_type"],
    token=CFG["token"],
    path_map=CFG["path_map"],
    param_map=CFG["param_map"],
    drop_params=CFG["drop_params"],
    extra_allow=CFG["extra_allow"],
    model_routes=CFG["model_routes"],
    disable_streaming=CFG["disable_streaming"],
    default_headers=CFG["default_headers"],
)

import openai as _openai  # patched

OpenAI = _openai.OpenAI
AsyncOpenAI = getattr(_openai, "AsyncOpenAI", None)
__all__ = ["OpenAI", "AsyncOpenAI"]

for name in (
    "APIConnectionError", "APIStatusError", "AuthenticationError",
    "RateLimitError", "BadRequestError", "APIError",
    "resources", "types", "responses", "chat", "embeddings", "images", "audio",
):
    if hasattr(_openai, name):
        globals()[name] = getattr(_openai, name)
        __all__.append(name)

if os.getenv("OPENAI_BASIC_ALIAS_OPENAI", "0") not in ("", "0", "false", "False"):
    sys.modules.setdefault("openai", sys.modules[__name__])
