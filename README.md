# openai-basic-compat (Basic-token mode)

This variant treats your **API key** as a **Basic token**. No username/password are stored or used.

## Two ways to use

### 1) Explicit (recommended)
```python
import openai_monkey as openai
client = openai.OpenAI()
print(client.responses.create(model="gpt-4o-mini", input="ping")["output_text"])
```

### 2) Stealth mode (alias to `openai`)
```bash
export OPENAI_BASIC_ALIAS_OPENAI=1
python your_app.py
```

## Configuration (env)

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `OPENAI_AUTH_TYPE` | ✅ | `basic` | Selects the authentication scheme. Set to `basic` to send a Basic token or `bearer` to send a Bearer token. |
| `OPENAI_BASE_URL` | ✅ | — | Base URL of the upstream API. Used to build request URLs regardless of auth type. |
| `OPENAI_TOKEN` | ✅ | — | Raw credential appended to the `Authorization` header. Interpreted as either a Basic or Bearer token depending on `OPENAI_AUTH_TYPE`. |
| `OPENAI_BASIC_BASE_URL` | ⛔️ | — | Legacy override for `OPENAI_BASE_URL`. Checked only when the primary variable is unset. |
| `OPENAI_BASIC_TOKEN` | ⛔️ | — | Legacy override for `OPENAI_TOKEN`. Checked only when the primary variable is unset. |
| `OPENAI_BEARER_TOKEN` / `OPENAI_API_KEY` / `OPENAI_KEY` | ⛔️ | — | Additional fallbacks that populate `OPENAI_TOKEN` when neither the primary nor legacy variables are present. |
| `OPENAI_BASIC_PATH_MAP` | ⛔️ | `{}` | JSON object remapping request paths (e.g., `{"/responses": "/api/generate"}`). |
| `OPENAI_BASIC_PARAM_MAP` | ⛔️ | `{}` | JSON object translating payload parameter names before forwarding. |
| `OPENAI_BASIC_DROP_PARAMS` | ⛔️ | `[]` | JSON array listing payload parameters to remove entirely. |
| `OPENAI_BASIC_EXTRA_ALLOW` | ⛔️ | `[]` | JSON array of extra payload fields allowed to pass through without filtering. |
| `OPENAI_BASIC_MODEL_ROUTES` | ⛔️ | `{}` | JSON object for model-specific overrides such as custom paths. |
| `OPENAI_BASIC_HEADERS` | ⛔️ | `{}` | JSON object of additional headers to merge into every outgoing request. |
| `OPENAI_BASIC_DISABLE_STREAMING` | ⛔️ | `0` | Set to `1` to disable streaming even when the client requests it. |

**Usage examples**

Minimal (Basic auth, default):
```bash
export OPENAI_AUTH_TYPE="basic"
export OPENAI_BASE_URL="https://internal.company.ai"
export OPENAI_TOKEN="$YOUR_BASIC_TOKEN"   # sends: Authorization: Basic $YOUR_BASIC_TOKEN
# The token must be configured; the adapter refuses to start with placeholder values.
```

Bearer support:
```bash
export OPENAI_AUTH_TYPE="bearer"
export OPENAI_BASE_URL="https://internal.company.ai"
export OPENAI_TOKEN="$YOUR_BEARER_TOKEN"   # sends: Authorization: Bearer $YOUR_BEARER_TOKEN
```

## Examples
- `examples/minimal_app.py` demonstrates the default Basic flow.
- `examples/bearer_app.py` shows how to call the adapter with Bearer tokens.

## CLI helpers

Installing the package exposes two helper commands:

- `openai-monkey-ify`: Recursively rewrites `import openai` statements in a
  repository so they load `openai_monkey` instead. Run it from the root of the
  project you want to update:

  ```bash
  openai-monkey-ify  # defaults to the current directory
  ```

  Pass `--dry-run` to preview which files would change without editing them.

- `openai-monkey-install-openai`: Creates a `.pth` alias so `import openai`
  automatically resolves to `openai_monkey` in the active Python environment:

  ```bash
  openai-monkey-install-openai
  ```

  Use `--site-packages=/custom/path` to target a specific environment.

## Development workflow

The repository ships with formatter, linter, and type-checker defaults that
target `src/openai_monkey` and `tests`. Install the development dependencies and
pre-commit hooks to keep your changes consistent:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[test]
pip install pre-commit
pre-commit install
```

Once configured, the hooks run the following tools automatically on staged
files:

| Tool | Command | Purpose |
| --- | --- | --- |
| Black | `black` | Enforces formatting. |
| Ruff | `ruff format` and `ruff check --fix` | Applies import sorting and lint fixes. |
| MyPy | `mypy src/openai_monkey tests` | Enforces strict static typing. |

Run them manually across the full codebase before pushing:

```bash
ruff format src/openai_monkey tests
ruff check src/openai_monkey tests
black src/openai_monkey tests
mypy
pytest
```
