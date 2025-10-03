"""Example application for Bearer authentication with openai_monkey."""

import os

import openai_monkey as openai


def main() -> None:
    """Run a minimal request using Bearer auth."""

    os.environ.setdefault("OPENAI_AUTH_TYPE", "bearer")
    os.environ.setdefault("OPENAI_BASE_URL", "https://internal.company.ai")
    os.environ.setdefault("OPENAI_TOKEN", "REPLACE_WITH_BEARER")

    client = openai.OpenAI()
    response = client.responses.create(model="demo-model", input="ping", max_tokens=8)
    print(response["output_text"])


if __name__ == "__main__":
    main()
