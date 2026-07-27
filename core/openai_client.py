"""Lazy OpenAI client construction, shared by the embedder and generator so
importing either module doesn't require an API key unless it's actually used.
"""


def default_openai_client():
    from openai import OpenAI

    return OpenAI()
