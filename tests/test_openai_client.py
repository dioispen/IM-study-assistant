"""The provider boundary: what the OpenAI path does when the provider will not
answer.

The page renders an operational failure as a message rather than a traceback
(#27), and it does that by catching the pipeline's own ProviderFailure. A seam
test with a double that raises one proves the rendering; these prove there is
something to render -- that the real path raises a ProviderFailure rather than
letting an SDK exception escape to a surface that was never taught its name.

Offline, and needs no key: no key is needed to be told a key is missing, and
the two call cases inject a client that fails instead of reaching the network.
"""

import openai
import pytest
from core.embedder import OpenAIEmbedder
from core.generator import OpenAIGenerator
from core.openai_client import ProviderFailure, default_openai_client

# The provider refusing a call, as the SDK's own base error. Raised rather than
# one of its subclasses because what the translation promises is that *every*
# error the SDK defines arrives as a ProviderFailure -- a rate limit, a
# connection error and a revoked key are all this class, and catching the base
# is what makes the promise hold for the ones not thought of here.
PROVIDER_ERROR = "Connection error."

MODEL = "text-embedding-3-small"


class _FailingCalls:
    """Every call on this client fails, whichever namespace it is reached
    through -- `embeddings.create` and `chat.completions.create` are the two
    the pipeline makes."""

    def create(self, **kwargs):
        raise openai.OpenAIError(PROVIDER_ERROR)

    def __getattr__(self, name):
        return self


FAILING_CLIENT = _FailingCalls()


def test_a_missing_api_key_is_a_provider_failure(monkeypatch):
    # The client is constructed before anything is asked of it, so this is the
    # failure a page meets while it is being opened. It reaches the reader as
    # the SDK's own words, which are the half that says what to do about it.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ProviderFailure) as raised:
        default_openai_client()

    assert "OPENAI_API_KEY" in str(raised.value)


def test_an_embedding_call_that_fails_is_a_provider_failure():
    embedder = OpenAIEmbedder(model=MODEL, client=FAILING_CLIENT)

    with pytest.raises(ProviderFailure) as raised:
        embedder.embed(["what is a hash collision"])

    assert PROVIDER_ERROR in str(raised.value)


def test_a_generation_call_that_fails_is_a_provider_failure():
    generator = OpenAIGenerator(model=MODEL, client=FAILING_CLIENT)

    with pytest.raises(ProviderFailure) as raised:
        generator.generate("Question: what is a hash collision")

    assert PROVIDER_ERROR in str(raised.value)


def test_a_provider_failure_keeps_the_error_it_was_translated_from():
    # Chained rather than replaced: a surface shows the reader the message, and
    # whoever is looking at a console still gets the SDK's own traceback under
    # it. Losing the cause would make an outage and a bug in this repo look
    # alike to whoever has to fix one of them.
    embedder = OpenAIEmbedder(model=MODEL, client=FAILING_CLIENT)

    with pytest.raises(ProviderFailure) as raised:
        embedder.embed(["what is a hash collision"])

    assert isinstance(raised.value.__cause__, openai.OpenAIError)


def test_a_provider_failure_is_not_a_runtime_error():
    # The threshold-to-embedding-model guard in config.py raises RuntimeError
    # and the page catches ProviderFailure, so the two must not be one
    # another's subclass: a misconfigured tau is not an operational hiccup to
    # render politely (ADR-0003, #27), and what keeps it uncatchable there is
    # the class hierarchy rather than the order of any except clause.
    assert not issubclass(ProviderFailure, RuntimeError)
    assert issubclass(ProviderFailure, Exception)
