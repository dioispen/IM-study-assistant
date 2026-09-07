"""Lazy OpenAI client construction, shared by the embedder and generator so
importing either module doesn't require an API key unless it's actually used --
and the one operational failure both of them report.

The failure type is the pipeline's; the translation into it is this provider's.
A surface renders an operational failure as a message rather than a traceback
(#27), and it can only do that against a name it knows: a page that caught
`openai.OpenAIError` would be a page that had to import the provider SDK in
order to describe the pipeline it is a front end for, and it would go silent the
day a second provider raised something else. So the SDK's errors are translated
here, at the one boundary that already knows which provider is being reached.
"""

from contextlib import contextmanager


class ProviderFailure(Exception):
    """The language model could not be reached, or would not answer.

    An operational failure and never a statement about the corpus: it is
    neither an answer nor an Abstention, and a surface that let it read as one
    would tell a reader their notes do not cover a question that was never
    asked (ADR-0003 has the two Abstentions, and this is neither).

    Deliberately not a RuntimeError, which is what config.py raises when a tau
    is paired with the wrong embedding model. That one is a misconfiguration
    rather than a hiccup and must keep failing loudly at startup, so it has to
    be uncatchable by whoever catches this -- and a class that is not an
    ancestor of it cannot be caught by mistake, however the except clauses
    around a call site are later reordered.
    """


@contextmanager
def provider_failures():
    """Everything the OpenAI SDK raises, as one ProviderFailure.

    `openai.OpenAIError` is the SDK's base class: the missing-key error the
    client constructor raises is that class itself, and every call-time failure
    -- a rate limit, a connection error, a revoked key, a 500 -- is one of its
    subclasses. Catching the base rather than a list of them is what makes this
    hold for the failure modes not thought of here, and there is nothing a
    surface would do differently for one of them anyway: the reader is told the
    provider could not be reached, in the provider's own words.

    The message is kept as the provider wrote it, because it is the half that
    says what to do next -- "set the OPENAI_API_KEY environment variable" is an
    instruction, and re-wording it here would cost the reader that. The cause is
    chained rather than dropped, so a console still gets the real traceback
    under whatever a surface chose to show.

    Imported inside the manager for the reason `default_openai_client` imports
    inside itself: importing this module must stay free of the SDK, so the
    offline suite and anything under core can import it without openai being
    reachable.
    """
    from openai import OpenAIError

    try:
        yield
    except OpenAIError as error:
        raise ProviderFailure(str(error)) from error


def default_openai_client():
    with provider_failures():
        from openai import OpenAI

        return OpenAI()
