# SPDX-License-Identifier: Apache-2.0
"""
Regression tests for MooncakeLookupClient.

These tests guard against known bugs in the MooncakeLookupClient,
notably the infinite-retry loop caused by inheriting the base class
default of ``lookup_cache() -> None`` (which the scheduler interprets
as "still ongoing").
"""

# Third Party
import pytest


# Skip the whole module if the mooncake package is not installed.
# MooncakeLookupClient imports ``from mooncake.store import
# MooncakeDistributedStore`` inside __init__, so the import itself does
# not require mooncake, but instantiation does.
mooncake = pytest.importorskip("mooncake.store")


def test_lookup_cache_returns_minus_one_not_none():
    """
    Regression test for the infinite-loop bug.

    ``MooncakeLookupClient`` is a synchronous lookup client and must
    override ``lookup_cache()`` to return -1 ("not found").  If it
    inherits the base class default (``None`` = "ongoing"), the vLLM
    scheduler interprets every probe as "still ongoing" and retries
    forever without ever calling the real ``lookup()``.
    """
    # First Party
    from lmcache.v1.lookup_client.mooncake_lookup_client import (
        MooncakeLookupClient,
    )

    # Skip Mooncake store setup by constructing without __init__ side
    # effects.  The test targets only the lookup_cache() method.
    client = MooncakeLookupClient.__new__(MooncakeLookupClient)

    result = client.lookup_cache(lookup_id="test-request-id")

    assert result == -1, (
        "MooncakeLookupClient.lookup_cache() must return -1 "
        "(not found).  Returning None (the base class default) "
        "causes the scheduler to retry indefinitely."
    )
    assert result is not None


def test_lookup_cache_override_does_not_raise_for_any_lookup_id():
    """``lookup_cache()`` must never raise — it is called per request."""
    # First Party
    from lmcache.v1.lookup_client.mooncake_lookup_client import (
        MooncakeLookupClient,
    )

    client = MooncakeLookupClient.__new__(MooncakeLookupClient)

    for lookup_id in ("req-0", "req-1", "", "with-dashes-123"):
        assert client.lookup_cache(lookup_id=lookup_id) == -1
