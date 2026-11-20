"""The background-review fork keeps a fallback chain (#93592).

``_fork_init_kwargs`` used to build the fork without ``fallback_model``, so the fork started with an
empty chain: a review route that resolved but then failed (429, overload, a transient 404) had no
failover and the review was lost. The fork now follows the same owner rule as delegation and cron
(``hermes_cli.fallback_config.scoped_fallback_chain``): an unrouted fork is the parent's runtime and
inherits the parent's live chain; a routed fork is a pinned owner, never borrows that chain, and
uses the documented auxiliary order ``auxiliary.background_review.fallback_chain`` -> main agent
model. Re-implements the background-review half of #79750 (Cloudymap1e) on current main.
"""
from __future__ import annotations

import logging
from unittest.mock import patch

import run_agent
import agent.background_review as bg_review
from agent.agent_init_fallback import _fallback_entries, _init_fallback_chain
from agent.background_review import build_cache_parity_fork

from tests.agent.test_background_review_cache_parity import _SyncThread, _make_agent_stub, _make_recorder_class

PARENT_CHAIN = [
    {"provider": "openrouter", "model": "parent-fallback-a"},
    {"provider": "nous", "model": "parent-fallback-b"},
]
ROUTED_RUNTIME = {
    "provider": "openrouter", "model": "aux-review-model", "api_key": "test-key",
    "base_url": None, "api_mode": None, "credential_pool": None, "request_overrides": {},
    "max_tokens": None, "command": None, "args": [], "routed": True,
}
MAIN_MODEL = {"provider": "openai", "model": "test-model"}  # the stub parent's primary


def _parent():
    agent = _make_agent_stub(run_agent.AIAgent)
    agent._fallback_chain = [dict(entry) for entry in PARENT_CHAIN]
    return agent


def _fork_kwargs(task_cfg, *, routed, agent=None, write_origin="background_review"):
    agent = agent or _parent()
    captured = {}
    runtime = dict(ROUTED_RUNTIME) if routed else {
        "provider": agent.provider, "model": agent.model, "api_key": "parent-key", "base_url": None,
        "api_mode": None, "credential_pool": None, "request_overrides": {}, "max_tokens": None,
        "command": None, "args": [], "routed": False,
    }
    with patch.object(run_agent, "AIAgent", _make_recorder_class(captured)), \
            patch.object(bg_review, "_resolve_review_runtime", return_value=runtime):
        build_cache_parity_fork(agent, task_cfg, max_iterations=5, write_origin=write_origin)
    return captured["init_kwargs"], agent


def _chain(kwargs):
    """The chain the fork AIAgent will hold, through AIAgent's own normalization."""
    return _fallback_entries(kwargs.get("fallback_model"))


def _routes(chain):
    return [(entry["provider"], entry["model"]) for entry in chain]


# --- unrouted fork: the parent's runtime, so it inherits the parent's live chain ----------------
def test_unrouted_fork_inherits_the_parents_live_chain():
    kwargs, _ = _fork_kwargs({}, routed=False)
    assert _routes(_chain(kwargs)) == _routes(PARENT_CHAIN)


def test_unrouted_fork_gets_a_copy_so_its_failover_cannot_touch_the_parent():
    kwargs, parent = _fork_kwargs({}, routed=False)
    fork_chain = kwargs.get("fallback_model")
    assert fork_chain, "the fork got no fallback_model"
    fork_chain[0]["model"] = "mutated-by-fork"
    fork_chain.append({"provider": "x", "model": "y"})
    assert parent._fallback_chain == PARENT_CHAIN


def test_unrouted_fork_with_no_parent_chain_has_none():
    agent = _parent()
    agent._fallback_chain = []
    kwargs, _ = _fork_kwargs({}, routed=False, agent=agent)
    assert _chain(kwargs) == []


def test_unrouted_fork_uses_a_declared_review_chain_instead():
    declared = [{"provider": "anthropic", "model": "review-spare"}]
    kwargs, _ = _fork_kwargs({"fallback_chain": declared}, routed=False)
    assert _routes(_chain(kwargs)) == [("anthropic", "review-spare")]


def test_unrouted_fork_with_an_empty_declared_chain_has_no_fallback():
    kwargs, _ = _fork_kwargs({"fallback_chain": []}, routed=False)
    assert _chain(kwargs) == []


def test_unrouted_fork_with_only_malformed_entries_keeps_the_inherited_chain(caplog):
    with caplog.at_level(logging.WARNING):
        kwargs, _ = _fork_kwargs({"fallback_chain": [{"model": "no-provider"}, "junk"]}, routed=False)
    assert _routes(_chain(kwargs)) == _routes(PARENT_CHAIN)
    assert "background_review fork fallback_providers has no usable routes" in caplog.text


def test_a_side_question_fork_reads_and_names_its_own_task_block(caplog):
    """/btw passes ``auxiliary.side_question`` as task_cfg: the chain and its warnings are that task's."""
    with caplog.at_level(logging.WARNING):
        kwargs, _ = _fork_kwargs({"provider": "openrouter", "model": "aux-review-model",
                                  "fallback_chain": ["junk"]}, routed=True, write_origin="side_question")
    assert _routes(_chain(kwargs)) == [(MAIN_MODEL["provider"], MAIN_MODEL["model"])]
    assert "side_question fork fallback_providers has no usable routes" in caplog.text
    assert "background_review fork" not in caplog.text


# --- routed fork: a pinned owner; its own chain, then the main agent model, never the parent's ----
def test_routed_fork_without_a_declared_chain_falls_back_to_the_main_agent_model():
    kwargs, _ = _fork_kwargs({"provider": "openrouter", "model": "aux-review-model"}, routed=True)
    assert _routes(_chain(kwargs)) == [(MAIN_MODEL["provider"], MAIN_MODEL["model"])]


def test_routed_fork_tries_its_declared_chain_then_the_main_agent_model():
    declared = [{"provider": "anthropic", "model": "review-spare"}, {"provider": "nous", "model": "spare-2"}]
    kwargs, _ = _fork_kwargs({"provider": "openrouter", "model": "aux-review-model",
                              "fallback_chain": declared}, routed=True)
    assert _routes(_chain(kwargs)) == [("anthropic", "review-spare"), ("nous", "spare-2"),
                                       (MAIN_MODEL["provider"], MAIN_MODEL["model"])]


def test_routed_fork_never_borrows_the_parents_chain():
    kwargs, _ = _fork_kwargs({"provider": "openrouter", "model": "aux-review-model"}, routed=True)
    parent_routes = set(_routes(PARENT_CHAIN))
    assert not parent_routes & set(_routes(_chain(kwargs)))


def test_routed_fork_with_an_empty_declared_chain_has_no_fallback_at_all():
    kwargs, _ = _fork_kwargs({"provider": "openrouter", "model": "aux-review-model",
                              "fallback_chain": []}, routed=True)
    assert _chain(kwargs) == []


def test_routed_fork_lists_the_main_model_once_when_the_declared_chain_already_names_it():
    declared = [{"provider": "openai", "model": "test-model"}, {"provider": "anthropic", "model": "spare"}]
    kwargs, _ = _fork_kwargs({"provider": "openrouter", "model": "aux-review-model",
                              "fallback_chain": declared}, routed=True)
    assert _routes(_chain(kwargs)) == [("openai", "test-model"), ("anthropic", "spare")]


def test_routed_fork_main_model_entry_carries_the_parent_base_url_and_no_credential():
    agent = _parent()
    agent.provider, agent.base_url, agent.api_key = "custom", "http://127.0.0.1:8080/v1", "parent-secret"
    kwargs, _ = _fork_kwargs({"provider": "openrouter", "model": "aux-review-model"}, routed=True, agent=agent)
    chain = _chain(kwargs)
    assert chain == [{"provider": "custom", "model": "test-model", "base_url": "http://127.0.0.1:8080/v1"}]


# --- the chain reaches the fork AIAgent the failover loop reads ---------------------------------
def test_the_fork_agent_holds_the_chain_its_failover_loop_walks():
    """``try_activate_fallback`` walks ``agent._fallback_chain``; AIAgent.__init__ builds it from the
    ``fallback_model`` kwarg with ``_init_fallback_chain``. Apply that same init to the kwargs."""
    kwargs, _ = _fork_kwargs({}, routed=False)
    fork = _make_agent_stub(run_agent.AIAgent)
    fork.provider, fork.model = kwargs["provider"], kwargs["model"]
    _init_fallback_chain(fork, kwargs.get("fallback_model"))
    assert _routes(fork._fallback_chain) == _routes(PARENT_CHAIN)
    assert fork._fallback_index == 0


def test_refine_and_resident_reviews_hand_the_parents_chain_to_the_fork():
    """``/refine`` (``gateway/session_cold_review.start_cold_refine``) builds its cold parent with the
    session's chain ("the reviewer inherits the chain") and then calls the same
    ``_spawn_background_review`` a resident agent uses; the spawned fork must carry that chain."""
    parent = _make_agent_stub(run_agent.AIAgent)
    _init_fallback_chain(parent, [dict(entry) for entry in PARENT_CHAIN])  # what AIAgent(fallback_model=...) does
    captured = {}
    with patch.object(run_agent, "AIAgent", _make_recorder_class(captured)), \
            patch("threading.Thread", _SyncThread), \
            patch.object(bg_review, "_background_review_task_config", return_value={}):
        parent._spawn_background_review(messages_snapshot=[], review_memory=True, review_skills=False,
                                        explicit=True)
    assert _routes(_chain(captured["init_kwargs"])) == _routes(PARENT_CHAIN)
