"""Choosing the background reviewer by fallback slot, its replay context, and naming it in the summary.

``auxiliary.background_review.model: fallback[N]`` reuses the parent's Nth fallback entry (0 = first),
so the reviewer follows the user's own capability ranking. A slot that does not exist fails loudly and
the review runs on the main model. ``context: digest|full`` overrides the replay choice (today: digest
when routed, full on the main model). A review that ran on another model says which one in its summary.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from agent import background_review as br


class _Agent:
    def __init__(self, chain=None, provider="openai-codex", model="gpt-5.5"):
        self.provider, self.model = provider, model
        self._credential_pool: Any = None
        self.request_overrides: dict = {}
        self.max_tokens = None
        self._fallback_chain = list(chain or [])
        self.notices: list[str] = []

    def _current_main_runtime(self):
        return {"api_key": "parent-key", "base_url": "https://parent.example/v1", "api_mode": "chat_completions"}

    def _emit_warning(self, message):
        self.notices.append(message)


CHAIN = [
    {"provider": "openrouter", "model": "anthropic/claude-opus-4.8"},
    {"provider": "custom", "model": "local-model", "base_url": "http://127.0.0.1:8080/v1", "key_env": "TEST_SLOT_KEY"},
]


def _resolve(agent, block, rp=None, side_effect=None):
    """Run ``_resolve_review_runtime`` with ``block`` as auxiliary.background_review; returns
    (runtime, calls made to resolve_runtime_provider)."""
    cfg = {"auxiliary": {"background_review": block}}
    calls = []

    def fake_rp(**kwargs):
        calls.append(kwargs)
        if side_effect is not None:
            raise side_effect
        return rp or {"provider": kwargs["requested"], "model": kwargs["target_model"], "api_key": "resolved-key",
                      "base_url": kwargs.get("explicit_base_url") or "https://routed.example/v1",
                      "api_mode": "chat_completions"}
    with patch("hermes_cli.config.load_config_readonly", return_value=cfg), \
            patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=fake_rp):
        return br._resolve_review_runtime(agent), calls


@pytest.mark.parametrize("slot, entry", [("fallback[0]", 0), (" Fallback[1] ", 1)])
def test_a_fallback_slot_routes_the_review_to_that_entry(monkeypatch, slot, entry):
    monkeypatch.setenv("TEST_SLOT_KEY", "slot-key-from-env")
    rt, calls = _resolve(_Agent(CHAIN), {"provider": "auto", "model": slot})
    want = CHAIN[entry]
    assert rt["routed"] is True and rt["model"] == want["model"] and rt["provider"] == want["provider"]
    (call,) = calls
    assert call["requested"] == want["provider"] and call["target_model"] == want["model"]
    assert call.get("explicit_base_url") == want.get("base_url")
    assert call.get("explicit_api_key") == (None if entry == 0 else "slot-key-from-env")


@pytest.mark.parametrize("chain, slot", [(CHAIN, "fallback[2]"), ([], "fallback[0]"), (CHAIN, "fallback[x]"),
                                         (CHAIN, "fallback[-1]"), (CHAIN, "fallback[]"),
                                         ([{"provider": "auto", "model": "x"}], "fallback[0]"),
                                         ([{"provider": " ", "model": "x"}], "fallback[0]"),
                                         ([{"provider": "openrouter", "model": " "}], "fallback[0]"),
                                         (CHAIN, "fallback[" + "9" * 5000 + "]"),
                                         (CHAIN, "fallback[0002]"),
                                         (CHAIN, "fallback[" + "0" * 5000 + "2]"),
                                         (CHAIN, "fallback[\u0661]"), (CHAIN, "fallback[\u0660\u0661]"),
                                         (CHAIN, "fallback[\uff101]"), (CHAIN, "fallback[1\u0661]"),
                                         (CHAIN, "fallback[0]junk"), (CHAIN, "fallback[0]\nx"), (CHAIN, "fallback[ 0]"),
                                         (CHAIN, "fallback[+1]"), (CHAIN, "fallbac\u212a[0]"),
                                         ([{"provider": "AUTO", "model": "x"}], "fallback[0]"),
                                         ([{"provider": " Auto ", "model": "x"}], "fallback[0]")])
def test_a_slot_that_does_not_exist_fails_loudly_and_reviews_on_the_main_model(caplog, chain, slot):
    agent = _Agent(chain)
    with caplog.at_level(logging.WARNING, logger="agent.background_review"):
        rt, calls = _resolve(agent, {"model": slot})
        _resolve(agent, {"model": slot})
    assert rt["routed"] is False and rt["model"] == "gpt-5.5" and calls == []
    shown = slot.strip()[:40].split("\n")[0]  # the log line repr()s the value, escaping a newline
    assert len(agent.notices) == 1 and shown in agent.notices[0], agent.notices
    assert sum(shown in r.getMessage() for r in caplog.records) == 2, "the log must say so on every review"


@pytest.mark.parametrize("slot, reason", [("fallback[x]", "not a fallback slot"), ("fallback[\u0661]", "not a fallback slot"),
                                          ("fallback[-1]", "not a fallback slot"), ("fallback[0]junk", "not a fallback slot"),
                                          ("fallback[2]", "no such fallback_providers entry"),
                                          ("fallback[" + "9" * 5000 + "]", "no such fallback_providers entry")])
def test_the_warning_says_whether_the_slot_is_malformed_or_missing(slot, reason):
    agent = _Agent(CHAIN)
    _resolve(agent, {"model": slot})
    assert reason in agent.notices[0], agent.notices


@pytest.mark.parametrize("slot, entry", [("fallback[001]", 1), ("fallback[00]", 0), ("fallback[1]\n", 1), ("FALLBACK[0]", 0),
                                         ("fallback[" + "0" * 5000 + "1]", 1), ("fallback[" + "0" * 5000 + "]", 0)])
def test_a_leading_zero_slot_index_selects_that_entry(slot, entry):
    """Leading zeros of any length select the entry (int() of the raw 5000-digit string would raise)."""
    rt, calls = _resolve(_Agent(CHAIN), {"model": slot})
    assert rt["routed"] is True and calls[0]["target_model"] == CHAIN[entry]["model"]


def test_a_slot_key_env_is_read_through_the_profile_scoped_secret_lookup(monkeypatch):
    """Under a multiplexed gateway a raw environment read can return another profile's key; the slot's
    key_env goes through agent.secret_scope.get_secret, like the main agent's fallback."""
    monkeypatch.setenv("TEST_SLOT_KEY", "process-env-key-of-another-profile")
    import agent.secret_scope as scope
    monkeypatch.setattr(scope, "get_secret", lambda name, *a, **k: "profile-scoped-key" if name == "TEST_SLOT_KEY" else None)
    _rt, calls = _resolve(_Agent(CHAIN), {"model": "fallback[1]"})
    assert calls[0]["explicit_api_key"] == "profile-scoped-key"


def test_a_slot_inline_api_key_is_used_over_key_env(monkeypatch):
    monkeypatch.setenv("TEST_SLOT_KEY", "env-key")
    chain = [{"provider": "custom", "model": "m", "base_url": "http://127.0.0.1:9/v1", "api_key": "inline-key",
              "key_env": "TEST_SLOT_KEY"}]
    _rt, calls = _resolve(_Agent(chain), {"model": "fallback[0]"})
    assert calls[0]["explicit_api_key"] == "inline-key" and calls[0]["explicit_base_url"] == "http://127.0.0.1:9/v1"


def test_a_slot_that_fails_to_resolve_warns_and_reviews_on_the_main_model():
    agent = _Agent(CHAIN)
    rt, _calls = _resolve(agent, {"model": "fallback[0]"}, side_effect=RuntimeError("no credentials"))
    assert rt["routed"] is False and rt["model"] == "gpt-5.5"
    assert len(agent.notices) == 1 and "no credentials" in agent.notices[0]


def test_a_slot_whose_key_lookup_fails_warns_and_reviews_on_the_main_model(monkeypatch):
    import agent.secret_scope as scope
    real = scope.get_secret

    def broken(name, *a, **k):  # only the slot's key fails; other lookups (e.g. the UI language) still work
        if name == "TEST_SLOT_KEY":
            raise RuntimeError("secret lookup failed")
        return real(name, *a, **k)
    monkeypatch.setattr(scope, "get_secret", broken)
    agent = _Agent(CHAIN)
    rt, calls = _resolve(agent, {"model": "fallback[1]"})
    assert rt["routed"] is False and rt["model"] == "gpt-5.5" and calls == []
    assert len(agent.notices) == 1 and "secret lookup failed" in agent.notices[0]


@pytest.mark.parametrize("bad_field", ["provider", "model", "base_url"])
def test_an_entry_value_that_cannot_be_read_warns_and_reviews_on_the_main_model(bad_field):
    """str() of a 5000-digit int raises (integer-string conversion limit); no entry value may abort the review."""
    entry = {"provider": "custom", "model": "local-model", "base_url": "http://127.0.0.1:9/v1"}
    entry[bad_field] = int("f" * 5000, 16)
    agent = _Agent([entry])
    rt, calls = _resolve(agent, {"model": "fallback[0]"})
    assert rt["routed"] is False and rt["model"] == "gpt-5.5" and calls == []
    assert len(agent.notices) == 1 and "fallback[0]" in agent.notices[0]


def test_a_key_lookup_failure_names_the_slot(caplog, monkeypatch):
    import agent.secret_scope as scope
    real = scope.get_secret
    monkeypatch.setattr(scope, "get_secret", lambda n, *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
                        if n == "TEST_SLOT_KEY" else real(n, *a, **k))
    with caplog.at_level(logging.WARNING, logger="agent.background_review"):
        _resolve(_Agent(CHAIN), {"model": "fallback[1]"})
    assert any("fallback[1]" in r.getMessage() and "local-model" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("block, resolver_error", [
    ({"model": "fallback[0]"}, None),  # a missing slot (empty chain)
    ({"provider": "openrouter", "model": "some/model"}, RuntimeError("no credentials")),  # the existing route
])
def test_a_failing_notice_does_not_abort_the_review(caplog, monkeypatch, block, resolver_error):
    def broken_t(*a, **k):
        raise RuntimeError("translation failed")
    monkeypatch.setattr(br, "t", broken_t)
    with caplog.at_level(logging.WARNING, logger="agent.background_review"):
        rt, _calls = _resolve(_Agent([]), block, side_effect=resolver_error)
    assert rt["routed"] is False and rt["model"] == "gpt-5.5"
    assert any("could not be resolved" in r.getMessage() for r in caplog.records), "the log line is still written"


def test_a_slot_in_the_side_question_block_resolves_through_the_shared_route():
    """Side questions (/btw) and plugin session forks resolve their runtime with the same function and their
    own task block, so a slot there means the same entry."""
    from agent import side_question as sq
    cfg = {"auxiliary": {"side_question": {"model": "fallback[1]"}}}
    calls = []

    def fake_rp(**kwargs):
        calls.append(kwargs)
        return {"provider": kwargs["requested"], "model": kwargs["target_model"], "api_key": "k",
                "base_url": kwargs.get("explicit_base_url"), "api_mode": "chat_completions"}
    with patch("hermes_cli.config.load_config_readonly", return_value=cfg), \
            patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=fake_rp):
        rt = br._resolve_review_runtime(_Agent(CHAIN), sq._side_question_task_config())
    assert rt["routed"] is True and calls[0]["target_model"] == CHAIN[1]["model"]


def test_the_ignored_provider_warning_does_not_name_the_background_review_block(caplog):
    """The resolver serves /btw and plugin forks too, so the line must not blame auxiliary.background_review."""
    with caplog.at_level(logging.WARNING, logger="agent.background_review"):
        _resolve(_Agent(CHAIN), {"provider": "nous", "model": "fallback[0]"})
    lines = [r.getMessage() for r in caplog.records if "is ignored" in r.getMessage()]
    assert lines and all("background_review" not in line and "'nous'" in line for line in lines), lines


def test_a_slot_naming_the_main_model_is_not_routed():
    agent = _Agent([{"provider": "openai-codex", "model": "gpt-5.5"}])
    rt, calls = _resolve(agent, {"model": "fallback[0]"})
    assert rt["routed"] is False and calls == []


def test_a_provider_set_beside_a_slot_is_ignored_with_a_warning_once_per_agent(caplog):
    agent = _Agent(CHAIN)
    with caplog.at_level(logging.WARNING, logger="agent.background_review"):
        first, calls = _resolve(agent, {"provider": "nous", "model": "fallback[0]"})
        second, more = _resolve(agent, {"provider": "nous", "model": "fallback[0]"})
    assert first["routed"] is second["routed"] is True
    assert calls[0]["requested"] == more[0]["requested"] == "openrouter"
    assert sum("provider" in r.getMessage() and "fallback[0]" in r.getMessage() for r in caplog.records) == 1


def _history_replayed(routed: bool, task_cfg, snapshot):
    """Run ``_run_review_fork`` with a fake fork; return the conversation_history it replayed."""
    seen = {}

    def fake_build(agent, task_cfg=None, *, max_iterations, write_origin="background_review"):
        def run_conversation(**kw):
            seen["history"] = kw["conversation_history"]
        fork = SimpleNamespace(_memory_enabled=False, _user_profile_enabled=False, run_conversation=run_conversation,
                               _session_messages=[])
        return fork, {"provider": "p", "model": "m", "routed": routed}, routed
    noop = lambda *a, **k: None
    with patch.object(br, "build_cache_parity_fork", fake_build), patch.object(br, "_track_review_fork", noop), \
            patch.object(br, "_snapshot_review_usage", lambda a: {}), \
            patch.object(br, "_record_review_usage_to_parent", noop), \
            patch.object(br, "finish_background_review_run", noop), patch.object(br, "_release_fork_clients", noop):
        br._run_review_fork(SimpleNamespace(), snapshot, "p", task_cfg, None, br._ReviewForkState())
    return seen["history"]


SNAPSHOT = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i}"} for i in range(60)]


@pytest.mark.parametrize("routed, context, full", [
    (False, None, True), (True, None, False), (False, "auto", True), (True, "auto", False),
    (False, "digest", False), (True, "full", True), (True, "FULL", True), (False, "Digest ", False),
])
def test_the_replay_context_follows_the_setting(routed, context, full):
    task_cfg = {} if context is None else {"context": context}
    history = _history_replayed(routed, task_cfg, SNAPSHOT)
    assert (history == SNAPSHOT) is full
    if not full:
        assert history == br._digest_history(SNAPSHOT)


@pytest.mark.parametrize("bad", ["everything", 3, ["full"]])
def test_an_invalid_context_warns_and_keeps_todays_choice(caplog, bad):
    with caplog.at_level(logging.WARNING, logger="agent.background_review"):
        history = _history_replayed(True, {"context": bad}, SNAPSHOT)
    assert history == br._digest_history(SNAPSHOT)
    assert any("context" in r.getMessage() for r in caplog.records)


def _summary_lines(rt: dict, routed: bool):
    """Run a whole review in-thread with a fake fork whose run reports one skill action; return
    (cli line, callback line)."""
    printed, called = [], []
    agent = SimpleNamespace(_safe_print=printed.append, background_review_callback=called.append,
                            memory_notifications="on", provider="openai-codex", model="gpt-5.5",
                            _emit_auxiliary_failure=lambda *a, **k: None)

    def fake_build(agent, task_cfg=None, *, max_iterations, write_origin="background_review"):
        fork = SimpleNamespace(_memory_enabled=False, _user_profile_enabled=False,
                               run_conversation=lambda **kw: None, _session_messages=[])
        return fork, rt, routed
    noop = lambda *a, **k: None
    with patch.object(br, "build_cache_parity_fork", fake_build), patch.object(br, "_track_review_fork", noop), \
            patch.object(br, "_snapshot_review_usage", lambda a: {}), \
            patch.object(br, "_record_review_usage_to_parent", noop), \
            patch.object(br, "finish_background_review_run", noop), patch.object(br, "_release_fork_clients", noop), \
            patch.object(br, "_parent_can_emit_tool_calls", lambda a: True), \
            patch.object(br, "summarize_background_review_actions", lambda *a, **k: ["Skill 'demo' patched"]):
        br._run_review_in_thread(agent, [], "p", {})
    return printed[-1], called[-1]


def test_a_routed_review_names_its_reviewer_in_the_summary():
    cli, callback = _summary_lines({"provider": "openrouter", "model": "anthropic/claude-opus-4.8", "routed": True},
                                   routed=True)
    for line in (cli, callback):
        assert "Skill 'demo' patched" in line
        assert "anthropic/claude-opus-4.8" in line and "openrouter" in line, line


def test_a_main_model_review_summary_is_unchanged():
    from agent.i18n import t
    cli, callback = _summary_lines({"provider": "openai-codex", "model": "gpt-5.5", "routed": False}, routed=False)
    assert cli == t("display.review.summary_cli", summary="Skill 'demo' patched")
    assert callback == t("display.review.summary_callback", summary="Skill 'demo' patched")
