"""``auxiliary.background_review.skill_scope: all|local`` (#156124 item 2).

``all`` (the default, and unset) keeps today's behavior. ``local`` keeps the profile's own skills
folder writable to forked writers (the background review, /refine, the curator's consolidation fork,
plugin forks) and makes every other skill root proposal-only to them: a write whose resolved skill
folder lies outside the profile's skills folder is staged in the pending store, whatever
``skills.write_approval`` says. The user's own turns are never affected.
"""

import json

import pytest

FORK = "background_review"


def _skill_md(name):
    return f"---\nname: {name}\ndescription: Use when testing skill scope.\n---\n\n# {name}\n\nStep one.\n"


@pytest.fixture
def roots(tmp_path, monkeypatch):
    """A profile with one local skill, one skill in an external root, and config written fresh per test."""
    home = tmp_path / ".hermes"
    local = home / "skills"
    (local / "own-skill").mkdir(parents=True)
    (local / "own-skill" / "SKILL.md").write_text(_skill_md("own-skill"), encoding="utf-8")
    vault = tmp_path / "vault"
    (vault / "shared-skill" / "references").mkdir(parents=True)
    (vault / "shared-skill" / "SKILL.md").write_text(_skill_md("shared-skill"), encoding="utf-8")
    (vault / "shared-skill" / "references" / "ref.md").write_text("Ref.\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    import hermes_constants
    monkeypatch.setattr(hermes_constants, "_hermes_home_cache", None, raising=False)
    from tools import skills_tool
    import tools.skill_manager_tool as smt
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", local)
    monkeypatch.setattr(smt, "SKILLS_DIR", local)
    from tools.skill_manager_guards import _reset_background_review_read_marks
    _reset_background_review_read_marks()
    yield {"home": home, "local": local, "vault": vault, "tmp": tmp_path}
    _reset_background_review_read_marks()
    from agent import skill_utils as su
    su._external_dirs_cache_clear()


def _config(roots, *, scope=None, approval=False, create_dir=None, raw_scope_line=None):
    lines = ["skills:", "  external_dirs:", f"    - {roots['vault']}", f"  write_approval: {str(approval).lower()}"]
    if create_dir is not None:
        lines.append(f"  create_dir: {create_dir}")
    if scope is not None or raw_scope_line is not None:
        lines += ["auxiliary:", "  background_review:", raw_scope_line or f"    skill_scope: {scope}"]
    (roots["home"] / "config.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    from agent import skill_utils as su
    su._raw_config_cache_clear()
    su._external_dirs_cache_clear()


def _as(origin, fn):
    from tools.skill_provenance import reset_current_write_origin, set_current_write_origin
    token = set_current_write_origin(origin)
    try:
        return fn()
    finally:
        reset_current_write_origin(token)


def _manage(**kwargs):
    from tools.skill_manager_tool import skill_manage
    return json.loads(skill_manage(**kwargs))


def _view(name, file_path=None):
    from tools.skills_tool import skill_view
    out = json.loads(skill_view(name, file_path) if file_path else skill_view(name))
    assert out["success"] is True, out


def _pending():
    from tools import write_approval as wa
    return wa.list_pending(wa.SKILLS)


# action -> (kwargs, file it changes relative to the skill folder, needs a read first)
WRITES = {
    "edit": (dict(action="edit", content=_skill_md("x").replace("Step one.", "Rewritten.")), "SKILL.md"),
    "patch": (dict(action="patch", old_string="Step one.", new_string="Step two."), "SKILL.md"),
    "write_file": (dict(action="write_file", file_path="references/ref.md", file_content="New.\n"),
                   "references/ref.md"),
    "remove_file": (dict(action="remove_file", file_path="references/ref.md"), "references/ref.md"),
}


def _give_own_skill_a_reference(roots):
    (roots["local"] / "own-skill" / "references").mkdir(exist_ok=True)
    (roots["local"] / "own-skill" / "references" / "ref.md").write_text("Ref.\n", encoding="utf-8")


def _fork_write(roots, skill, action):
    kwargs, rel = WRITES[action]

    def run():
        _view(skill, rel if rel != "SKILL.md" else None)
        return _manage(name=skill, **kwargs)
    return _as(FORK, run)


def _snapshot(folder):
    return {p.relative_to(folder).as_posix(): p.read_bytes() for p in sorted(folder.rglob("*")) if p.is_file()}


@pytest.mark.parametrize("scope", [None, "all"])
@pytest.mark.parametrize("action", sorted(WRITES))
def test_all_or_unset_lets_a_fork_write_a_shared_skill_as_today(roots, scope, action):
    _config(roots, scope=scope)
    before = _snapshot(roots["vault"] / "shared-skill")
    out = _fork_write(roots, "shared-skill", action)
    assert out.get("success") is True and not out.get("staged"), out
    assert _snapshot(roots["vault"] / "shared-skill") != before
    assert _pending() == []


@pytest.mark.parametrize("action", sorted(WRITES))
def test_local_stages_a_forks_write_to_a_shared_skill(roots, action):
    _config(roots, scope="local")
    before = _snapshot(roots["vault"] / "shared-skill")
    out = _fork_write(roots, "shared-skill", action)
    assert out.get("success") is True and out.get("staged") is True, out
    assert "skill_scope" in out.get("message", ""), out
    assert _snapshot(roots["vault"] / "shared-skill") == before
    [record] = _pending()
    assert record["payload"]["action"] == action and record["payload"]["name"] == "shared-skill"


@pytest.mark.parametrize("action", sorted(WRITES))
def test_local_lets_a_fork_write_the_profiles_own_skill(roots, action):
    _config(roots, scope="local")
    _give_own_skill_a_reference(roots)
    before = _snapshot(roots["local"] / "own-skill")
    out = _fork_write(roots, "own-skill", action)
    assert out.get("success") is True and not out.get("staged"), out
    assert _snapshot(roots["local"] / "own-skill") != before
    assert _pending() == []


def test_local_stages_a_forks_create_into_a_create_dir_outside_the_profile(roots):
    create_dir = roots["tmp"] / "fleet"
    create_dir.mkdir()
    _config(roots, scope="local", create_dir=create_dir)
    out = _as(FORK, lambda: _manage(action="create", name="new-skill", content=_skill_md("new-skill")))
    assert out.get("staged") is True, out
    assert not (create_dir / "new-skill").exists()
    assert len(_pending()) == 1


def test_local_lets_a_fork_create_in_the_profiles_own_folder(roots):
    _config(roots, scope="local")
    out = _as(FORK, lambda: _manage(action="create", name="new-skill", content=_skill_md("new-skill")))
    assert out.get("success") is True and not out.get("staged"), out
    assert (roots["local"] / "new-skill" / "SKILL.md").exists()


@pytest.mark.parametrize("origin", ["foreground", "assistant_tool"])
def test_local_never_touches_the_users_own_writes(roots, origin):
    _config(roots, scope="local")
    out = _as(origin, lambda: _manage(action="patch", name="shared-skill",
                                      old_string="Step one.", new_string="Step two."))
    assert out.get("success") is True and not out.get("staged"), out
    assert "Step two." in (roots["vault"] / "shared-skill" / "SKILL.md").read_text(encoding="utf-8")


def test_local_stages_a_batch_with_one_shared_op_as_one_record(roots):
    _config(roots, scope="local")
    own_before = _snapshot(roots["local"] / "own-skill")
    shared_before = _snapshot(roots["vault"] / "shared-skill")

    def run():
        _view("own-skill")
        _view("shared-skill")
        return _manage(action="", name="", operations=[
            {"action": "patch", "name": "own-skill", "old_string": "Step one.", "new_string": "Own."},
            {"action": "patch", "name": "shared-skill", "old_string": "Step one.", "new_string": "Shared."}])
    out = _as(FORK, run)
    assert out.get("staged") is True, out
    assert _snapshot(roots["local"] / "own-skill") == own_before
    assert _snapshot(roots["vault"] / "shared-skill") == shared_before
    [record] = _pending()
    assert record["payload"]["action"] == "batch"


def test_local_lets_an_all_local_batch_through(roots):
    _config(roots, scope="local")

    def run():
        _view("own-skill")
        return _manage(action="", name="", operations=[
            {"action": "patch", "name": "own-skill", "old_string": "Step one.", "new_string": "Own."}])
    out = _as(FORK, run)
    assert out.get("success") is True and not out.get("staged"), out
    assert _pending() == []


def test_local_counts_skills_in_a_symlinked_profile_skills_folder_as_local(roots, monkeypatch):
    """Fleets often point a profile's skills folder at a shared tree with a symlink; that folder is
    still the profile's own, so its skills stay writable (resolved paths on both sides)."""
    real = roots["tmp"] / "real-skills"
    roots["local"].rename(real)
    roots["local"].symlink_to(real, target_is_directory=True)
    _config(roots, scope="local")
    before = _snapshot(real / "own-skill")
    out = _fork_write(roots, "own-skill", "patch")
    assert out.get("success") is True and not out.get("staged"), out
    assert _snapshot(real / "own-skill") != before


def test_approving_a_scope_staged_write_applies_it(roots):
    _config(roots, scope="local")
    out = _fork_write(roots, "shared-skill", "patch")
    assert out.get("staged") is True, out
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa
    reply = handle_pending_subcommand(wa.SKILLS, ["approve", out["pending_id"]])
    assert reply is not None
    assert "Step two." in (roots["vault"] / "shared-skill" / "SKILL.md").read_text(encoding="utf-8"), reply
    assert _pending() == []


@pytest.mark.parametrize("raw, staged", [
    ("local", True), ("' LOCAL '", True), ("all", False), ("''", False), ("null", False),
    ("bogus", True), ("123", True), ("[local]", True)])
def test_skill_scope_values(roots, raw, staged, caplog):
    """``local`` in any case/spacing stages; ``all``, empty and null keep today's behavior; any other
    value (a typo, a number, a list) reads as ``local``: never less staging by accident. Only those
    other values are warned about."""
    _config(roots, raw_scope_line=f"    skill_scope: {raw}")
    with caplog.at_level("WARNING", logger="tools.write_approval"):
        out = _fork_write(roots, "shared-skill", "patch")
    assert bool(out.get("staged")) is staged, out
    unknown = raw in {"bogus", "123", "[local]"}
    assert ("Unknown auxiliary.background_review.skill_scope" in caplog.text) is unknown, caplog.text


def test_an_unreadable_config_reads_as_local(roots, monkeypatch):
    """If the setting cannot be read, the fork's shared write is staged, not applied blind."""
    _config(roots)
    import hermes_cli.config as cfg

    def unreadable():
        raise OSError("config unreadable")
    _as(FORK, lambda: _view("shared-skill"))
    monkeypatch.setattr(cfg, "load_config", unreadable)
    out = _as(FORK, lambda: _manage(action="patch", name="shared-skill", old_string="Step one.", new_string="Two."))
    assert out.get("staged") is True, out


def test_local_with_write_approval_on_stages_a_shared_write_once(roots):
    _config(roots, scope="local", approval=True)
    out = _fork_write(roots, "shared-skill", "patch")
    assert out.get("staged") is True, out
    assert len(_pending()) == 1


def test_local_covers_a_fork_origin_it_has_never_heard_of(roots):
    _config(roots, scope="local")
    out = _as("plugin_session_fork", lambda: _manage(
        action="patch", name="shared-skill", old_string="Step one.", new_string="Two."))
    assert out.get("staged") is True, out


def test_local_stages_nothing_for_a_skill_that_does_not_exist(roots):
    _config(roots, scope="local")
    out = _as(FORK, lambda: _manage(action="patch", name="no-such-skill", old_string="a", new_string="b"))
    assert not out.get("staged") and out.get("success") is False, out
    assert _pending() == []


def test_local_keeps_refusing_a_forks_delete_of_a_shared_skill(roots):
    """Deletes of externally owned skills were already refused to autonomous passes; the refusal runs
    before the staging gate and stays a refusal, not a proposal."""
    _config(roots, scope="local")
    out = _as(FORK, lambda: _manage(action="delete", name="shared-skill"))
    assert out.get("success") is False and not out.get("staged"), out
    assert (roots["vault"] / "shared-skill" / "SKILL.md").exists()
    assert _pending() == []


def test_a_batch_cannot_reach_an_outside_skill_by_removing_the_local_copy_that_shadowed_it(roots):
    """Both ops resolve to the local copy when the batch is checked; the first removes that copy's
    SKILL.md, so the second's name would now resolve to the external copy. Each write is re-checked
    where it lands: the second op is refused and the batch rolls back, the external skill untouched."""
    _config(roots, scope="local")
    (roots["local"] / "shared-skill").mkdir()
    (roots["local"] / "shared-skill" / "SKILL.md").write_text(_skill_md("shared-skill"), encoding="utf-8")
    shared_before = _snapshot(roots["vault"] / "shared-skill")
    local_before = _snapshot(roots["local"] / "shared-skill")

    def run():
        _view("shared-skill")
        return _manage(action="", name="", operations=[
            {"action": "remove_file", "name": "shared-skill", "file_path": "SKILL.md"},
            {"action": "write_file", "name": "shared-skill", "file_path": "references/ref.md",
             "file_content": "Unapproved.\n"}])
    out = _as("plugin_session_fork", run)
    assert out.get("success") is False and not out.get("staged"), out
    assert _snapshot(roots["vault"] / "shared-skill") == shared_before
    assert _snapshot(roots["local"] / "shared-skill") == local_before
    assert _pending() == []


@pytest.mark.parametrize("action", sorted(WRITES) + ["create"])
def test_the_handler_refuses_an_outside_write_the_gate_did_not_stage(roots, monkeypatch, action):
    """The handler checks the folder it resolved, on its own: with the gate out of the way, a fork's
    write to an outside skill (or a create outside the profile) is refused, not applied."""
    create_dir = roots["tmp"] / "fleet"
    _config(roots, scope="local", create_dir=create_dir if action == "create" else None)
    import tools.skill_manager_tool as smt
    monkeypatch.setattr(smt, "_run_write_gate", lambda *args, **kwargs: None)
    before = _snapshot(roots["vault"] / "shared-skill")
    if action == "create":
        out = _as(FORK, lambda: _manage(action="create", name="new-skill", content=_skill_md("new-skill")))
        assert not (create_dir / "new-skill").exists()
    else:
        out = _fork_write(roots, "shared-skill", action)
    assert out.get("success") is False and out.get("_skill_scope_refused") is True, out
    assert _snapshot(roots["vault"] / "shared-skill") == before


def test_a_name_that_resolves_elsewhere_by_the_time_the_handler_looks_is_refused(roots, monkeypatch):
    """Gate and checks see the local copy; then, before the handler looks the name up, another writer
    removes that copy's SKILL.md (through a display name, so it holds a different lock), revealing
    the external skill. The handler checks the folder it actually resolved and refuses."""
    _config(roots, scope="local")
    (roots["local"] / "shared-skill").mkdir()
    (roots["local"] / "shared-skill" / "SKILL.md").write_text(_skill_md("shared-skill"), encoding="utf-8")
    import tools.skill_manager_tool as smt
    real_locate = smt._locate_for_write

    def concurrent_remove_then_locate(name, action, *args, **kwargs):
        (roots["local"] / "shared-skill" / "SKILL.md").unlink(missing_ok=True)  # the other writer lands here
        return real_locate(name, action, *args, **kwargs)
    monkeypatch.setattr(smt, "_locate_for_write", concurrent_remove_then_locate)
    before = _snapshot(roots["vault"] / "shared-skill")
    out = _as(FORK, lambda: _manage(action="write_file", name="shared-skill", file_path="references/new.md",
                                    file_content="Unapproved.\n"))
    assert out.get("success") is False and out.get("_skill_scope_refused") is True, out
    assert _snapshot(roots["vault"] / "shared-skill") == before
    assert _pending() == []


def _curator_managed(name):
    from tools import skill_usage
    skill_usage.record_created(name, agent_created=True)


def test_local_stages_a_forks_delete_of_a_curator_managed_skill_outside_the_profile(roots):
    """A delete the autonomous delete guards allow (a curator-managed skill, absorbed into an umbrella)
    follows the same rule as content writes: outside the profile it is staged, not applied."""
    create_dir = roots["tmp"] / "fleet"
    (create_dir / "old-skill").mkdir(parents=True)
    (create_dir / "old-skill" / "SKILL.md").write_text(_skill_md("old-skill"), encoding="utf-8")
    _config(roots, scope="local", create_dir=create_dir)
    _curator_managed("old-skill")
    out = _as(FORK, lambda: _manage(action="delete", name="old-skill", absorbed_into="own-skill"))
    assert out.get("staged") is True, out
    assert (create_dir / "old-skill" / "SKILL.md").exists()
    [record] = _pending()
    assert record["payload"]["action"] == "delete"


def test_local_lets_a_forks_permitted_delete_inside_the_profile_through(roots):
    (roots["local"] / "old-skill").mkdir()
    (roots["local"] / "old-skill" / "SKILL.md").write_text(_skill_md("old-skill"), encoding="utf-8")
    _config(roots, scope="local")
    _curator_managed("old-skill")
    out = _as(FORK, lambda: _manage(action="delete", name="old-skill", absorbed_into="own-skill"))
    assert out.get("success") is True and not out.get("staged"), out
    assert not (roots["local"] / "old-skill").exists()
    assert _pending() == []


def test_the_handler_refuses_a_forks_delete_outside_the_profile_the_gate_did_not_stage(roots, monkeypatch):
    """A delete the autonomous delete guards admit (curator-managed, absorbed into an umbrella), with
    the gate out of the way: the delete handler checks the folder it resolved and refuses."""
    create_dir = roots["tmp"] / "fleet"
    (create_dir / "old-skill").mkdir(parents=True)
    (create_dir / "old-skill" / "SKILL.md").write_text(_skill_md("old-skill"), encoding="utf-8")
    _config(roots, scope="local", create_dir=create_dir)
    _curator_managed("old-skill")
    import tools.skill_manager_tool as smt
    monkeypatch.setattr(smt, "_run_write_gate", lambda *args, **kwargs: None)
    before = _snapshot(create_dir)
    out = _as(FORK, lambda: _manage(action="delete", name="old-skill", absorbed_into="own-skill"))
    assert out.get("success") is False and out.get("_skill_scope_refused") is True, out
    assert _snapshot(create_dir) == before
    assert _pending() == []


def test_a_refused_batch_op_leaves_an_outside_skill_another_writer_changed_alone(roots, monkeypatch):
    """The gate sees the local copy; before the batch snapshots, another writer removes that copy's
    SKILL.md (through a display name, another lock), so the batch snapshots the external skill. After
    the snapshot, before the op runs, a writer changes the external skill. The handler refuses the
    fork's op there, and the rollback (which restores only folders the batch changed) leaves that
    writer's change alone."""
    _config(roots, scope="local")
    (roots["local"] / "shared-skill").mkdir()
    (roots["local"] / "shared-skill" / "SKILL.md").write_text(_skill_md("shared-skill"), encoding="utf-8")
    import tools.skill_manager_batch as smb
    real_snapshot = smb._snapshot_skills
    other = roots["vault"] / "shared-skill" / "notes.md"

    def shadow_removed_before_snapshot_then_external_changed(*args, **kwargs):
        (roots["local"] / "shared-skill" / "SKILL.md").unlink(missing_ok=True)
        result = real_snapshot(*args, **kwargs)
        other.write_text("Another writer's change.\n", encoding="utf-8")
        return result
    monkeypatch.setattr(smb, "_snapshot_skills", shadow_removed_before_snapshot_then_external_changed)
    out = _as(FORK, lambda: _manage(action="", name="", operations=[
        {"action": "write_file", "name": "shared-skill", "file_path": "references/new.md", "file_content": "Unapproved.\n"}]))
    assert out.get("success") is False and out.get("failed_index") == 0, out
    assert "_skill_scope_refused" in json.dumps(out) or "outside this profile" in out.get("error", ""), out
    assert other.exists() and other.read_bytes() == b"Another writer's change.\n"
    assert not (roots["vault"] / "shared-skill" / "references" / "new.md").exists()
    assert _pending() == []

