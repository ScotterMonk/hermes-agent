"""Where a ``skill_manage`` write lands, for ``auxiliary.background_review.skill_scope``.

``local`` keeps a forked writer's writes inside the profile's own skills folder, in two places:
the approval gate stages a write it can see will land elsewhere (the proposal flow), and every
handler checks the folder it resolved, right where it resolves it and before writing anything
(``refuse_outside``, called from ``_locate_for_write`` and ``_create_skill``). The handler then
writes into that same folder, so the folder checked is the folder written: a name that resolves
somewhere else by then (a shadowing local copy removed by an earlier batch op or by another writer)
is refused, not followed. The skill lookups only run for a forked writer under ``local``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Iterable, Optional

logger = logging.getLogger(__name__)


def target_folder(action: str, name: str, category: Optional[str] = None) -> Optional[Path]:
    """The skill folder a mutation would change: where ``create`` puts it, else the existing skill
    (None when it cannot be found)."""
    from tools import skill_manager_tool as smt
    if action == "create":
        return smt._resolve_skill_dir(name, category)
    found = smt._find_skill(name) if name else None
    return found["path"] if found else None


def outside_profile_skills(folder: Optional[Path]) -> bool:
    """Whether ``folder`` lies outside this profile's own skills folder. Both sides are resolved, so a
    profile whose skills folder is a symlink still owns its skills. A skill that cannot be found is
    not shared: its action fails with the usual error and there is nothing to stage. A folder that
    cannot be resolved is treated as shared: staging beats writing a root we cannot place."""
    if folder is None:
        return False
    from tools import skill_manager_tool as smt
    try:
        return not folder.resolve().is_relative_to(smt._skills_dir().resolve())
    except (OSError, RuntimeError):
        logger.warning("could not place skill folder %s; treating it as shared", folder, exc_info=True)
        return True


def shared_target_check(
        targets: Optional[Iterable[tuple[str, str, Optional[str]]]]) -> Optional[Callable[[], bool]]:
    """A deferred check: does any ``(action, name, category)`` target lie outside the profile's own
    skills folder? None when there are no targets."""
    items = list(targets or ())
    if not items:
        return None
    return lambda: any(outside_profile_skills(target_folder(*t)) for t in items)


def refuse_outside(action: str, name: str, skill_dir: Path) -> Optional[dict]:
    """Refusal for a forked writer's write into ``skill_dir`` (the folder the handler resolved and is
    about to write) when it lies outside the profile under ``local``, else None. Approved replays of
    staged writes run as the approving user, so they are not covered."""
    from tools import write_approval as wa
    if not wa.is_forked_writer() or wa.background_skill_scope() != wa.SKILL_SCOPE_LOCAL:
        return None
    if not outside_profile_skills(Path(skill_dir)):
        return None
    return {"success": False, "_skill_scope_refused": True, "error": (
        f"Refusing {action} for skill '{name}': auxiliary.background_review.skill_scope is local and "
        f"'{name}' resolves to {skill_dir}, outside this profile's skills folder (it changed after the "
        f"write was checked). Nothing was written.")}
