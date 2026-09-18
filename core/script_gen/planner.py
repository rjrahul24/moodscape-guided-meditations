"""Pass 0: turn a genre pack and one angle into a prose creative brief.

The brief is plain prose with nothing to parse. Everything the pipeline needs
to act on deterministically -- content_type, music_tags, pause_ratio -- is
already decided by the pack, so the planner is free to be purely creative and
there is no structured-output failure mode to handle.
"""

from collections.abc import Sequence

from core.genres import Angle, GenrePack
from core.script_gen.engine import ScriptEngine
from core.script_gen.generator import strip_wrapper


def _bullet(items: Sequence[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def plan(
    engine: ScriptEngine,
    pack: GenrePack,
    angle: Angle,
    *,
    system: str,
    target_min_sec: float,
    target_max_sec: float,
    avoid: Sequence[str] = (),
    steer: str = "",
    max_tokens: int = 1024,
) -> str:
    """Produce a creative brief for one run of one genre.

    Only the chosen angle is shown. Handing the model every angle at once
    produces a blend of all of them, which is both worse and less varied than
    committing to one.
    """
    sections = [
        f"Genre: {pack.label} ({pack.family})",
        f"Technique:\n{pack.technique}",
        f"Session arc:\n{_bullet(pack.arc)}",
        f"Angle for this session: {angle.name}",
        f"Imagery to build on:\n{_bullet(angle.imagery)}",
        (
            f"Silence budget: about {pack.pause_ratio:.0%} of the runtime "
            "should be silence held by [pause:Xs] markers. Say in the brief "
            "where the long pauses belong."
        ),
        f"Genre safety notes:\n{pack.safety}",
    ]

    if pack.banned:
        sections.append(
            "Never use these phrasings:\n" + _bullet(pack.banned)
        )

    if avoid:
        sections.append(
            "Do not reuse these images or phrases — recent meditations in "
            "this genre already used them:\n" + _bullet(avoid)
        )

    if steer.strip():
        sections.append(f"Additional request from the listener:\n{steer.strip()}")

    user = (
        "Write the creative brief for one session.\n\n"
        + "\n\n".join(sections)
        + "\n\nOutput only the brief."
    )

    return strip_wrapper(engine.complete(system, user, max_tokens=max_tokens))
