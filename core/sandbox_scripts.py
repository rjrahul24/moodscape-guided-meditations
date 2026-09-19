"""Pre-written benchmark scripts for TTS model evaluation in the MoodScape Sandbox.

Provides two rigorously calibrated 5-minute benchmark scripts:
1. Guided Meditation: Spaced mindfulness guidance with authentic meditation markers
   ([pause:Xs], [breath]), testing pause naturalness, soft articulation, and silence handling.
2. Sleep Story: Continuous, descriptive bedtime storytelling with gentle pacing,
   testing narrative rhythm, vocal warmth, and sleep-bed ducking over extended paragraphs.
"""

from __future__ import annotations

# ── 5-Minute Guided Meditation Benchmark Script ──────────────────────────────
# Pacing breakdown: ~250 spoken words (~145s speech @ 100 WPM) + ~155s timed pauses
# Total duration: ~300 seconds (~5.0 minutes).
MEDITATION_5MIN_SCRIPT = """\
Welcome. Take a moment to settle comfortably into your space. [pause:4s]

Allow your posture to feel upright, yet completely relaxed. Soften your hands in your lap, and gently close your eyes. [pause:6s]

Let us begin with a slow, intentional breath. Take a deep breath in through your nose... [breath] [pause:3s] and exhale softly through your mouth. [pause:6s]

Once more, a gentle, full breath in... [breath] [pause:3s] and release, letting go of any tension you have carried today. [pause:8s]

Now, allow your breath to find its own natural rhythm. There is nothing you need to control or force. [pause:6s]

Notice the subtle sensation of the breath as it enters your body... cool at the tip of the nose... warm as it leaves. [pause:8s]

Bring gentle awareness to the crown of your head. Feel the forehead softening... smoothing out any lines of worry. [pause:5s]

Release your jaw, letting your tongue rest softly. Let your shoulders drop away from your ears, releasing their weight. [pause:8s]

Feel your chest and abdomen gently rising and falling. Like quiet waves lapping against a peaceful shore. [pause:10s]

Notice your arms... your hands... resting heavy, warm, and still. [pause:8s]

Feel the support beneath you, holding you safely in this present moment. Grounded. Supported. At ease. [pause:10s]

If thoughts arise, simply observe them like clouds drifting across an open sky. You do not need to follow them. [pause:6s]

Gently guide your attention back to the breath... here... and now. [pause:12s]

Rest here in this open, quiet stillness. There is nowhere else you need to be. Nothing else you need to do. [pause:15s]

Simply being. Whole, peaceful, and present. [pause:10s]

As our practice comes to a close, take one more deep, nourishing breath in... [breath] [pause:4s] and let it go. [pause:6s]

Begin to bring gentle movement back to your fingers and toes. [pause:5s]

When you feel ready, softly open your eyes. Carry this quiet clarity and peace with you into the rest of your day. [pause:4s]\
"""

# ── 5-Minute Sleep Story Benchmark Script ─────────────────────────────────────
# Pacing breakdown: ~475 spoken words (~255s speech @ 110 WPM) + ~45s subtle pauses
# Total duration: ~300 seconds (~5.0 minutes).
SLEEP_STORY_5MIN_SCRIPT = """\
Welcome to tonight's sleep journey: The Sanctuary of Whispering Pines. [pause:3s]

Make yourself completely comfortable beneath the blankets. Feel the pillow cradling your head, and allow your eyes to close softly. [pause:4s]

Take a slow, comforting breath in... and let your body sink a little deeper into the warmth of your bed. [pause:5s]

Imagine yourself standing at the edge of a serene mountain trail just as twilight settles over the land. The sky above is painted in shades of deep indigo, soft violet, and dusted with the first faint stars of the evening. [pause:4s]

A gentle, cooling breeze carries the sweet, grounding fragrance of cedar needles and fresh pine. With every step forward along the moss-covered path, your footsteps are quiet and cushioned, completely absorbed by the soft earth below. [pause:3s]

The evening air is still and peaceful. High in the canopy above, the tall pine trees sway gently in the breeze, whispering a soft, soothing lullaby that seems to say: it is time to rest. [pause:5s]

You follow the winding path down toward a secluded valley, where a glassy alpine lake rests in undisturbed quiet. The surface of the water is smooth as dark silver, mirroring the crescent moon and the vast constellation of stars scattered across the night sky. [pause:4s]

Near the water's edge stands a cozy wooden cabin, its windows glowing with a warm, amber lantern light. You step inside, feeling the welcoming comfort of the room. A stone hearth holds glowing embers, radiating a soft, gentle warmth that fills the quiet space. [pause:4s]

In the corner, an inviting daybed is piled high with plush down comforters and freshly laundered linens. You lie down and draw the blankets up to your shoulders. [pause:4s]

From here, you can see through the window into the tranquil night. The silver lake... the distant silhouette of the mountains... the gentle swaying of the pines. [pause:4s]

Every muscle in your body softens. Your eyelids feel heavy, comfortably weighted. The quiet murmur of the wind outside weaves together with the warmth of the fire, creating a peaceful cocoon of safety and comfort. [pause:5s]

With each breath you take, the thoughts of the day dissolve into the stillness of the valley. Drifting farther and farther away. [pause:4s]

There is nothing to worry about. Nothing left to do. You are safe, warm, and completely at peace. [pause:5s]

Allow yourself to drift now... effortlessly sinking deeper... floating upon this tranquil river of dreams. Deep, peaceful, restorative sleep... wrapping around you... like the quiet night sky. [pause:6s]

Rest peacefully... and sleep well. [pause:4s]\
"""

BENCHMARK_SCRIPTS = {
    "meditation": MEDITATION_5MIN_SCRIPT,
    "sleep_story": SLEEP_STORY_5MIN_SCRIPT,
}

BENCHMARK_METADATA = {
    "meditation": {
        "title": "5-Minute Guided Meditation Benchmark",
        "content_type": "meditation",
        "target_duration_sec": 300,
        "word_count": len(MEDITATION_5MIN_SCRIPT.split()),
        "description": "Mindfulness body scan with authentic pause and breath cues. Tests pause naturalness, soft articulation, and silence handling.",
    },
    "sleep_story": {
        "title": "5-Minute Sleep Story Benchmark (The Sanctuary of Whispering Pines)",
        "content_type": "sleep_story",
        "target_duration_sec": 300,
        "word_count": len(SLEEP_STORY_5MIN_SCRIPT.split()),
        "description": "Continuous narrative bedtime story with hypnotic cadence and subtle pauses. Tests vocal warmth, phrasing rhythm, and ambient bed ducking.",
    },
}


def get_benchmark_script(content_type: str = "meditation") -> str:
    """Return the pre-written 5-minute benchmark script for the given content type.

    Args:
        content_type: "meditation" (or "Guided Meditation") or
                      "sleep_story" (or "Sleep Story").

    Returns:
        The benchmark script string.
    """
    from core.content_profiles import normalize_content_type
    key = normalize_content_type(content_type)
    return BENCHMARK_SCRIPTS.get(key, MEDITATION_5MIN_SCRIPT)


def get_script_metadata(content_type: str = "meditation") -> dict:
    """Return metadata (word count, target duration, description) for the benchmark script."""
    from core.content_profiles import normalize_content_type
    key = normalize_content_type(content_type)
    return BENCHMARK_METADATA.get(key, BENCHMARK_METADATA["meditation"])
