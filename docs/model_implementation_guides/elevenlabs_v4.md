# ElevenLabs Eleven v4 (cloud TTS provider)

Code: `core/elevenlabs_tts/` (`client.py`, `engine.py`, `preprocessor.py`, `direction.py`, `assembly.py`, `chunk_cache.py`, `postprocessor.py`, `voices.py`). Voice list: `assets/elevenlabs/voices.toml`. Selected by the top-of-app **Voice Provider** toggle (`core/provider_ui.py`); the local engines (Kokoro, F5-TTS, Chatterbox) remain the default.

## What v4 is

Released 2026-09-28. Sources: [What is Eleven v4](https://elevenlabs.io/docs/help-center/product/core-capabilities/text-to-speech/what-is-eleven-v4), [Audio tags v3/v4](https://elevenlabs.io/docs/help-center/product/core-capabilities/text-to-speech/how-do-audio-tags-work-with-eleven-v3-and-v4), [Eleven v4 blog](https://elevenlabs.io/blog/eleven-v4), [TTS best practices](https://elevenlabs.io/docs/overview/capabilities/text-to-speech/best-practices), [Models](https://elevenlabs.io/docs/overview/models), [TTS convert API](https://elevenlabs.io/docs/api-reference/text-to-speech/convert), [Morphic v4 guide (3rd party)](https://morphic.com/resources/how-to/eleven-v4-guide).

- Model IDs: `eleven_v4` (long-form/narration) and `eleven_v4_turbo` (real-time). MoodScape defaults to `eleven_v4`.
- 10,000 characters per request (v3: 5,000); 90+ languages.
- Only **Stability** and **Similarity** apply. Style and Speed sliders do not exist in v4 and SSML is unsupported, so all pacing comes from text (direction, punctuation, ellipses) and programmatic silence.
- Audio tags are free-text in square brackets; delivery is best-effort and voice-dependent. Descriptive voice-quality phrasing (`[low, gravelly voice]`) avoids being read as a sound effect.
- Request stitching is back (v3 rejected it) and is "significantly more reliable in v4". The API takes up to 3 `previous_request_ids`.
- `seed` (0-4294967295) gives best-effort determinism. `output_format` includes `pcm_48000`.
- One secondary source says older IVC/PVC clones may need retraining for v4; preflight voice validation surfaces this per voice.

## How MoodScape uses it

- **API key**: read from `ELEVENLABS_API_KEY`, falling back to `ELEVEN_LABS_API_KEY`. Preflight fails fast (before any LLM/TTS work) on a missing key or a voice the account cannot use.
- **Chunking**: the script is split into chunks of about `MOODSCAPE_ELEVENLABS_CHUNK_MAX_CHARS` (1500), well under the 10,000 limit.
- **Direction**: every chunk is prefixed with one bracketed voice-quality direction (default `[soft, slow, calm voice]`, overridable in the UI and via `MOODSCAPE_ELEVENLABS_DIRECTION`). `direction.sanitize_direction()` rejects sound words (breath, sigh, laugh, yawn, hum, etc., including inflections and interjections such as "mm-hmm") and anything over 80 characters.
- **No vocal sounds**: the preprocessor strips every user-typed `[...]` tag from the spoken text. `[breath]`, `[inhale]`, `[exhale]` and `[pause:Xs]` become programmatic silence.
- **Stitching**: each request sends up to 3 `previous_request_ids` (`MOODSCAPE_ELEVENLABS_STITCH`, default on) for prosodic continuity across chunks. If a request carrying ids fails 400/422 (stale ids, e.g. from old cache hits), the chunk is retried once without ids and stitching stays off for the rest of that render.
- **Silence and joins**: pauses are inserted programmatically; chunk joins use 30 ms equal-power crossfades; each chunk is loudness-matched to -21 LUFS before joining.
- **48 kHz, clean chain**: audio is requested as `pcm_48000`, so there is no 24 to 48 kHz upsample and no DeepFilterNet or F5 mastering engine. Post-processing is a gentle split-band de-ess (`MOODSCAPE_ELEVENLABS_DEESS`), an HPF and optional room reverb.
- **Caches**:
  - Per-chunk PCM cache in `var/elevenlabs_cache` (`MOODSCAPE_ELEVENLABS_CACHE_DIR`). Request ids are excluded from the key, and speed is not part of it (v4 has no speed). It is never pruned: 48 kHz int16 PCM is ~5.5 MB per minute of speech for every voice/settings/direction combination. Delete the directory to clear it (the next render re-bills those chunks).
  - Whole-render cache via `core/tts_cache.py`. All `MOODSCAPE_ELEVENLABS_*` vars except `_CACHE_DIR` and `_MAX_RETRIES` enter the fingerprint, and only for ElevenLabs keys (`tts_env_fingerprint(engine="elevenlabs")`); they never affect F5/Kokoro/Chatterbox keys.
  - A whole-render cache hit reports `chars_billed: 0` in meta (the original count is kept as `chars_cached`).
- **Billing**: per character; the UI shows a cost line and `meta.json` records `chars_billed`.
- **Remix**: for voice + music renders only (not Vocals Only / Instrumental Only), after mixing, the pipeline saves a mix session (`var/mix_sessions/`, `core/mix_session.py`) so `core/remix.py` can re-run only mix, master and export without paying for TTS again. Sessions are large (~350 MB for 15 min of 48 kHz float32); the newest `MOODSCAPE_MIX_SESSIONS_KEEP` (default 5) are kept. Remix, Manual Generate and Auto Generate share one Gradio concurrency group (`"render"`) so they never run together, and Remix swaps only the mix-affecting env vars (`core/remix.py::MIX_ENV_PREFIXES`).

## Tuning knobs

| Var | Default | Purpose |
|---|---|---|
| `ELEVENLABS_API_KEY` (fallback `ELEVEN_LABS_API_KEY`) | - | Auth (not a cache-key var) |
| `MOODSCAPE_ELEVENLABS_MODEL` | `eleven_v4` | Override the model (experiments only) |
| `MOODSCAPE_ELEVENLABS_STABILITY` / `_SIMILARITY` | 0.5 / 0.75 | Defaults behind the UI sliders |
| `MOODSCAPE_ELEVENLABS_DIRECTION` | `[soft, slow, calm voice]` | Default delivery direction |
| `MOODSCAPE_ELEVENLABS_SEED` | 42 | Best-effort determinism |
| `MOODSCAPE_ELEVENLABS_CHUNK_MAX_CHARS` | 1500 | Target chunk size |
| `MOODSCAPE_ELEVENLABS_XFADE_MS` | 30 | Join crossfade |
| `MOODSCAPE_ELEVENLABS_CHUNK_LUFS` | -21 (`off` disables) | Per-chunk loudness match |
| `MOODSCAPE_ELEVENLABS_STITCH` | 1 | Send `previous_request_ids` (max 3) |
| `MOODSCAPE_ELEVENLABS_MAX_RETRIES` | 3 | Transient retry budget (network/timeout, 408, 429, 5xx; TypeError/AttributeError are never retried) |
| `MOODSCAPE_ELEVENLABS_DEESS` | 1 | Gentle split-band de-ess (applied after the cache, but still in the ElevenLabs cache key via its prefix; harmless) |
| `MOODSCAPE_ELEVENLABS_CACHE_DIR` | `var/elevenlabs_cache` | Per-chunk PCM cache |
| `MOODSCAPE_MIX_SESSIONS_KEEP` | 5 | Remix session retention (not a TTS var) |

## Known limits

- No speed or style control; the UI hides Speed for ElevenLabs. Pacing is direction + punctuation + programmatic silence.
- Direction tags are best-effort and voice-dependent.
- Older cloned voices may need retraining for v4.
- Billed per character; a whole-render cache hit costs nothing.
- Breaths, sighs and other vocal sounds are never rendered (user rule), in any engine.
