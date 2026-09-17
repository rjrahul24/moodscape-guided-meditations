# Article 1 — The TTS Gauntlet: Choosing an AI Voice for Meditation

**Thesis (one line):** I auditioned six speech engines on a single 32 GB Apple Silicon
machine and shipped two — and the *selection logic* matters more than any one engine.

**Target reader takeaway:** How to evaluate neural TTS engines for a specific,
demanding use-case, and how to design your pipeline so swapping/culling engines is
cheap rather than a rewrite.

**Suggested length:** 1,800–2,400 words.

---

## Narrative beats (in order)

1. **Hook — the constraint frames everything.** One Mac, 32 GB unified RAM, target
   quality "Calm/Headspace". You cannot hold two large models in memory at once, so
   the pipeline *loads engines sequentially* and every engine is a swappable module.
   That constraint is the reason this became a "gauntlet" and not a "pick one".

2. **The contract that made the gauntlet possible.** Before comparing engines, show the
   ABC every engine implements. This is the load-bearing design decision of the whole
   project: the mixer and FX never know which engine produced the audio.

3. **The six contenders and why each was on the shortlist.**
   - **Kokoro** — tiny (82M), fast, CPU-only on Apple Silicon (MPS caused
     deallocation bus errors), great for *blendable* preset voices.
   - **F5-TTS** — zero-shot voice cloning from a reference clip; expressive.
   - **Chatterbox** — cut early.
   - **HeartMuLa** — cut early (a 1,178-line engine — a big bet that didn't pay off).
   - **IndexTTS-2** — added mid-project for zero-shot cloning + explicit emotion
     control; later cut.
   - *(music side, for contrast)* ACE-Step vs Lyria — foreshadow Article 3's pivot.

4. **The three culls, and the honest reasons.**
   - Cull #1 (`8e8e5b3`): Chatterbox + HeartMuLa out, IndexTTS-2 in — net −1,211 lines.
   - Cull #2 (`7aca804`): IndexTTS-2 out.
   - Cull #3 (`f24dcdd`): ACE-Step out.
   - The lesson: *an engine that needs constant special-casing to sound good is a
     liability even when it occasionally sounds best.* Kokoro + F5 covered the range
     (blendable presets + clonable references) with the least maintenance.

5. **The subtractive-blending trick (Kokoro).** The most shareable idea in the article:
   you can blend voice embeddings with **negative weights** to *remove* a quality
   (tension/energy) rather than only mixing in more voices. Then renormalize so the
   subtraction doesn't drift the amplitude.

6. **Close — a roster is a product decision.** Shipping two engines wasn't a
   compromise; it was the result. End on the through-line: *subtraction beat addition.*

---

## Code snippets to include

### 1. The engine-agnostic contract (`core/speech_engine.py:10`)

```python
class SpeechEngine(ABC):
    """Interface that all TTS engines must implement.

    Every engine produces mono float32 audio at 24 000 Hz together with a
    boolean voice-activity mask of the same length.  The rest of the pipeline
    (Pedalboard FX, mixer) is engine-agnostic — it only depends on
    this contract.
    """

    @abstractmethod
    def synthesize(
        self, segments: list[dict], voice: str, speed: float, progress_cb=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (voice_audio float32@24k, voice_activity bool mask)."""
```

> Talking point: the boolean **voice-activity mask** returned here is what the mixer's
> breathing duck later keys off of (Article 3). The voice engine and the ducker are
> decoupled by this one array.

### 2. Subtractive voice blending (`core/kokoro_tts/voice_manager.py:38` and `:140`)

The `pure_calm` preset — note the negative weight:

```python
"pure_calm": {
    "description": "Ultra-low tension — warmth with conversational energy subtracted",
    "blend": {
        "af_heart":  0.60,   # primary warmth and stability
        "af_sarah":  0.30,   # soft, natural breathiness
        "af_aoede":  0.10,   # musical prosody quality
        "af_bella": -0.05,   # subtract 5% of energetic/tension traits
    },
    "method": "extrapolation",
},
```

```python
def blend_with_extrapolation(voice_weights: dict[str, float]) -> torch.Tensor:
    """Negative weights subtract a voice's characteristics from the blend.
    After blending, renormalize to the primary voice's L2 norm to prevent
    the amplitude drift that subtraction introduces."""
    result, primary_norm = None, None
    for voice_id, weight in voice_weights.items():
        tensor = load_voice_tensor(voice_id).float()
        if result is None:
            result = tensor * weight
            primary_norm = float(torch.norm(tensor.flatten()))   # capture primary norm
        else:
            result = result + tensor * weight
    # Renormalize — prevents amplitude drift from the subtraction
    if primary_norm is not None and primary_norm > 1e-6:
        current_norm = float(torch.norm(result.flatten()))
        if current_norm > 1e-6:
            result = result * (primary_norm / current_norm)
    return result
```

### 3. (Optional) stress markers as prosody control (`core/kokoro_tts/preprocessor.py`)

Mention that tension words are wrapped with misaki `(-1)` and affirmations `(+1)`,
applied *after* IPA injection so a collision guard never double-wraps Sanskrit IPA
blocks (commit `c85832b`). One or two lines of the wrapped output is enough.

---

## Learnings box

> - **Design for the cull.** A stable engine contract (mono float32 @ 24 kHz + VAD
>   mask) turned "remove an engine" into a delete, not a refactor. Four engines came
>   out over the project with near-zero blast radius.
> - **RAM ceiling → sequential loading → modular engines.** The hardware constraint
>   drove the architecture, not the other way around.
> - **Blend embeddings subtractively.** Negative weights let you *remove* a vocal
>   quality; always renormalize to the primary voice's norm afterward.
> - **Kokoro is CPU-only on Apple Silicon** — MPS caused deallocation bus errors
>   (`docs/GOTCHAS.md`). British voices need a separate `KPipeline(lang_code="b")`.
> - **"Sometimes best" loses to "reliably good with less babysitting."** That is why
>   IndexTTS-2, despite emotion control, didn't survive.

## Source appendix
Commits: `c85832b`, `ab041ef`, `e06e70b`, `19abbf3`, `8e8e5b3`, `7aca804`, `f24dcdd`.
Code: `core/speech_engine.py`, `core/kokoro_tts/voice_manager.py`, `core/kokoro_tts/preprocessor.py`.
Docs: `docs/COMPONENT_REGISTRY.md`, `docs/GOTCHAS.md`, `CLAUDE.md` (engine roster).
