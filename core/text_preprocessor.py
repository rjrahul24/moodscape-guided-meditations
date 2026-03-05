"""Text preprocessing to optimise scripts for Kokoro TTS meditation delivery."""

import re


def preprocess_for_meditation(text: str) -> str:
    """Optimise text for calm, meditation-style Kokoro TTS delivery.

    Rules derived from Kokoro TTS optimisation research:
    - Commas = micro-pauses
    - Ellipsis (...) = reflective breath (longer internal pause)
    - Periods = full stop + intonation reset
    - Em dashes = slight rhythmic break
    - Semicolons = brief pause with anticipation

    Applied to each speech segment's text *before* it is sent to the TTS
    engine.  Does NOT alter [pause:Xs] markers — those are handled by the
    script parser.
    """
    # Ensure ellipsis is properly formatted (not separate dots)
    text = re.sub(r'\.{2,}', '...', text)

    # Add breathing room: ensure there is always trailing punctuation
    # before [pause:Xs] markers for natural trailing prosody
    text = re.sub(r'(\w)\s*(\[pause:)', r'\1... \2', text)

    # Normalise excessive whitespace
    text = re.sub(r'  +', ' ', text)

    return text.strip()


def validate_chunk_length(text: str, max_tokens: int = 200) -> list[str]:
    """Split text into chunks that do not exceed Kokoro's sweet spot.

    Kokoro performs best with 100-200 tokens per chunk.
    Above ~400-510 tokens it starts rushing.  This function enforces a hard
    ceiling of 400 tokens per chunk.
    """
    HARD_CEILING = 400

    words = text.split()
    estimated_tokens = int(len(words) * 1.3)

    if estimated_tokens <= max_tokens:
        return [text]

    # Split at sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = int(len(sentence.split()) * 1.3)

        # If a single sentence exceeds the hard ceiling, force-split by word count
        if sentence_tokens > HARD_CEILING:
            # Flush current chunk first
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_tokens = 0

            s_words = sentence.split()
            max_words = int(HARD_CEILING / 1.3)
            for i in range(0, len(s_words), max_words):
                chunks.append(' '.join(s_words[i:i + max_words]))
            continue

        if current_tokens + sentence_tokens > max_tokens and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_tokens = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks
