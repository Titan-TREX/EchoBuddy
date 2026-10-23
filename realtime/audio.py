"""
PCM / WAV helpers shared by the realtime session, the WebSocket handler and
the text-chat endpoint.

Everything here is 16-bit signed little-endian mono PCM. `audioop` was removed
from the standard library in Python 3.13; the `audioop-lts` backport (declared
in pyproject.toml for those versions) provides the same module, so this is the
only file that needs to know about it.
"""

from __future__ import annotations

import audioop
import io
import wave

_SAMPLE_WIDTH_BYTES = 2


def pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    """Wrap raw mono PCM16 bytes in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(_SAMPLE_WIDTH_BYTES)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def rms(frame: bytes) -> int:
    """Root-mean-square loudness of a PCM16 frame."""
    return audioop.rms(frame, _SAMPLE_WIDTH_BYTES)


class Resampler:
    """Stateful PCM16 mono resampler (keeps filter state across chunks so
    consecutive chunks join cleanly)."""

    def __init__(self, source_rate: int, target_rate: int) -> None:
        self.source_rate = source_rate
        self.target_rate = target_rate
        self._state = None

    def convert(self, pcm: bytes) -> bytes:
        if self.source_rate == self.target_rate:
            return pcm
        converted, self._state = audioop.ratecv(
            pcm, _SAMPLE_WIDTH_BYTES, 1, self.source_rate, self.target_rate, self._state
        )
        return converted
