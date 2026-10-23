"""Tests for realtime/audio.py."""

from __future__ import annotations

import io
import struct
import wave

from realtime.audio import Resampler, pcm_to_wav, rms


def _pcm(samples: list[int]) -> bytes:
    return struct.pack(f"<{len(samples)}h", *samples)


class TestPcmToWav:
    def test_produces_valid_mono_16bit_wav(self):
        pcm = _pcm([0, 1000, -1000, 500] * 100)
        with wave.open(io.BytesIO(pcm_to_wav(pcm, 16000)), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000
            assert wf.readframes(wf.getnframes()) == pcm

    def test_empty_audio_is_still_a_valid_wav(self):
        with wave.open(io.BytesIO(pcm_to_wav(b"", 16000)), "rb") as wf:
            assert wf.getnframes() == 0


class TestRms:
    def test_silence_is_zero(self):
        assert rms(_pcm([0] * 160)) == 0

    def test_constant_signal(self):
        assert rms(_pcm([300] * 160)) == 300


class TestResampler:
    def test_same_rate_is_passthrough(self):
        pcm = _pcm(list(range(100)))
        assert Resampler(16000, 16000).convert(pcm) == pcm

    def test_downsampling_48k_to_16k_shrinks_by_three(self):
        resampler = Resampler(48000, 16000)
        out = b"".join(resampler.convert(_pcm([100] * 480)) for _ in range(10))
        assert abs(len(out) - 4800 * 2 // 3) <= 8

    def test_state_is_kept_across_chunks(self):
        resampler = Resampler(48000, 16000)
        resampler.convert(_pcm([1000] * 480))
        assert resampler._state is not None
