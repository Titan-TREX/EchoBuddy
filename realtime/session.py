"""
Per-WebSocket-connection orchestration.

This replaces the role `VoiceChatbotWorkflow` played in the batch pipeline,
but runs as a plain asyncio task instead of a Temporal workflow - a
real-time turn-taking loop doesn't fit Temporal's execution model well
(workflow/activity scheduling overhead is fine for a "record then process"
pipeline, but adds latency that defeats the purpose here).

State machine, conceptually:
    LISTENING  -> speech detected, buffer frames
               -> silence timeout -> finalize utterance -> BUSY
    BUSY       -> STT -> streaming LLM -> streaming TTS -> back to LISTENING

The mic is fully ignored (`bot_speaking` stays True) for the entire BUSY
window - from the moment an utterance is finalized, through STT and LLM
generation, through TTS playback, until the bot's reply has fully finished
playing. There is no interrupt/barge-in: the user can only speak again once
`bot_speaking` goes back to False.
"""

from __future__ import annotations

import asyncio
import audioop
import io
import logging
import uuid
import wave
from typing import Awaitable, Callable

import webrtcvad

from config.settings import settings
from realtime.db import ensure_user_and_session, fetch_recent_history, log_message
from realtime.llm_stream import stream_reply_chunks
from realtime.stt import transcribe
from realtime.tts_stream import stream_speech_chunks, TTS_SAMPLE_RATE

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_BYTES = int(SAMPLE_RATE * FRAME_MS / 1000) * 2
SILENCE_TIMEOUT_MS = 500
PRE_SPEECH_PADDING_MS = 300
MIN_UTTERANCE_MS = 250
VAD_AGGRESSIVENESS = 2
MIN_SPEECH_RMS = 150          # tune this to your mic/room - see calibration note below


class VoiceSession:
    def __init__(
        self,
        send_json: Callable[[dict], Awaitable[None]],
        send_bytes: Callable[[bytes], Awaitable[None]],
        user_id: str,
        session_id: str,
    ) -> None:
        self.send_json = send_json
        self.send_bytes = send_bytes
        self.user_id = user_id
        self.session_id = session_id

        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.pcm_buffer = bytearray()
        self.ring_buffer: list[bytes] = []
        self.utterance_frames: list[bytes] = []
        self.is_speaking = False
        self.silence_ms = 0

        self.bot_speaking = False
        self.gen_task: asyncio.Task | None = None

    async def start(self) -> None:
        await ensure_user_and_session(self.user_id, self.session_id, self.user_id)

    def _trim_ring(self) -> None:
        max_frames = PRE_SPEECH_PADDING_MS // FRAME_MS
        while len(self.ring_buffer) > max_frames:
            self.ring_buffer.pop(0)

    async def feed_pcm16(self, pcm: bytes) -> None:
        """pcm: 16kHz mono signed 16-bit little-endian bytes, any length."""
        self.pcm_buffer.extend(pcm)
        while len(self.pcm_buffer) >= FRAME_BYTES:
            frame = bytes(self.pcm_buffer[:FRAME_BYTES])
            del self.pcm_buffer[:FRAME_BYTES]
            await self._process_frame(frame)

    async def _process_frame(self, frame: bytes) -> None:
        if self.bot_speaking:
            return  # bot is thinking/speaking - mic fully ignored, no interrupts

        is_speech = (
            self.vad.is_speech(frame, SAMPLE_RATE)
            and audioop.rms(frame, 2) >= MIN_SPEECH_RMS
        )

        if is_speech:
            if not self.is_speaking:
                self.utterance_frames = list(self.ring_buffer)
                self.is_speaking = True
            self.utterance_frames.append(frame)
            self.silence_ms = 0
        else:
            self.ring_buffer.append(frame)
            self._trim_ring()
            if self.is_speaking:
                self.utterance_frames.append(frame)
                self.silence_ms += FRAME_MS
                if self.silence_ms >= SILENCE_TIMEOUT_MS:
                    await self._finalize_utterance()

    async def _finalize_utterance(self) -> None:
        frames = self.utterance_frames
        duration_ms = len(frames) * FRAME_MS

        self.is_speaking = False
        self.utterance_frames = []
        self.silence_ms = 0

        if duration_ms < MIN_UTTERANCE_MS:
            return

        self.bot_speaking = True  # mute mic now - we're processing this turn
        wav_bytes = self._frames_to_wav(frames)
        filename = f"usr_{uuid.uuid4().hex[:8]}.wav"
        wav_path = settings.assets_dir / filename
        wav_path.write_bytes(wav_bytes)
        relative_path = wav_path.relative_to(settings.base_dir).as_posix()

        self.gen_task = asyncio.create_task(self._handle_turn(wav_bytes, relative_path))

    @staticmethod
    def _frames_to_wav(frames: list[bytes]) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b"".join(frames))
        return buf.getvalue()

    async def _handle_turn(self, wav_bytes: bytes, audio_path: str) -> None:
        try:
            await self.send_json({"type": "listening_stopped"})

            user_text = await transcribe(wav_bytes)
            if not user_text:
                self.bot_speaking = False  # nothing heard - go back to listening
                await self.send_json({"type": "no_speech"})
                return

            await self.send_json({"type": "user_transcript", "text": user_text})
            await log_message(self.session_id, "user", user_text, audio_path)

            history = await fetch_recent_history(
                self.session_id, settings.gemini_max_history_turns
            )

            self.bot_speaking = True
            await self.send_json({"type": "bot_speaking_start"})

            full_text = ""
            audio_seconds_sent = 0.0
            stream_started_at = asyncio.get_event_loop().time()
            async for sentence, accumulated in stream_reply_chunks(user_text, history):
                full_text = accumulated
                await self.send_json({"type": "bot_text_chunk", "text": sentence})
                async for audio_chunk in stream_speech_chunks(sentence):
                    await self.send_bytes(audio_chunk)
                    audio_seconds_sent += len(audio_chunk) / 2 / TTS_SAMPLE_RATE

            # Chunks reach the browser faster than they play - wait out the
            # remaining estimated playback time before unmuting the mic.
            elapsed = asyncio.get_event_loop().time() - stream_started_at
            remaining = audio_seconds_sent - elapsed
            if remaining > 0:
                await asyncio.sleep(remaining)

            self.bot_speaking = False
            await self.send_json({"type": "bot_speaking_end"})
            await log_message(self.session_id, "assistant", full_text, None)

        except asyncio.CancelledError:
            logger.info("Turn cancelled: session=%s", self.session_id)
        except Exception:
            logger.exception("Turn handling failed: session=%s", self.session_id)
            self.bot_speaking = False
            await self.send_json({"type": "error", "message": "Something went wrong."})