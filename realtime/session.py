"""
Per-WebSocket-connection orchestration for the realtime voice pipeline.

This replaces the role `VoiceChatbotWorkflow` plays in the batch pipeline, but
runs as a plain asyncio task instead of a Temporal workflow - a real-time
turn-taking loop doesn't fit Temporal's execution model well (activity
scheduling overhead is fine for "record then process", but adds latency that
defeats the purpose here).

State machine, conceptually:
    LISTENING  -> speech detected, buffer frames
               -> silence timeout -> finalize utterance -> BUSY
    BUSY       -> STT -> streaming LLM -> streaming TTS -> back to LISTENING

The mic is fully ignored (`bot_speaking` stays True) for the entire BUSY
window - from the moment an utterance is finalized, through STT and LLM
generation, through TTS playback, until the bot's reply has fully finished
playing. There is no barge-in: the user can only speak again once
`bot_speaking` goes back to False.

All VAD / audio parameters live in `config/settings.py` (`vad_*`, `audio_*`).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import deque
from collections.abc import Awaitable, Callable

import webrtcvad

from config.settings import settings
from config.tracing import observe, propagate_attributes
from core.assets import asset_ref
from database.repository import ensure_user_and_session, fetch_recent_history, log_message
from realtime.audio import pcm_to_wav, rms
from realtime.llm_stream import stream_reply_chunks
from realtime.stt import transcribe
from realtime.tts_stream import stream_speech_chunks

logger = logging.getLogger(__name__)

_BYTES_PER_SAMPLE = 2


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

        self.frame_bytes = settings.vad_frame_bytes
        self.vad = webrtcvad.Vad(settings.vad_aggressiveness)
        self.pcm_buffer = bytearray()
        # Rolling window of recent non-speech frames, prepended to an utterance
        # when speech starts so the first syllable isn't clipped.
        self.ring_buffer: deque[bytes] = deque(
            maxlen=max(1, settings.vad_pre_speech_padding_ms // settings.vad_frame_ms)
        )
        self.utterance_frames: list[bytes] = []
        self.is_speaking = False
        self.silence_ms = 0

        self.bot_speaking = False
        self.gen_task: asyncio.Task | None = None

    async def start(self) -> None:
        await ensure_user_and_session(self.user_id, self.session_id, self.user_id)

    async def feed_pcm16(self, pcm: bytes) -> None:
        """pcm: mono signed 16-bit little-endian bytes at `settings.audio_sample_rate`."""
        self.pcm_buffer.extend(pcm)
        while len(self.pcm_buffer) >= self.frame_bytes:
            frame = bytes(self.pcm_buffer[: self.frame_bytes])
            del self.pcm_buffer[: self.frame_bytes]
            await self._process_frame(frame)

    async def _process_frame(self, frame: bytes) -> None:
        if self.bot_speaking:
            return  # bot is thinking/speaking - mic fully ignored, no interrupts

        is_speech = (
            self.vad.is_speech(frame, settings.audio_sample_rate)
            and rms(frame) >= settings.vad_min_speech_rms
        )

        if is_speech:
            if not self.is_speaking:
                self.utterance_frames = list(self.ring_buffer)
                self.is_speaking = True
            self.utterance_frames.append(frame)
            self.silence_ms = 0
            return

        self.ring_buffer.append(frame)
        if self.is_speaking:
            self.utterance_frames.append(frame)
            self.silence_ms += settings.vad_frame_ms
            if self.silence_ms >= settings.vad_silence_timeout_ms:
                await self._finalize_utterance()

    async def _finalize_utterance(self) -> None:
        frames = self.utterance_frames
        duration_ms = len(frames) * settings.vad_frame_ms

        self.is_speaking = False
        self.utterance_frames = []
        self.silence_ms = 0

        if duration_ms < settings.vad_min_utterance_ms:
            return

        self.bot_speaking = True  # mute mic now - we're processing this turn
        wav_bytes = pcm_to_wav(b"".join(frames), settings.audio_sample_rate)
        wav_path = settings.assets_dir / f"usr_{uuid.uuid4().hex[:8]}.wav"
        await asyncio.to_thread(wav_path.write_bytes, wav_bytes)
        relative_path = asset_ref(wav_path)

        self.gen_task = asyncio.create_task(self._handle_turn(wav_bytes, relative_path))

    @observe(name="voice-turn", capture_input=False, capture_output=False)
    async def _handle_turn(self, wav_bytes: bytes, audio_path: str) -> None:
        # One Langfuse trace per turn; STT, RAG, tools, LLM and TTS nest beneath it.
        with propagate_attributes(user_id=self.user_id, session_id=self.session_id):
            await self._run_turn(wav_bytes, audio_path)

    async def _run_turn(self, wav_bytes: bytes, audio_path: str) -> None:
        try:
            await self.send_json({"type": "listening_stopped"})

            user_text = await transcribe(wav_bytes)
            if not user_text:
                self.bot_speaking = False  # nothing heard - go back to listening
                await self.send_json({"type": "no_speech"})
                return

            await self.send_json({"type": "user_transcript", "text": user_text})

            # Fetch history BEFORE logging this utterance: the LLM layer appends
            # the current user message itself, so logging first would send it twice.
            history = await fetch_recent_history(self.session_id, settings.history_max_turns)
            await log_message(self.session_id, "user", user_text, audio_path)

            await self.send_json({"type": "bot_speaking_start"})

            full_text = ""
            audio_seconds_sent = 0.0
            loop = asyncio.get_running_loop()
            stream_started_at = loop.time()
            async for sentence, accumulated in stream_reply_chunks(user_text, history, self.user_id):
                full_text = accumulated
                await self.send_json({"type": "bot_text_chunk", "text": sentence})
                async for audio_chunk in stream_speech_chunks(sentence):
                    await self.send_bytes(audio_chunk)
                    audio_seconds_sent += len(audio_chunk) / _BYTES_PER_SAMPLE / settings.tts_sample_rate

            # Chunks reach the browser faster than they play - wait out the
            # remaining estimated playback time before unmuting the mic.
            remaining = audio_seconds_sent - (loop.time() - stream_started_at)
            if remaining > 0:
                await asyncio.sleep(remaining)

            self.bot_speaking = False
            await self.send_json({"type": "bot_speaking_end"})
            await log_message(self.session_id, "assistant", full_text, None)

        except asyncio.CancelledError:
            logger.info("Turn cancelled: session=%s", self.session_id)
            raise
        except Exception:
            logger.exception("Turn handling failed: session=%s", self.session_id)
            self.bot_speaking = False
            await self.send_json({"type": "error", "message": "Something went wrong."})
