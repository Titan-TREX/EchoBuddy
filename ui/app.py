"""
Presentation layer: Streamlit voice chat UI.

Records audio with the native `st.audio_input` widget, persists it to
`assets/` with a collision-proof UUID filename, starts a
`VoiceChatbotWorkflow` execution on Temporal, and renders the resulting
transcript plus an autoplaying reply.

Run with:

    streamlit run ui/app.py

A Temporal server and a running `temporal/worker.py` process must both be
reachable for this to work end-to-end.
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

import streamlit as st
from temporalio.client import Client

# Allow `streamlit run ui/app.py` to resolve the project's top-level packages
# (config, database, temporal) regardless of the current working directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

from config.settings import settings  # noqa: E402
from temporal.dto import VoiceChatbotInput, VoiceChatbotResult  # noqa: E402
from temporal.workflows import VoiceChatbotWorkflow  # noqa: E402


def _init_session_state() -> None:
    """Seed per-browser-session identifiers and chat history."""
    if "user_id" not in st.session_state:
        st.session_state.user_id = f"user_{uuid.uuid4().hex[:8]}"
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        # Each entry: {"role": "user"|"assistant", "text": str,
        #               "audio_path": str | None, "autoplay": bool}
        st.session_state.messages = []


def _save_recording(audio_value) -> str:
    """Persist a recorded clip to `assets/` with a UUID filename.

    Returns:
        The new file's path, relative to the project root, suitable for
        passing straight into `VoiceChatbotInput.input_audio_path`.
    """
    filename = f"usr_{uuid.uuid4().hex[:8]}.wav"
    output_path = settings.assets_dir / filename
    output_path.write_bytes(audio_value.getvalue())
    return output_path.relative_to(settings.base_dir).as_posix()


async def _run_workflow(audio_path: str) -> VoiceChatbotResult:
    """Start (and await) one `VoiceChatbotWorkflow` execution."""
    client = await Client.connect(
        settings.temporal_address,
        namespace=settings.temporal_namespace,
    )

    workflow_input = VoiceChatbotInput(
        user_id=st.session_state.user_id,
        session_id=st.session_state.session_id,
        input_audio_path=audio_path,
    )

    return await client.execute_workflow(
        VoiceChatbotWorkflow.run,
        workflow_input,
        id=f"voice-chat-{uuid.uuid4()}",
        task_queue=settings.temporal_task_queue,
    )


def _render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.write(message["text"])
        audio_path = message.get("audio_path")
        if audio_path:
            audio_full_path = settings.base_dir / audio_path
            if audio_full_path.exists():
                st.audio(str(audio_full_path), autoplay=message.get("autoplay", False))


def main() -> None:
    st.set_page_config(page_title="Voice Chatbot", page_icon="🎙️")
    st.title("🎙️ Voice-to-Voice AI Chatbot")
    st.caption("Temporal · Deepgram (STT) · Gemini (LLM) · Cartesia (TTS)")

    _init_session_state()

    for message in st.session_state.messages:
        _render_message(message)

    audio_value = st.audio_input("Press to record your message")

    if audio_value is not None:
        audio_path = _save_recording(audio_value)

        with st.spinner("Thinking..."):
            try:
                result = asyncio.run(_run_workflow(audio_path))
            except Exception as exc:  # noqa: BLE001 - surface any pipeline failure to the user
                st.error(f"Workflow failed: {exc}")
                return

        st.session_state.messages.append(
            {
                "role": "user",
                "text": result.user_text,
                "audio_path": audio_path,
                "autoplay": False,
            }
        )
        st.session_state.messages.append(
            {
                "role": "assistant",
                "text": result.bot_text,
                "audio_path": result.bot_audio_path,
                "autoplay": True,
            }
        )
        st.rerun()


if __name__ == "__main__":
    main()
