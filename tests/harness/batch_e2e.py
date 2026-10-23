"""Real end-to-end check of the Temporal batch pipeline (hits the live Deepgram, OpenRouter
and Cartesia APIs, and starts a local Temporal dev server - it costs a few API calls).

    uv run python -m tests.harness.batch_e2e

For each question it synthesises speech with Cartesia, runs it through the real worker
and VoiceChatbotWorkflow, and prints what the user would hear. Uses a throwaway SQLite
database; nothing else in your data is touched (the one test document is deleted again).
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import uuid
from pathlib import Path

from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from config.settings import settings
from core.assets import asset_ref
from rag.ingestor import delete_doc, ingest_file
from realtime.audio import pcm_to_wav
from realtime.tts_stream import stream_speech_chunks
from temporal.activities import ALL_ACTIVITIES
from temporal.dto import VoiceChatbotInput
from temporal.workflows import VoiceChatbotWorkflow

OWNER = "u_batche2e" + uuid.uuid4().hex[:12]
SECRET = "The reactor passphrase is TANGERINE-4417."


async def speak(text: str) -> str:
    """Synthesise `text` to a WAV in the assets folder and return its `assets/<name>` reference."""
    pcm = b"".join([chunk async for chunk in stream_speech_chunks(text)])
    path = settings.assets_dir / f"usr_{uuid.uuid4().hex[:8]}.wav"
    path.write_bytes(pcm_to_wav(pcm, settings.tts_sample_rate))
    return asset_ref(path)


async def ask(client, question: str, session_id: str):
    audio_path = await speak(question)
    result = await client.execute_workflow(
        VoiceChatbotWorkflow.run,
        VoiceChatbotInput(user_id=OWNER, session_id=session_id, input_audio_path=audio_path),
        id=f"e2e-{uuid.uuid4()}",
        task_queue=settings.temporal_task_queue,
    )
    print(f"\nQ (heard): {result.user_text!r}\nA        : {result.bot_text!r}\naudio    : {result.bot_audio_path}")
    return result


async def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    settings.database_path = Path(tempfile.mkdtemp()) / "e2e.db"

    secret_file = Path(tempfile.mkdtemp()) / "secret.txt"
    secret_file.write_text(SECRET, encoding="utf-8")
    await asyncio.to_thread(ingest_file, str(secret_file), OWNER)

    failures: list[str] = []
    try:
        async with await WorkflowEnvironment.start_local() as env:
            async with Worker(
                env.client,
                task_queue=settings.temporal_task_queue,
                workflows=[VoiceChatbotWorkflow],
                activities=ALL_ACTIVITIES,
            ):
                session = str(uuid.uuid4())
                math = await ask(env.client, "What is twelve times seven?", session)
                if "84" not in math.bot_text:
                    failures.append("calculation: expected 84")

                date = await ask(env.client, "What is the date today?", session)
                if not any(ch.isdigit() for ch in date.bot_text):
                    failures.append("date: expected a date from the tool")

                doc = await ask(env.client, "What is the reactor passphrase in my document?", session)
                if "TANGERINE" not in doc.bot_text.upper():
                    failures.append("document: expected TANGERINE-4417 from the owner's upload")

                if len(doc.history) < 2:
                    failures.append("history: earlier turns were not carried into the later ones")
    finally:
        await asyncio.to_thread(delete_doc, "secret.txt", OWNER)

    print("\nRESULT:", "PASS" if not failures else f"FAIL {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
