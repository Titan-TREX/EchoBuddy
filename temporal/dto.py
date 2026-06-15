"""
Shared Data Transfer Objects (DTOs).

Every value that crosses a Workflow <-> Activity boundary must be a plain,
JSON-serializable shape (dataclasses, primitives, and lists/dicts thereof)
per the Temporal Python SDK's default data converter. Centralizing those
shapes here keeps `workflows.py` and every activity module in sync without
import cycles, and gives the pipeline a single, scannable "schema".
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ConversationTurn:
    """A single past message, used to seed the LLM's context window."""

    role: str  # "user" | "assistant"
    content: str


@dataclass(frozen=True)
class TranscriptionResult:
    """Output of the Deepgram speech-to-text activity."""

    text: str
    confidence: float


@dataclass(frozen=True)
class LLMResult:
    """Output of the Gemini "brain" activity."""

    text: str


@dataclass(frozen=True)
class SpeechResult:
    """Output of the Cartesia text-to-speech activity."""

    audio_path: str  # relative path under assets/, e.g. "assets/bot_f2e4c1.mp3"


@dataclass(frozen=True)
class MessageRecord:
    """A single chat message to be persisted to SQLite."""

    session_id: str
    role: str  # "user" | "assistant"
    content: str
    audio_path: str | None = None


@dataclass(frozen=True)
class VoiceChatbotInput:
    """Input to `VoiceChatbotWorkflow.run`."""

    user_id: str
    session_id: str
    input_audio_path: str  # relative path under assets/, e.g. "assets/usr_a7b9e3.wav"


@dataclass(frozen=True)
class VoiceChatbotResult:
    """Final result returned from the workflow to the Streamlit UI."""

    user_text: str
    bot_text: str
    bot_audio_path: str
    history: list[ConversationTurn] = field(default_factory=list)
