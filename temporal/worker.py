"""
Temporal Worker entrypoint.

Boots a worker process that polls the configured task queue
(`voice-bot-queue` by default), hosting `VoiceChatbotWorkflow` and every
activity implementation in `temporal/activities/`.

Run this in its own terminal/process, alongside a running Temporal server:

    python -m temporal.worker
"""

from __future__ import annotations

import asyncio
import logging

from temporalio.client import Client
from temporalio.worker import Worker

from config.settings import settings
from temporal.activities import ALL_ACTIVITIES
from temporal.workflows import VoiceChatbotWorkflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main() -> None:
    client = await Client.connect(
        settings.temporal_address,
        namespace=settings.temporal_namespace,
    )

    worker = Worker(
        client,
        task_queue=settings.temporal_task_queue,
        workflows=[VoiceChatbotWorkflow],
        activities=ALL_ACTIVITIES,
    )

    logger.info(
        "Starting Temporal worker | task_queue=%s address=%s namespace=%s",
        settings.temporal_task_queue,
        settings.temporal_address,
        settings.temporal_namespace,
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
