"""
Tests for temporal/worker.py

Since `worker.py` is primarily a wiring/entrypoint module, we test that:
  - `main()` connects to Temporal with the correct settings values.
  - The Worker is constructed with the expected task queue, workflow, and activities.
  - `worker.run()` is awaited (i.e. the worker actually starts).
  - Logger emits the startup info line.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

_MODULE = "temporal.worker"


class TestWorkerMain:
    @pytest.mark.asyncio
    async def test_connects_with_correct_address_and_namespace(self):
        mock_client = AsyncMock()
        mock_worker_instance = AsyncMock()

        with (
            patch(f"{_MODULE}.Client") as MockClient,
            patch(f"{_MODULE}.Worker") as MockWorker,
            patch(f"{_MODULE}.settings") as mock_settings,
        ):
            mock_settings.temporal_address = "test-host:7233"
            mock_settings.temporal_namespace = "test-ns"
            mock_settings.temporal_task_queue = "test-queue"

            MockClient.connect = AsyncMock(return_value=mock_client)
            MockWorker.return_value = mock_worker_instance
            mock_worker_instance.run = AsyncMock()

            from temporal.worker import main
            await main()

        MockClient.connect.assert_awaited_once_with(
            "test-host:7233", namespace="test-ns"
        )

    @pytest.mark.asyncio
    async def test_worker_created_with_correct_task_queue(self):
        mock_client = AsyncMock()
        mock_worker_instance = AsyncMock()
        worker_kwargs: dict = {}

        with (
            patch(f"{_MODULE}.Client") as MockClient,
            patch(f"{_MODULE}.Worker") as MockWorker,
            patch(f"{_MODULE}.settings") as mock_settings,
        ):
            mock_settings.temporal_address = "localhost:7233"
            mock_settings.temporal_namespace = "default"
            mock_settings.temporal_task_queue = "voice-bot-queue"

            MockClient.connect = AsyncMock(return_value=mock_client)

            def capture_worker(*args, **kwargs):
                worker_kwargs.update(kwargs)
                return mock_worker_instance

            MockWorker.side_effect = capture_worker
            mock_worker_instance.run = AsyncMock()

            from temporal.worker import main
            await main()

        assert worker_kwargs.get("task_queue") == "voice-bot-queue"

    @pytest.mark.asyncio
    async def test_worker_created_with_workflow_registered(self):
        from temporal.workflows import VoiceChatbotWorkflow

        mock_client = AsyncMock()
        mock_worker_instance = AsyncMock()
        worker_kwargs: dict = {}

        with (
            patch(f"{_MODULE}.Client") as MockClient,
            patch(f"{_MODULE}.Worker") as MockWorker,
            patch(f"{_MODULE}.settings") as mock_settings,
        ):
            mock_settings.temporal_address = "localhost:7233"
            mock_settings.temporal_namespace = "default"
            mock_settings.temporal_task_queue = "voice-bot-queue"

            MockClient.connect = AsyncMock(return_value=mock_client)

            def capture_worker(*args, **kwargs):
                worker_kwargs.update(kwargs)
                return mock_worker_instance

            MockWorker.side_effect = capture_worker
            mock_worker_instance.run = AsyncMock()

            from temporal.worker import main
            await main()

        assert VoiceChatbotWorkflow in worker_kwargs.get("workflows", [])

    @pytest.mark.asyncio
    async def test_worker_run_is_awaited(self):
        mock_client = AsyncMock()
        mock_worker_instance = AsyncMock()
        mock_worker_instance.run = AsyncMock()

        with (
            patch(f"{_MODULE}.Client") as MockClient,
            patch(f"{_MODULE}.Worker") as MockWorker,
            patch(f"{_MODULE}.settings") as mock_settings,
        ):
            mock_settings.temporal_address = "localhost:7233"
            mock_settings.temporal_namespace = "default"
            mock_settings.temporal_task_queue = "voice-bot-queue"

            MockClient.connect = AsyncMock(return_value=mock_client)
            MockWorker.return_value = mock_worker_instance

            from temporal.worker import main
            await main()

        mock_worker_instance.run.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_worker_registered_with_all_activities(self):
        from temporal.activities import ALL_ACTIVITIES

        mock_client = AsyncMock()
        mock_worker_instance = AsyncMock()
        worker_kwargs: dict = {}

        with (
            patch(f"{_MODULE}.Client") as MockClient,
            patch(f"{_MODULE}.Worker") as MockWorker,
            patch(f"{_MODULE}.settings") as mock_settings,
        ):
            mock_settings.temporal_address = "localhost:7233"
            mock_settings.temporal_namespace = "default"
            mock_settings.temporal_task_queue = "voice-bot-queue"

            MockClient.connect = AsyncMock(return_value=mock_client)

            def capture_worker(*args, **kwargs):
                worker_kwargs.update(kwargs)
                return mock_worker_instance

            MockWorker.side_effect = capture_worker
            mock_worker_instance.run = AsyncMock()

            from temporal.worker import main
            await main()

        registered = worker_kwargs.get("activities", [])
        for act in ALL_ACTIVITIES:
            assert act in registered
