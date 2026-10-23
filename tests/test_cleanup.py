"""Tests for realtime/cleanup.py. Everything runs in a temp directory, never the real assets folder."""

from __future__ import annotations

import asyncio
import os
import time

import pytest

from config.settings import settings
from realtime import cleanup
from realtime.cleanup import cleanup_loop, purge_old_files

HOUR = 3600


def _make(directory, name: str, age_hours: float = 0):
    path = directory / name
    path.write_bytes(b"audio")
    mtime = time.time() - age_hours * HOUR
    os.utime(path, (mtime, mtime))
    return path


class TestPurgeOldFiles:
    def test_removes_old_generated_audio_and_keeps_recent(self, tmp_path):
        old_user = _make(tmp_path, "usr_aaaa1111.wav", age_hours=30)
        old_bot = _make(tmp_path, "bot_bbbb2222.mp3", age_hours=48)
        old_webm = _make(tmp_path, "usr_cccc3333.webm", age_hours=25)
        fresh = _make(tmp_path, "bot_dddd4444.wav", age_hours=1)

        assert purge_old_files(tmp_path, 24 * HOUR) == 3

        assert not old_user.exists() and not old_bot.exists() and not old_webm.exists()
        assert fresh.exists()

    def test_never_touches_files_the_app_did_not_generate(self, tmp_path):
        keep = [
            _make(tmp_path, ".gitkeep", age_hours=999),
            _make(tmp_path, "notes.txt", age_hours=999),
            _make(tmp_path, "my_recording.wav", age_hours=999),   # right extension, wrong prefix
            _make(tmp_path, "usr_secret.txt", age_hours=999),     # right prefix, not audio
            _make(tmp_path, "usr_abcd1234", age_hours=999),       # no extension
        ]
        assert purge_old_files(tmp_path, HOUR) == 0
        assert all(path.exists() for path in keep)

    def test_does_not_descend_into_or_remove_subfolders(self, tmp_path):
        sub = tmp_path / "usr_folder.wav"          # a directory that merely looks like audio
        sub.mkdir()
        nested = _make(sub, "bot_old.wav", age_hours=999)
        assert purge_old_files(tmp_path, HOUR) == 0
        assert sub.is_dir() and nested.exists()

    def test_extension_match_is_case_insensitive(self, tmp_path):
        loud = _make(tmp_path, "bot_aaaa1111.WAV", age_hours=30)
        assert purge_old_files(tmp_path, 24 * HOUR) == 1
        assert not loud.exists()

    def test_boundary_uses_the_age_limit(self, tmp_path):
        just_under = _make(tmp_path, "bot_aaaa1111.wav", age_hours=23.9)
        just_over = _make(tmp_path, "bot_bbbb2222.wav", age_hours=24.1)
        purge_old_files(tmp_path, 24 * HOUR)
        assert just_under.exists() and not just_over.exists()

    def test_empty_directory(self, tmp_path):
        assert purge_old_files(tmp_path, HOUR) == 0

    def test_file_that_cannot_be_removed_does_not_stop_the_sweep(self, tmp_path, monkeypatch):
        stuck = _make(tmp_path, "usr_stuck111.wav", age_hours=30)
        other = _make(tmp_path, "usr_other222.wav", age_hours=30)
        real_unlink = type(stuck).unlink

        def flaky_unlink(self, *args, **kwargs):
            if self.name == "usr_stuck111.wav":
                raise PermissionError("in use")
            return real_unlink(self, *args, **kwargs)

        monkeypatch.setattr(type(stuck), "unlink", flaky_unlink)

        assert purge_old_files(tmp_path, 24 * HOUR) == 1
        assert stuck.exists() and not other.exists()


class TestCleanupLoop:
    @pytest.mark.asyncio
    async def test_disabled_when_retention_is_zero(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "assets_dir", tmp_path)
        monkeypatch.setattr(settings, "assets_retention_hours", 0)
        old = _make(tmp_path, "usr_aaaa1111.wav", age_hours=999)

        await asyncio.wait_for(cleanup_loop(), timeout=2)  # returns immediately, doesn't loop

        assert old.exists()

    @pytest.mark.asyncio
    async def test_sweeps_repeatedly_and_survives_a_failed_sweep(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "assets_dir", tmp_path)
        monkeypatch.setattr(settings, "assets_retention_hours", 1)
        monkeypatch.setattr(settings, "assets_cleanup_interval_seconds", 0.01)
        calls = {"n": 0}
        real_purge = cleanup.purge_old_files

        def flaky(directory, max_age, now=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("disk hiccup")
            return real_purge(directory, max_age, now)

        monkeypatch.setattr(cleanup, "purge_old_files", flaky)
        old = _make(tmp_path, "bot_aaaa1111.wav", age_hours=5)

        task = asyncio.create_task(cleanup_loop())
        for _ in range(100):
            if not old.exists():
                break
            await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert calls["n"] >= 2          # kept going after the first sweep blew up
        assert not old.exists()
