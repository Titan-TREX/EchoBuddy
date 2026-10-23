"""Tests for core/assets.py and its use by the speech-to-text activity."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.assets import asset_path, asset_ref


class TestAssetRef:
    def test_reference_is_assets_slash_name(self, tmp_path):
        assert asset_ref(tmp_path / "bot_1234abcd.mp3") == "assets/bot_1234abcd.mp3"

    def test_works_for_a_folder_outside_the_project(self):
        """Regression: refs used to be computed relative to the project folder, which crashed
        once the assets folder moved to a mounted DATA_DIR such as /data/assets."""
        assert asset_ref(Path("/data/assets/usr_aaaa1111.wav")) == "assets/usr_aaaa1111.wav"


class TestAssetPath:
    def test_resolves_inside_the_assets_folder(self, tmp_path):
        assert asset_path(tmp_path, "assets/usr_1.wav") == tmp_path / "usr_1.wav"

    def test_bare_name_is_accepted(self, tmp_path):
        assert asset_path(tmp_path, "usr_1.wav") == tmp_path / "usr_1.wav"

    @pytest.mark.parametrize(
        "evil",
        ["../../.env", "assets/../../.env", "..\\..\\secrets.txt", "/etc/passwd", "C:\\Windows\\win.ini", "assets/sub/../../x"],
    )
    def test_traversal_cannot_leave_the_assets_folder(self, tmp_path, evil):
        resolved = asset_path(tmp_path, evil)
        assert resolved.parent == tmp_path

    def test_roundtrip(self, tmp_path):
        path = tmp_path / "bot_9.wav"
        assert asset_path(tmp_path, asset_ref(path)) == path


class TestSttCannotReadOutsideAssets:
    @pytest.mark.asyncio
    async def test_crafted_path_is_not_followed(self, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.wav").write_bytes(b"top secret")
        assets = tmp_path / "assets"
        assets.mkdir()

        with (
            patch("temporal.activities.stt_activity.AsyncDeepgramClient") as client,
            patch("temporal.activities.stt_activity.settings") as mock_settings,
        ):
            mock_settings.assets_dir = assets
            client.return_value = MagicMock(listen=MagicMock())
            client.return_value.listen.v1.media.transcribe_file = AsyncMock()

            from temporal.activities.stt_activity import transcribe_audio

            with pytest.raises(FileNotFoundError):
                await transcribe_audio("../outside/secret.wav")

        client.return_value.listen.v1.media.transcribe_file.assert_not_called()
