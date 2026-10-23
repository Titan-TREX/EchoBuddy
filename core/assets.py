"""
References to audio files, shared by both pipelines and the UI.

Audio is referred to everywhere (API responses, workflow inputs, database rows) as
`assets/<filename>`, matching the `/assets` URL the server serves. The folder it
actually lives in is `settings.assets_dir`, which can be anywhere (e.g. under a
mounted `DATA_DIR`), so a reference never encodes a real filesystem path and is
resolved by file name only - a crafted reference like `../../.env` cannot reach
outside the assets folder.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

ASSET_PREFIX = "assets"


def asset_ref(path: Path) -> str:
    """`assets/<filename>` for a file stored in the assets folder."""
    return f"{ASSET_PREFIX}/{path.name}"


def asset_path(assets_dir: Path, ref: str) -> Path:
    """The file in `assets_dir` a reference points to (by name only)."""
    return assets_dir / PurePosixPath(ref.replace("\\", "/")).name
