"""static/build.mjs injects API_BASE_URL into the page at deploy time (skipped if Node is absent)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parent.parent / "static"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js not installed")


def _build(tmp_path: Path, **env) -> subprocess.CompletedProcess:
    # Work on a copy so the real static/ folder never gets a dist/ left in it.
    shutil.copy(STATIC / "build.mjs", tmp_path / "build.mjs")
    shutil.copy(STATIC / "index.html", tmp_path / "index.html")
    return subprocess.run(
        [NODE, "build.mjs"], cwd=tmp_path, env={"PATH": "", **env}, capture_output=True, text=True, timeout=30
    )


def _api_base(tmp_path: Path) -> str:
    html = (tmp_path / "dist" / "index.html").read_text(encoding="utf-8")
    marker = '<meta name="api-base" content="'
    start = html.index(marker) + len(marker)
    return html[start: html.index('"', start)]


def test_injects_the_backend_origin(tmp_path):
    result = _build(tmp_path, API_BASE_URL="https://api.example.com/")
    assert result.returncode == 0, result.stderr
    assert _api_base(tmp_path) == "https://api.example.com"   # trailing slash dropped


def test_only_the_origin_is_kept_so_markup_cannot_be_injected(tmp_path):
    result = _build(tmp_path, API_BASE_URL='https://a.com/"><script>alert(1)</script>')
    assert result.returncode == 0, result.stderr
    html = (tmp_path / "dist" / "index.html").read_text(encoding="utf-8")
    assert _api_base(tmp_path) == "https://a.com"
    assert "alert(1)" not in html.split("<style>")[0]


def test_preview_build_without_a_url_calls_its_own_origin(tmp_path):
    assert _build(tmp_path).returncode == 0
    assert _api_base(tmp_path) == ""


@pytest.mark.parametrize("bad", ["http://api.example.com", "not a url", "ftp://x.example.com"])
def test_insecure_or_invalid_urls_fail_the_build(tmp_path, bad):
    result = _build(tmp_path, API_BASE_URL=bad)
    assert result.returncode != 0
    assert not (tmp_path / "dist").exists()


def test_local_http_is_allowed_for_development(tmp_path):
    assert _build(tmp_path, API_BASE_URL="http://localhost:8000").returncode == 0
    assert _api_base(tmp_path) == "http://localhost:8000"


def test_a_production_build_without_a_url_fails_loudly(tmp_path):
    result = _build(tmp_path, VERCEL_ENV="production")
    assert result.returncode != 0
    assert "API_BASE_URL" in result.stderr
    assert not (tmp_path / "dist").exists()


class TestExplicitNoBackendOptIn:
    def test_production_build_passes_only_with_the_explicit_opt_in(self, tmp_path):
        result = _build(tmp_path, VERCEL_ENV="production", ALLOW_NO_BACKEND="1")
        assert result.returncode == 0, result.stderr
        assert _api_base(tmp_path) == ""

    @pytest.mark.parametrize("value", ["0", "", "true", "yes"])
    def test_anything_other_than_1_keeps_the_guard_on(self, tmp_path, value):
        assert _build(tmp_path, VERCEL_ENV="production", ALLOW_NO_BACKEND=value).returncode != 0

    def test_a_real_backend_url_still_wins_over_the_opt_in(self, tmp_path):
        result = _build(
            tmp_path, VERCEL_ENV="production", ALLOW_NO_BACKEND="1", API_BASE_URL="https://api.example.com"
        )
        assert result.returncode == 0, result.stderr
        assert _api_base(tmp_path) == "https://api.example.com"


def test_the_page_checks_for_a_backend_and_says_so_when_there_is_none():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert "checkBackend" in html
    assert "No backend connected" in html
