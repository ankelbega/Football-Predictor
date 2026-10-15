"""Exercise mocked HTTP download and cache behavior."""


from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

from footballpred.config import DataConfig
from footballpred.ingest.download import download_source, ingest_dataset


class FakeResponse:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def read(self) -> bytes:
        return self.content

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> bool:
        return False


def make_config(tmp_path: Path, source_url: str = "https://example.test/{season}/{league}.csv") -> DataConfig:
    return DataConfig(
        source_url=source_url,
        leagues=["E0"],
        seasons=["2122"],
        cache_dir=tmp_path / "raw",
        processed_path=tmp_path / "processed.csv",
        validation_report_path=tmp_path / "validation.json",
        stats_path=tmp_path / "summary.json",
    )


def test_download_source_reuses_cached_file_without_http(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Risk: cached raw files must prevent unnecessary network calls."""
    config = make_config(tmp_path)
    cached = config.cache_dir / "E0_2122.csv"
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    cached.write_text("cached", encoding="utf-8")

    def fail_urlopen(*args, **kwargs):
        raise AssertionError("network should not be called")

    monkeypatch.setattr("footballpred.ingest.download.urlopen", fail_urlopen)
    paths = download_source(config, refresh=False)
    assert paths == [cached]


def test_download_source_writes_successful_http_response(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Risk: successful downloads must be cached as the exact response bytes."""
    config = make_config(tmp_path)

    def fake_urlopen(request, timeout):
        assert timeout == config.request_timeout_seconds
        return FakeResponse(b"Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n01/01/21,A,B,1,0,H\n")

    monkeypatch.setattr("footballpred.ingest.download.urlopen", fake_urlopen)
    paths = download_source(config, refresh=False)
    assert len(paths) == 1
    assert paths[0].read_bytes() == b"Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n01/01/21,A,B,1,0,H\n"


def test_download_source_refresh_bypasses_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Risk: refresh mode must not silently reuse a stale cached file."""
    config = make_config(tmp_path)
    cached = config.cache_dir / "E0_2122.csv"
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    cached.write_text("old", encoding="utf-8")

    def fake_urlopen(request, timeout):
        return FakeResponse(b"new")

    monkeypatch.setattr("footballpred.ingest.download.urlopen", fake_urlopen)
    paths = download_source(config, refresh=True)
    assert paths[0].read_bytes() == b"new"


def test_download_source_skips_failed_and_empty_responses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Risk: transient HTTP failures and empty responses must be logged and skipped, not crash callers."""
    config = DataConfig(
        source_url="https://example.test/{season}/{league}.csv",
        leagues=["E0", "E1"],
        seasons=["2122"],
        cache_dir=tmp_path / "raw",
        processed_path=tmp_path / "processed.csv",
        validation_report_path=tmp_path / "validation.json",
        stats_path=tmp_path / "summary.json",
    )

    def fake_urlopen(request, timeout):
        if "E1" in request.full_url:
            raise HTTPError(request.full_url, 404, "not found", hdrs=None, fp=None)
        return FakeResponse(b"   ")

    monkeypatch.setattr("footballpred.ingest.download.urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="No source files"):
        download_source(config, refresh=True)


def test_download_source_raises_when_all_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Risk: callers should know when no source data is available instead of receiving an empty list."""
    config = make_config(tmp_path)

    def fake_urlopen(request, timeout):
        raise URLError("boom")

    monkeypatch.setattr("footballpred.ingest.download.urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="No source files"):
        download_source(config, refresh=True)


def test_ingest_dataset_skips_unusable_source_and_raises_if_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Risk: unusable cached sources must be skipped when alternatives exist but failure must be reported when all fail."""
    source = tmp_path / "E0_2122.csv"
    source.write_text("not a csv with required columns", encoding="utf-8")
    config = make_config(tmp_path)
    monkeypatch.setattr("footballpred.ingest.download.download_source", lambda cfg, refresh=False: [source])
    with pytest.raises(RuntimeError, match="No usable match rows"):
        ingest_dataset(config, refresh=False)
