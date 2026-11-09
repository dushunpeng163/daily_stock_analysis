#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""下载、校验并规范化 Qlib 社区 A 股二进制数据目录。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile

import requests


def _download(url: str, destination: Path, token: str = "") -> None:
    headers = {"Accept": "application/octet-stream"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with requests.get(url, headers=headers, stream=True, timeout=(15, 120)) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def _download_with_fallback(urls: list[str], destination: Path, token: str = "") -> str:
    errors = []
    for url in urls:
        try:
            destination.unlink(missing_ok=True)
            _download(url, destination, token)
            return url
        except (requests.RequestException, OSError) as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError("Qlib 数据所有下载通道均失败: " + " | ".join(errors))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            if member.issym() or member.islnk():
                raise RuntimeError(f"压缩包包含链接，拒绝解压: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise RuntimeError(f"压缩包包含特殊文件，拒绝解压: {member.name}")
            target = (destination / member.name).resolve()
            if destination not in target.parents and target != destination:
                raise RuntimeError(f"压缩包包含越界路径: {member.name}")
        tar.extractall(destination)


def _find_cn_data(root: Path) -> Path:
    candidates = [root] + [path for path in root.rglob("*") if path.is_dir()]
    for path in candidates:
        if all((path / child).exists() for child in ("calendars", "instruments", "features")):
            return path
    raise RuntimeError("压缩包中未找到包含 calendars/instruments/features 的 Qlib 数据目录")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-url", required=True)
    parser.add_argument("--manifest-file", type=Path)
    parser.add_argument("--archive-url", required=True)
    parser.add_argument("--archive-fallback-url", action="append", default=[])
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--metadata-out", type=Path, required=True)
    args = parser.parse_args()

    if args.manifest_file:
        manifest = json.loads(args.manifest_file.read_text(encoding="utf-8"))
    else:
        manifest_response = requests.get(args.manifest_url, timeout=30)
        manifest_response.raise_for_status()
        manifest = manifest_response.json()
    expected_sha = str(manifest["archive_sha256"]).removeprefix("sha256:")
    with tempfile.TemporaryDirectory(prefix="qlib-data-") as tmp:
        tmp_path = Path(tmp)
        archive = tmp_path / "qlib_bin.tar.gz"
        extracted = tmp_path / "extracted"
        extracted.mkdir()
        _download_with_fallback(
            [args.archive_url, *args.archive_fallback_url],
            archive,
            os.getenv("GH_TOKEN", ""),
        )
        actual_sha = _sha256(archive)
        if actual_sha != expected_sha:
            raise RuntimeError(f"Qlib 数据 SHA256 不匹配: expected={expected_sha}, actual={actual_sha}")
        _safe_extract(archive, extracted)
        source = _find_cn_data(extracted)
        if args.destination.exists():
            shutil.rmtree(args.destination)
        args.destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, args.destination)

    args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
