#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#####################################################################################
"""Download and unpack ONNX Model Zoo fixtures used by these examples."""

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
from urllib.request import Request, urlopen

MODEL_ZOO_ROOT = (
    "https://github.com/mitzen/onnx-zoo-models/raw/main/"
    "vision/object_detection_segmentation")


@dataclass(frozen=True)
class ModelZooFixture:
    directory: str
    archive: str
    model: str
    sha256: str

    @property
    def url(self):
        return f"{MODEL_ZOO_ROOT}/{self.directory}/model/{self.archive}"


def default_cache_dir():
    cache_home = os.environ.get("XDG_CACHE_HOME")
    if cache_home:
        return Path(cache_home) / "migraphx" / "model_zoo"
    return Path.home() / ".cache" / "migraphx" / "model_zoo"


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as data:
        for block in iter(lambda: data.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _find_fixture(root, fixture):
    marker = root / ".complete"
    if (not marker.is_file()
            or marker.read_text(encoding="utf-8").strip() != fixture.sha256):
        return None

    models = list(root.rglob(fixture.model))
    if len(models) != 1:
        return None

    data_dir = models[0].parent / "test_data_set_0"
    if not data_dir.is_dir():
        return None
    return models[0], data_dir


def _download(fixture, destination):
    request = Request(fixture.url, headers={"User-Agent": "MIGraphX-example"})
    temporary = destination.with_name(f"{destination.name}.part-{os.getpid()}")
    digest = hashlib.sha256()
    downloaded = 0

    print(f"Downloading {fixture.url}")
    try:
        with urlopen(request, timeout=60) as response, temporary.open(
                "wb") as output:
            size_header = response.headers.get("Content-Length")
            expected_size = int(size_header) if size_header else None
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                output.write(block)
                digest.update(block)
                downloaded += len(block)
                if expected_size is not None and expected_size > 0:
                    percent = downloaded * 100 / expected_size
                    print(f"\rDownloaded {percent:5.1f}%", end="", flush=True)
        if expected_size is not None and expected_size > 0:
            print()
        if expected_size is not None and downloaded != expected_size:
            raise RuntimeError(
                f"Downloaded {downloaded} bytes, expected {expected_size}")
        if digest.hexdigest() != fixture.sha256:
            raise RuntimeError(
                f"Checksum mismatch for {fixture.archive}: "
                f"expected {fixture.sha256}, found {digest.hexdigest()}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_archive_members(archive, destination):
    destination = destination.resolve()
    members = archive.getmembers()
    for member in members:
        try:
            (destination / member.name).resolve().relative_to(destination)
        except ValueError as error:
            raise RuntimeError(
                f"Archive member escapes extraction directory: {member.name}"
            ) from error
        if not (member.isfile() or member.isdir()):
            raise RuntimeError(
                f"Unsupported archive member type: {member.name}")
    return members


def _extract(fixture, archive_path, destination):
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-",
                         dir=destination.parent))
    try:
        print(f"Extracting {archive_path}")
        with tarfile.open(archive_path, "r:gz") as archive:
            members = _validate_archive_members(archive, temporary)
            archive.extractall(temporary, members=members)
        (temporary / ".complete").write_text(f"{fixture.sha256}\n",
                                             encoding="utf-8")
        if destination.exists():
            shutil.rmtree(destination)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def get_fixture(fixture, cache_dir):
    """Return the downloaded model and packaged test-data directory."""
    cache_dir = Path(cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not fixture.archive.endswith(".tar.gz"):
        raise ValueError(
            f"Expected a .tar.gz fixture archive, found {fixture.archive}")
    destination = cache_dir / fixture.archive[:-len(".tar.gz")]

    located = _find_fixture(destination, fixture)
    if located is not None:
        return located

    archive_path = cache_dir / fixture.archive
    if archive_path.is_file():
        if _file_sha256(archive_path) != fixture.sha256:
            archive_path.unlink()
    if not archive_path.is_file():
        _download(fixture, archive_path)

    _extract(fixture, archive_path, destination)
    archive_path.unlink()
    located = _find_fixture(destination, fixture)
    if located is None:
        raise RuntimeError(
            f"{fixture.archive} did not contain {fixture.model} and "
            "test_data_set_0 in the expected layout")
    return located
