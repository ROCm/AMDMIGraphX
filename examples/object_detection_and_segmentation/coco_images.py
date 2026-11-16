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
"""Download and cache the ten COCO test2017 images used by these examples."""

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from urllib.request import Request, urlopen

COCO_TEST2017_ROOT = "http://images.cocodataset.org/test2017"


@dataclass(frozen=True)
class CocoImage:
    filename: str
    size: int
    sha256: str

    @property
    def url(self):
        return f"{COCO_TEST2017_ROOT}/{self.filename}"


COCO_IMAGES = (
    CocoImage(
        "000000000001.jpg",
        159192,
        "91d74bd2a576101fdc78a448f8f8c52816dda6ec2e37e0bffcb9e30e950647ae",
    ),
    CocoImage(
        "000000000016.jpg",
        230884,
        "e77b3610a4d60decaf6399090e772f45f6bb69d13902e46f1aa67e6ffa44c05d",
    ),
    CocoImage(
        "000000000057.jpg",
        189943,
        "2c00c990da26ba8b195d8e1ffaa1085cad53be95ec8de83c5d4941c1dd0912f2",
    ),
    CocoImage(
        "000000000069.jpg",
        112853,
        "47193896385f8d284f3e20dd457253d2e51d4bd77e7e3531e4731a5f131a4493",
    ),
    CocoImage(
        "000000000080.jpg",
        113415,
        "77393198302233c5df1f3b5bfe9e7fcd20af37402db4e29692d608aa0090b678",
    ),
    CocoImage(
        "000000000090.jpg",
        254598,
        "9ff72dc5f69998b7777802f2d6c6dc261c67cf9f5ade8e0f94e030edd422467b",
    ),
    CocoImage(
        "000000000106.jpg",
        247602,
        "4797648ce66c98cd4d12945b314120aa85b73f32db6a33f5bd07d7a290e872bb",
    ),
    CocoImage(
        "000000000108.jpg",
        272953,
        "9dac65dacd78f1dbd389025524ca74055f3441419ec694e54633f699ef4ae1ab",
    ),
    CocoImage(
        "000000000128.jpg",
        212272,
        "4741971ce3b5f5d3cf46b91ecce5cde63bcad143045c9739ecd39fee8bcd1832",
    ),
    CocoImage(
        "000000000155.jpg",
        79729,
        "413ead1673ab92a827f3225de59b3100f591bb64f7886b31b692a712592ebe49",
    ),
)


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as data:
        for block in iter(lambda: data.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_image(path, image):
    return (path.is_file() and path.stat().st_size == image.size
            and _file_sha256(path) == image.sha256)


def _download_image(image, destination):
    request = Request(image.url, headers={"User-Agent": "MIGraphX-example"})
    temporary = destination.with_name(f"{destination.name}.part-{os.getpid()}")
    digest = hashlib.sha256()
    downloaded = 0

    print(f"Downloading {image.url}")
    try:
        with urlopen(request, timeout=60) as response, temporary.open(
                "wb") as output:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                output.write(block)
                digest.update(block)
                downloaded += len(block)
        if downloaded != image.size:
            raise RuntimeError(
                f"Downloaded {downloaded} bytes for {image.filename}, "
                f"expected {image.size}")
        if digest.hexdigest() != image.sha256:
            raise RuntimeError(
                f"Checksum mismatch for {image.filename}: "
                f"expected {image.sha256}, found {digest.hexdigest()}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def get_coco_images(cache_dir):
    """Return paths to the verified first ten sorted COCO test2017 images."""
    destination = (Path(cache_dir).expanduser().resolve() / "coco" /
                   "test2017")
    destination.mkdir(parents=True, exist_ok=True)

    paths = []
    for image in COCO_IMAGES:
        path = destination / image.filename
        if not _valid_image(path, image):
            path.unlink(missing_ok=True)
            _download_image(image, path)
        paths.append(path)
    return paths
