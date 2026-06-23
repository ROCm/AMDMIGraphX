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
"""Pytest bridge for the MIGraphX test suite

NOTE: the file is named ``test_*`` on purpose so pytest's default discovery can find it
"""
import os
import shutil
import subprocess

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))


def _resolve_test_dir():
    env_dir = os.environ.get("MIGRAPHX_TEST_DIR")
    if env_dir:
        return env_dir
    for candidate in (_HERE, os.getcwd()):
        if os.path.exists(os.path.join(candidate, "CTestTestfile.cmake")):
            return candidate
    return _HERE


def _migraphx_lib_dir():
    try:
        import migraphx
    except ImportError:
        return None
    return os.path.dirname(os.path.abspath(migraphx.__file__))


def _ctest_env(test_dir):
    env = dict(os.environ)
    lib_dirs = [d for d in (os.path.join(test_dir, "lib"), _migraphx_lib_dir())
                if d and os.path.isdir(d)]
    if lib_dirs:
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            lib_dirs + ([existing] if existing else []))
    return env


def _ensure_executable(test_dir):
    bin_dir = os.path.join(test_dir, "bin")
    if not os.path.isdir(bin_dir):
        return
    for name in os.listdir(bin_dir):
        try:
            os.chmod(os.path.join(bin_dir, name), 0o755)
        except OSError:
            pass


@pytest.mark.skipif(shutil.which("ctest") is None,
                    reason="ctest not found; install CMake to run the suite")
def test_migraphx():
    test_dir = _resolve_test_dir()
    if not os.path.exists(os.path.join(test_dir, "CTestTestfile.cmake")):
        pytest.skip(
            f"No CTestTestfile.cmake in {test_dir}; set MIGRAPHX_TEST_DIR to a "
            "build or installed-tests directory.")
    _ensure_executable(test_dir)
    result = subprocess.run(
        ["ctest", "--test-dir", test_dir, "-j", str(os.cpu_count() or 1),
         "--output-on-failure"],
        env=_ctest_env(test_dir),
    )
    assert result.returncode == 0, f"ctest reported failures (exit {result.returncode})"
