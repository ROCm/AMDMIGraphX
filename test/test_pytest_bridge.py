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
"""Pytest bridge for the MIGraphX test suite.

It exposes every test registered in a ``CTestTestfile.cmake`` to pytest as its
own parametrized case, reading the list (with arguments and per-test
environment) from ``ctest --show-only`` so pytest and ctest always run the exact
same set.

NOTE: the file is named ``test_*`` on purpose so pytest's default discovery
(``python_files``) collects it during directory/``--pyargs`` recursion
"""
import json
import os
import re
import shutil
import subprocess

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_FIXTURE_PROPS = ("FIXTURES_REQUIRED", "FIXTURES_SETUP", "FIXTURES_CLEANUP")


def _resolve_test_dir():
    env_dir = os.environ.get("MIGRAPHX_TEST_DIR")
    if env_dir:
        return env_dir
    for candidate in (_HERE, os.getcwd()):
        if os.path.exists(os.path.join(candidate, "CTestTestfile.cmake")):
            return candidate
    return _HERE


_TEST_DIR = _resolve_test_dir()
_BIN_DIR = os.path.join(_TEST_DIR, "bin")


def _migraphx_lib_dir():
    """Directory of the installed migraphx wheel (carries libmigraphx*.so)."""
    try:
        import migraphx
    except ImportError:
        return None
    return os.path.dirname(os.path.abspath(migraphx.__file__))

_LIB_DIRS = [d for d in (os.path.join(_TEST_DIR, "lib"), _migraphx_lib_dir())
             if d and os.path.isdir(d)]


def _make_test(name, command, env=None, fail_regexes=None, skip=None):
    return {
        "name": name,
        "command": command,
        "env": env or {},
        # CTest applies no default fail regex; the rocm harness sets "FAILED".
        "fail_regexes": fail_regexes or ["FAILED"],
        "skip": skip,
    }


def _make_env(extra):
    env = dict(os.environ)
    env.update(extra)
    # Prepend our lib dirs so they win over any LD_LIBRARY_PATH a test's own
    # ctest ENVIRONMENT may have set in `extra`.
    if _LIB_DIRS:
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            _LIB_DIRS + ([existing] if existing else []))
    return env


def _discover_via_ctest():
    ctest = shutil.which("ctest")
    if not ctest or not os.path.exists(os.path.join(_TEST_DIR, "CTestTestfile.cmake")):
        return None
    try:
        out = subprocess.check_output(
            [ctest, "--show-only=json-v1", "--test-dir", _TEST_DIR],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    data = json.loads(out)
    tests = []
    for entry in data.get("tests", []):
        command = entry.get("command")
        if not command:
            continue
        env_extra = {}
        fail_regexes = None
        needs_fixture = False
        for prop in entry.get("properties", []):
            name = prop.get("name")
            if name == "ENVIRONMENT":
                for item in prop.get("value", []):
                    key, _, value = item.partition("=")
                    env_extra[key] = value
            elif name == "FAIL_REGULAR_EXPRESSION" and prop.get("value"):
                value = prop["value"]
                fail_regexes = value if isinstance(value, list) else [value]
            elif name in _FIXTURE_PROPS and prop.get("value"):
                needs_fixture = True
        tests.append(_make_test(
            entry["name"], command, env=env_extra, fail_regexes=fail_regexes,
            skip="requires ctest fixtures; run via ctest" if needs_fixture else None))
    return tests


def _discover_via_bin():
    if not os.path.isdir(_BIN_DIR):
        return []
    tests = []
    for name in sorted(os.listdir(_BIN_DIR)):
        path = os.path.join(_BIN_DIR, name)
        # Only test executables; bin/ may also hold non-test binaries (driver).
        if name.startswith("test_") and os.path.isfile(path):
            tests.append(_make_test(name, [path]))
    return tests


def _discover():
    return _discover_via_ctest() or _discover_via_bin()


_TESTS = _discover()


@pytest.mark.parametrize("spec", _TESTS, ids=[t["name"] for t in _TESTS])
def test_migraphx(spec):
    if spec["skip"]:
        pytest.skip(spec["skip"])
    # Wheel installs can drop the executable bit; restore it on the binary we run.
    try:
        os.chmod(spec["command"][0], 0o755)
    except OSError:
        pass
    result = subprocess.run(
        spec["command"],
        cwd=_TEST_DIR,
        env=_make_env(spec["env"]),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    # Mirror the CTest pass criteria: zero exit and no FAIL_REGULAR_EXPRESSION match.
    failed = any(re.search(p, result.stdout) for p in spec["fail_regexes"])
    assert result.returncode == 0 and not failed, (
        "{name} failed (exit {code}):\n{out}".format(
            name=spec["name"], code=result.returncode, out=result.stdout))


if not _TESTS:
    def test_migraphx_suite_discovered():
        pytest.skip(
            "No MIGraphX tests found. Set MIGRAPHX_TEST_DIR to a build/install "
            "dir containing CTestTestfile.cmake, or run from one.")
