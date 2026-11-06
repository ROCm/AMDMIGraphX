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
import os
import tempfile

import migraphx


def test_log_level_enum_exists():
    assert hasattr(migraphx, 'log_level')
    for name in ['none', 'error', 'warn', 'info', 'debug', 'trace']:
        assert hasattr(migraphx.log_level, name)


def test_set_log_level_and_is_enabled():
    # Lowering the level to warn disables info/debug/trace but keeps error/warn.
    # This is how a Python user silences the INFO log spam (issue #5359).
    migraphx.set_log_level(migraphx.log_level.warn)
    assert migraphx.is_log_enabled(migraphx.log_level.error)
    assert migraphx.is_log_enabled(migraphx.log_level.warn)
    assert not migraphx.is_log_enabled(migraphx.log_level.info)
    assert not migraphx.is_log_enabled(migraphx.log_level.debug)

    # Raising the level back to info re-enables info (but not debug).
    migraphx.set_log_level(migraphx.log_level.info)
    assert migraphx.is_log_enabled(migraphx.log_level.info)
    assert not migraphx.is_log_enabled(migraphx.log_level.debug)

    # none disables every level.
    migraphx.set_log_level(migraphx.log_level.none)
    assert not migraphx.is_log_enabled(migraphx.log_level.error)

    # Restore the default stderr level.
    migraphx.set_log_level(migraphx.log_level.info)


def test_add_file_logger():
    # Keep stderr quiet while exercising the file sink.
    migraphx.set_log_level(migraphx.log_level.none)
    log_dir = tempfile.mkdtemp()
    log_path = os.path.join(log_dir, 'migraphx_py_test.log')

    sink_id = migraphx.add_file_logger(log_path, migraphx.log_level.info)
    # Sink 0 is the default stderr sink, so any added sink must have id > 0.
    assert sink_id > 0
    # add_file_logger opens the file (append mode), so it must now exist.
    assert os.path.exists(log_path)

    # Restore the default stderr level.
    migraphx.set_log_level(migraphx.log_level.info)


test_log_level_enum_exists()
test_set_log_level_and_is_enabled()
test_add_file_logger()
