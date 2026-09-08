/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <migraphx/logger.hpp>
#include <string_view>
#include <onnx_test.hpp>

// Returns the debug-level log emitted while running f() (which is expected to throw).
template <class F>
static std::string capture_log(F f)
{
    std::string captured;
    auto id = migraphx::log::add_sink(
        [&](migraphx::log::severity, std::string_view msg, migraphx::source_location) {
            captured.append(msg.begin(), msg.end());
        },
        migraphx::log::severity::debug);
    try
    {
        f();
    }
    // cppcheck-suppress migraphx-EmptyCatchStatement
    catch(...) // the sink references `captured`, so it must be removed on every path
    {
    }
    migraphx::log::remove_sink(id);
    return captured;
}

// Parsing a model that fails while use_debug_symbols is enabled exercises the parser's
// log-debug-symbols-on-exception path; the original parse error must still propagate and the
// failing node's debug symbol must be logged.
TEST_CASE(parse_error_with_debug_symbols_logged)
{
    migraphx::onnx_options options;
    options.use_debug_symbols = true;
    auto log = capture_log([&] { read_onnx("resize_invalid_mode_test.onnx", options); });
    EXPECT(log.find("Exception thrown while parsing node") != std::string::npos);
    EXPECT(log.find("Resize") != std::string::npos);
}

// Without debug symbols the parser must not emit the debug-symbol log, but the original parse error
// must still propagate.
TEST_CASE(parse_error_without_debug_symbols_still_throws)
{
    migraphx::onnx_options options;
    options.use_debug_symbols = false;
    auto log = capture_log([&] { read_onnx("resize_invalid_mode_test.onnx", options); });
    EXPECT(log.find("Exception thrown while parsing node") == std::string::npos);
}
