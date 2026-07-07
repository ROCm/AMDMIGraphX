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

#include <string_view>
#include <migraphx/logger.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include "test.hpp"

// An op whose compute_shape throws, used to drive the exception path of the debug-symbol guards.
struct throw_shape_op
{
    std::string name() const { return "throw_shape_op"; }
    migraphx::shape compute_shape(std::vector<migraphx::shape>) const
    {
        MIGRAPHX_THROW("throw_shape_op: shape failure");
    }
};

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

// Replacing a symbolized instruction with an op whose compute_shape throws fires the scope-fail
// guard, which must log the instruction's debug symbols.
TEST_CASE(replace_error_logs_debug_symbols)
{
    migraphx::module m;
    auto x    = m.add_parameter("x", {migraphx::shape::float_type, {2, 3}});
    auto relu = m.add_instruction(migraphx::make_op("relu"), x);
    m.add_debug_symbols(relu, {"test_symbol"});
    m.add_return({relu});

    auto log = capture_log([&] { m.replace_instruction(relu, throw_shape_op{}, {x}); });
    EXPECT(log.find("test_symbol") != std::string::npos);
}

// Without debug symbols nothing is logged, exercising the early-return path of the guard helper.
TEST_CASE(replace_error_without_debug_symbols_not_logged)
{
    migraphx::module m;
    auto x    = m.add_parameter("x", {migraphx::shape::float_type, {2, 3}});
    auto relu = m.add_instruction(migraphx::make_op("relu"), x);
    m.add_return({relu});

    auto log = capture_log([&] { m.replace_instruction(relu, throw_shape_op{}, {x}); });
    EXPECT(log.find("debug symbols") == std::string::npos);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
