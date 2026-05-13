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
#include <migraphx/migraphx.h>
#include <migraphx/migraphx.hpp>
#include <migraphx/logger.hpp>
#include <read_onnx.hpp>
#include "test.hpp"

namespace {
struct warning_sink
{
    std::vector<std::string> messages;
    std::size_t id;

    warning_sink()
    {
        id = migraphx::log::add_sink(
            [this](migraphx::log::severity, std::string_view msg, migraphx::source_location) {
                messages.emplace_back(msg);
            },
            migraphx::log::severity::warn);
    }

    ~warning_sink() { migraphx::log::remove_sink(id); }

    bool any_unbound_dim_warning() const
    {
        return std::any_of(messages.begin(), messages.end(), [](const std::string& m) {
            return m.find("unbound symbolic dimension") != std::string::npos;
        });
    }
};
} // namespace

TEST_CASE(set_default_dim_value_suppresses_unbound_dim_warning)
{
    warning_sink sink;
    migraphx::onnx_options opts;
    opts.set_default_dim_value(4);
    (void)read_onnx("dim_param_test.onnx", opts);

    EXPECT(not sink.any_unbound_dim_warning());
}

TEST_CASE(set_default_dyn_dim_value_suppresses_unbound_dim_warning)
{
    warning_sink sink;
    migraphx::onnx_options opts;
    opts.set_default_dyn_dim_value(migraphx::dynamic_dimension{1, 1});
    (void)read_onnx("dim_param_test.onnx", opts);

    EXPECT(not sink.any_unbound_dim_warning());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
