/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/compile_options.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/permutation.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/shape.hpp>
#include <test.hpp>

static migraphx::program compile_conv(const std::string& order)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", {migraphx::shape::float_type, {1, 8, 8, 8}});
    auto w   = mm->add_parameter("w", {migraphx::shape::float_type, {8, 8, 3, 3}});
    mm->add_return({mm->add_instruction(migraphx::make_op("convolution"), x, w)});

    migraphx::compile_options options;
    options.backend_options["convolution_layout"] = order;
    p.compile(migraphx::make_target("gpu"), options);
    return p;
}

static bool has_channels_last(const migraphx::program& p)
{
    const auto* mm = p.get_main_module();
    return std::any_of(mm->begin(), mm->end(), [](const migraphx::instruction& ins) {
        const auto& s = ins.get_shape();
        return s.ndim() == 4 and
               migraphx::find_permutation(s) == std::vector<std::int64_t>{0, 2, 3, 1};
    });
}

TEST_CASE(channels_first) { EXPECT(not has_channels_last(compile_conv("channels_first"))); }

TEST_CASE(channels_last) { EXPECT(has_channels_last(compile_conv("channels_last"))); }

TEST_CASE(unknown_order)
{
    EXPECT(test::throws([] { compile_conv("nhwc"); }));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
