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

#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/sym.hpp>
#include <test.hpp>

// Run a range of up to 8 elements on the GPU, where it is computed on the host
static std::vector<int64_t> run_gpu_range(int64_t limit)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::int64_type, {1}};
    auto start = mm->add_parameter("start", s);
    auto lim   = mm->add_parameter("limit", s);
    auto delta = mm->add_parameter("delta", s);
    const migraphx::shape::dynamic_dimension length{migraphx::sym::var("n", {0, 8})};
    auto r = mm->add_instruction(
        migraphx::make_op("dynamic_range",
                          {{"max_output", 8}, {"output_dim", migraphx::to_value(length)}}),
        start,
        lim,
        delta);
    mm->add_return({r});

    migraphx::target t = migraphx::make_target("gpu");
    p.compile(t);
    migraphx::parameter_map params;
    params["start"] = t.copy_to(migraphx::literal{s, {0}}.get_argument());
    params["limit"] = t.copy_to(migraphx::literal{s, {limit}}.get_argument());
    params["delta"] = t.copy_to(migraphx::literal{s, {1}}.get_argument());
    // The output buffer is a parameter of the compiled program as well
    for(auto&& x : p.get_parameter_shapes())
    {
        if(params.count(x.first) == 0)
            params[x.first] = t.allocate(x.second);
    }
    auto result = t.copy_from(p.eval(params).back());
    std::vector<int64_t> values;
    result.visit([&](auto v) { values.assign(v.begin(), v.end()); });
    return values;
}

TEST_CASE(dynamic_range_bounded) { EXPECT(run_gpu_range(3) == std::vector<int64_t>{0, 1, 2}); }

// An empty range, such as no detection above a threshold, gives an empty output
TEST_CASE(dynamic_range_bounded_empty) { EXPECT(run_gpu_range(0).empty()); }

int main(int argc, const char* argv[]) { test::run(argc, argv); }
