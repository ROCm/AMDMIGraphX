/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/rewrite_reduce.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/ranges.hpp>
#include <test.hpp>

void run_pass(migraphx::module& m)
{
    migraphx::run_passes(m, {migraphx::rewrite_reduce{}, migraphx::dead_code_elimination{}});
}

TEST_CASE(softmax)
{
    migraphx::shape s{migraphx::shape::float_type, {10, 1000}};
    migraphx::module m;
    auto x       = m.add_parameter("x", s);
    auto softmax = m.add_instruction(migraphx::make_op("softmax", {{"axis", 1}}), x);
    m.add_return({softmax});
    run_pass(m);
    EXPECT(none_of(migraphx::iterator_for(m), [](auto ins) { return ins->name() == "softmax"; }));

    auto reduces = find_all(migraphx::iterator_for(m),
                            [&](auto ins) { return migraphx::contains(ins->name(), "reduce"); });
    EXPECT(all_of(reduces, [](auto ins) {
        auto axes = ins->get_operator().to_value()["axes"].template to_vector<int64_t>();
        return axes.size() == 1 and axes[0] == 1;
    }));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
