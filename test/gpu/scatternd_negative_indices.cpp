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
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>

#include <test.hpp>

#include <cstdint>
#include <vector>

TEST_CASE(scatternd_negative_indices_gpu_test)
{
    // A negative index counts back from the end of its dimension.
    migraphx::shape ds{migraphx::shape::float_type, {1, 8, 4}};
    migraphx::shape is{migraphx::shape::int64_type, {1, 3, 2}};
    migraphx::shape us{migraphx::shape::float_type, {1, 3, 4}};

    migraphx::program p;
    auto* mm  = p.get_main_module();
    auto data = mm->add_parameter("data", ds);
    auto indices =
        mm->add_literal(migraphx::literal{is, std::vector<int64_t>{0, 2, -1, -1, -1, 1}});
    auto updates = mm->add_parameter("update", us);
    auto scatternd =
        mm->add_instruction(migraphx::make_op("scatternd_none"), data, indices, updates);
    mm->add_return({scatternd});

    migraphx::target t = migraphx::make_target("gpu");
    p.compile(t);
    std::vector<float> data_vec(ds.elements(), 0);
    std::vector<float> upd_vec{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    migraphx::parameter_map params;
    for(auto&& x : p.get_parameter_shapes())
        params[x.first] = t.allocate(x.second);
    params["data"]   = t.copy_to(migraphx::argument{ds, data_vec.data()});
    params["update"] = t.copy_to(migraphx::argument{us, upd_vec.data()});
    auto result      = t.copy_from(p.eval(params).back());
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{0, 0, 0, 0, 9, 10, 11, 12, 1, 2, 3, 4, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0,  0,  0,  0, 0, 0, 0, 5, 6, 7, 8};

    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
