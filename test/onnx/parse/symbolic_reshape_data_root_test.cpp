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

#include <migraphx/argument.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/register_target.hpp>
#include <onnx_test.hpp>

#include <vector>

TEST_CASE(symbolic_reshape_data_root_test)
{
    auto p   = read_onnx("symbolic_reshape_data_root_test.onnx");
    auto& mm = *p.get_main_module();

    auto evaluations = migraphx::find_all(
        migraphx::iterator_for(mm), [](auto ins) { return ins->name() == "eval_expr_from_shape"; });
    EXPECT(evaluations.size() == 1);
    if(evaluations.size() == 1)
    {
        EXPECT(evaluations.front()->inputs().size() == 1);
        EXPECT(evaluations.front()->inputs().front()->name() == "transpose");
    }

    p.compile(migraphx::make_target("ref"));
    std::vector<char> condition = {1, 0, 1, 0};
    migraphx::parameter_map params;
    params["condition"] = migraphx::argument{{migraphx::shape::bool_type, {4}}, condition.data()};
    auto result         = p.eval(params).back();
    EXPECT(result.get_shape().lens() == std::vector<std::size_t>{2, 1, 1, 1});
    EXPECT(result.to_vector<int64_t>() == std::vector<int64_t>{0, 2});
}
