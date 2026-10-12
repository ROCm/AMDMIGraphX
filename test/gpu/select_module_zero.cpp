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
#include <algorithm>
#include <iterator>

TEST_CASE(select_module_zero_short_circuit_gpu)
{
    auto n = migraphx::sym::var("n", {0, 4});
    migraphx::shape output{migraphx::shape::float_type,
                           {migraphx::shape::dynamic_dimension{n},
                            migraphx::shape::dynamic_dimension{migraphx::sym::lit(2)}}};
    migraphx::program p;
    auto* mm    = p.get_main_module();
    auto input  = mm->add_parameter("data", output);
    auto select = mm->add_instruction(
        migraphx::make_op(
            "select_module",
            {{"output_dyn_shapes",
              migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{output}})},
             {"logical_output_dyn_shapes",
              migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{output}})},
             {"num_inputs", 1}}),
        {input},
        {});
    auto result = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), select);
    mm->add_return({result});

    auto target = migraphx::make_target("gpu");
    p.compile(target);
    migraphx::parameter_map params;
    auto parameter_shapes = p.get_parameter_shapes();
    std::transform(parameter_shapes.begin(),
                   parameter_shapes.end(),
                   std::inserter(params, params.end()),
                   [&](const auto& item) {
                       const auto& [name, shape] = item;
                       auto allocation =
                           name == "data" ? target.allocate({migraphx::shape::float_type, {0, 2}})
                                          : target.allocate(shape);
                       return std::make_pair(name, allocation);
                   });
    auto result_arg = p.eval(params).back();
    EXPECT(result_arg.get_shape() == migraphx::shape{migraphx::shape::float_type, {0, 2}});
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
