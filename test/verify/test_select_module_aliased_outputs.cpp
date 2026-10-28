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

#include "verify_program.hpp"
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>

struct test_select_module_aliased_outputs : verify_program<test_select_module_aliased_outputs>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto create_submodule = [&](std::size_t batch_size, const std::string& name) {
            auto* submodule = p.create_module(name);
            auto input      = submodule->add_parameter(
                "data", migraphx::shape{migraphx::shape::float_type, {batch_size, 4}});
            auto small =
                submodule->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), input);
            auto large = submodule->add_instruction(
                migraphx::make_op("multibroadcast", {{"out_lens", {batch_size, 32}}}), small);
            submodule->add_return({small, small, large});
            return submodule;
        };
        auto* batch1 = create_submodule(1, "batch_1");
        auto* batch2 = create_submodule(2, "batch_2");
        auto* batch3 = create_submodule(3, "batch_3");
        auto* batch4 = create_submodule(4, "batch_4");

        auto* mm   = p.get_main_module();
        auto input = mm->add_parameter(
            "data", migraphx::shape{migraphx::shape::float_type, {{1, 4}, {4, 4}}});
        auto small_shape = migraphx::shape{migraphx::shape::float_type, {{1, 4}, {1, 1}}};
        auto large_shape = migraphx::shape{migraphx::shape::float_type, {{1, 4}, {32, 32}}};
        auto output_shape =
            migraphx::shape{std::vector<migraphx::shape>{small_shape, small_shape, large_shape}};
        auto selection = mm->add_instruction(
            migraphx::make_op("select_module",
                              {{"output_dyn_shapes", migraphx::to_value(output_shape)}}),
            {input},
            {batch1, batch2, batch3, batch4});
        auto output0 =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), selection);
        auto output1 =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), selection);
        auto output2 =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 2}}), selection);
        mm->add_return({output0, output1, output2});
        return p;
    }
};
