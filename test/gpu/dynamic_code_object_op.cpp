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

#include <migraphx/gpu/lowering.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <test.hpp>
#include <pointwise.hpp>

static void run_lowering(migraphx::program& p, bool offload_copy = false)
{
    auto ctx = migraphx::gpu::context{};
    migraphx::run_passes(*p.get_main_module(), {migraphx::gpu::lowering{&ctx, offload_copy}});
}

TEST_CASE(dynamic_code_object_op)
{
    migraphx::shape s{migraphx::shape::float_type, {{1, 3}, {2, 4}, {6, 6}}};
    migraphx::program p1;
    auto* mm = p1.get_main_module();
    auto a   = mm->add_parameter("a", s);
    auto b   = mm->add_parameter("b", s);

    auto pw               = add_pointwise(p1, "main:pointwise0", {a, b}, single_pointwise("add"));
    auto pw_module_inputs = pw->module_inputs();

    mm->add_return({pw});

    run_lowering(p1);

    bool found = false;
    for(auto ins : iterator_for(*p1.get_main_module()))
    {
        if(ins->name() == "gpu::dynamic_code_object_op")
        {
            found = true;
            EXPECT(ins->module_inputs() == pw_module_inputs);
        }
    }
    EXPECT(found);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
