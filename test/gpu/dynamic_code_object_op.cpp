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
#include <migraphx/gpu/hip.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/sym.hpp>
#include <test.hpp>
#include <pointwise.hpp>

static bool module_has_dynamic_code_object(const migraphx::module& m)
{
    for(auto ins : migraphx::iterator_for(m))
    {
        if(ins->name() == "gpu::dynamic_code_object_op")
            return true;
    }
    return false;
}

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

TEST_CASE(dynamic_concat_gpu_lowering)
{
    migraphx::shape s0{migraphx::shape::float_type, {{2, 4, {2}}, {2, 3, {2}}}};
    migraphx::shape s1{migraphx::shape::float_type, {{3, 4, {4}}, {2, 3, {2}}}};
    migraphx::shape s2{migraphx::shape::float_type, {{1, 5, {3}}, {2, 3, {2}}}};
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("X", s0);
    auto y   = mm->add_parameter("Y", s1);
    auto z   = mm->add_parameter("Z", s2);
    mm->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), x, y, z);

    run_lowering(p);

    bool found = false;
    for(auto ins : iterator_for(*p.get_main_module()))
    {
        if(ins->name() == "gpu::dynamic_code_object_op")
        {
            found = true;
            break;
        }
    }
    EXPECT(found);
}

// Mimics KV-cache concat: dynamic past_key + static current along axis 2 (half_type).
TEST_CASE(dynamic_concat_kv_cache_axis2_eval)
{
    using migraphx::sym::var;
    auto psl = var("psl", {1, 64});
    using dd = migraphx::shape::dynamic_dimension;

    migraphx::shape past_shape{migraphx::shape::half_type, {dd{1, 1}, dd{5, 5}, dd{psl}, dd{64, 64}}};
    migraphx::shape current_shape{migraphx::shape::half_type, {1, 5, 1, 64}};
    migraphx::shape out_dyn_shape{migraphx::shape::half_type, {dd{1, 1}, dd{5, 5}, dd{psl + 1}, dd{64, 64}}};

    migraphx::program p;
    auto* mm = p.get_main_module();
    auto past_key =
        mm->add_parameter("past_key_values.0.key", past_shape);
    auto current_key = mm->add_literal(migraphx::generate_literal(current_shape));
    auto concat =
        mm->add_instruction(migraphx::make_op("concat", {{"axis", 2}}), past_key, current_key);
    mm->add_return({concat});

    p.compile(migraphx::make_target("gpu"));

    migraphx::shape past_static{migraphx::shape::half_type, {1, 5, 1, 64}};
    migraphx::shape out_static{migraphx::shape::half_type, {1, 5, 2, 64}};
    auto past_arg  = migraphx::gpu::to_gpu(migraphx::generate_argument(past_static));
    auto out_arg   = migraphx::gpu::allocate_gpu(out_static);

    migraphx::parameter_map params;
    params["past_key_values.0.key"] = past_arg;
    params["main:#output_0"]        = out_arg;

    auto results = p.eval(params);
    EXPECT(results.size() == 1);
    auto host_out = migraphx::gpu::from_gpu(results.front());
    EXPECT(host_out.get_shape().lens() == out_static.lens());
}

TEST_CASE(symbolic_concat_gpu_lowering)
{
    using migraphx::sym::var;
    auto n  = var("n", {2, 3});
    auto d0 = var("d0", {2, 4});
    auto d1 = var("d1", {3, 4});
    auto d2 = var("d2", {1, 5});
    using dd = migraphx::shape::dynamic_dimension;

    migraphx::shape s0{migraphx::shape::float_type, {dd{d0}, dd{n}}};
    migraphx::shape s1{migraphx::shape::float_type, {dd{d1}, dd{n}}};
    migraphx::shape s2{migraphx::shape::float_type, {dd{d2}, dd{n}}};
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("X", s0);
    auto y   = mm->add_parameter("Y", s1);
    auto z   = mm->add_parameter("Z", s2);
    mm->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), x, y, z);

    run_lowering(p);

    EXPECT(module_has_dynamic_code_object(*p.get_main_module()));
}

TEST_CASE(dynamic_greater_gpu_lowering)
{
    migraphx::shape s{migraphx::shape::float_type, {{2, 4}, {8, 16}}};
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", s);
    auto y   = mm->add_parameter("y", s);
    auto gr  = mm->add_instruction(migraphx::make_op("greater"), x, y);
    mm->add_return({gr});

    run_lowering(p);

    EXPECT(module_has_dynamic_code_object(*p.get_main_module()));
}

TEST_CASE(symbolic_greater_gpu_lowering)
{
    using migraphx::sym::var;
    auto n = var("n", {1, 4});
    using dd = migraphx::shape::dynamic_dimension;

    migraphx::shape sx{migraphx::shape::float_type, {dd{n}, dd{2, 8}}};
    migraphx::shape sy{migraphx::shape::float_type, {dd{n}, dd{2, 8}}};
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", sx);
    auto y   = mm->add_parameter("y", sy);
    auto gr  = mm->add_instruction(migraphx::make_op("greater"), x, y);
    mm->add_return({gr});

    run_lowering(p);

    EXPECT(module_has_dynamic_code_object(*p.get_main_module()));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
