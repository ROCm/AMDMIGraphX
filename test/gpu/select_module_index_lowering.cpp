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
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/lowering.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <pointwise.hpp>
#include <test.hpp>

static void run_lowering(migraphx::module& m)
{
    auto ctx = migraphx::gpu::context{};
    migraphx::run_passes(m,
                         {migraphx::gpu::lowering{&ctx, false}, migraphx::dead_code_elimination{}});
}

// A device pointwise op produces the int64 slot. The select reads hip::load_scalar
// of that op's output, which is the index flash decoding lowers to.
TEST_CASE(select_module_index_lowering_device_index)
{
    migraphx::shape x_s{migraphx::shape::int64_type, {1}};
    migraphx::shape data_s{migraphx::shape::float_type, {2, 2}};
    migraphx::shape out_s{std::vector<migraphx::shape>{data_s}};
    auto precompile = migraphx::make_op(
        "gpu::precompile_op", {{"op", migraphx::to_value(migraphx::make_op("pointwise"))}});

    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", x_s);
        auto data  = mm->add_parameter("data", data_s);
        auto index = add_pointwise(p1, "slot", {x}, single_pointwise("abs"));
        auto* sub  = p1.create_module("sub");
        auto data0 = sub->add_parameter("data", data_s);
        sub->add_return({data0});
        auto smi =
            mm->add_instruction(migraphx::make_op("select_module_index"), {index, data}, {sub});
        mm->add_return({smi});
    }
    run_lowering(*p1.get_main_module());

    migraphx::program p2;
    {
        auto* mm   = p2.get_main_module();
        auto x     = mm->add_parameter("x", x_s);
        auto data  = mm->add_parameter("data", data_s);
        auto* slot = create_pointwise_module(p2, "slot", {x}, single_pointwise("abs"));
        auto* sub  = p2.create_module("sub");
        auto data0 = sub->add_parameter("data", data_s);
        sub->add_return({data0});
        auto idx_out = mm->add_instruction(
            migraphx::make_op("allocate", {{"shape", migraphx::to_value(x_s)}}));
        auto lowered = mm->add_instruction(precompile, {x, idx_out}, {slot});
        auto loaded  = mm->add_instruction(migraphx::make_op("hip::load_scalar"), lowered);
        auto output  = mm->add_instruction(
            migraphx::make_op("allocate", {{"shape", migraphx::to_value(out_s)}}));
        auto smi = mm->add_instruction(
            migraphx::make_op("select_module_index"), {loaded, data, output}, {sub});
        mm->add_return({smi});
    }
    EXPECT(p1 == p2);
}

// Each select gets its own hip::load_scalar on the index in lowering.
TEST_CASE(select_module_index_lowering_shared_index)
{
    migraphx::shape idx_s{migraphx::shape::int64_type, {1}};
    migraphx::shape data_s{migraphx::shape::float_type, {2, 2}};
    migraphx::shape out_s{std::vector<migraphx::shape>{data_s}};

    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto index = mm->add_parameter("index", idx_s);
        auto data  = mm->add_parameter("data", data_s);
        auto* sub  = p1.create_module("sub");
        auto data0 = sub->add_parameter("data", data_s);
        sub->add_return({data0});
        auto smi0 =
            mm->add_instruction(migraphx::make_op("select_module_index"), {index, data}, {sub});
        auto smi1 =
            mm->add_instruction(migraphx::make_op("select_module_index"), {index, data}, {sub});
        mm->add_return({smi0, smi1});
    }
    run_lowering(*p1.get_main_module());

    migraphx::program p2;
    {
        auto* mm   = p2.get_main_module();
        auto index = mm->add_parameter("index", idx_s);
        auto data  = mm->add_parameter("data", data_s);
        auto* sub  = p2.create_module("sub");
        auto data0 = sub->add_parameter("data", data_s);
        sub->add_return({data0});
        auto loaded0 = mm->add_instruction(migraphx::make_op("hip::load_scalar"), index);
        auto output0 = mm->add_instruction(
            migraphx::make_op("allocate", {{"shape", migraphx::to_value(out_s)}}));
        auto smi0 = mm->add_instruction(
            migraphx::make_op("select_module_index"), {loaded0, data, output0}, {sub});
        auto loaded1 = mm->add_instruction(migraphx::make_op("hip::load_scalar"), index);
        auto output1 = mm->add_instruction(
            migraphx::make_op("allocate", {{"shape", migraphx::to_value(out_s)}}));
        auto smi1 = mm->add_instruction(
            migraphx::make_op("select_module_index"), {loaded1, data, output1}, {sub});
        mm->add_return({smi0, smi1});
    }
    EXPECT(p1 == p2);
}

// Lowering always inserts hip::load_scalar on the index input.
TEST_CASE(select_module_index_lowering_existing_load)
{
    migraphx::shape idx_s{migraphx::shape::int64_type, {1}};
    migraphx::shape data_s{migraphx::shape::float_type, {2, 2}};
    migraphx::shape out_s{std::vector<migraphx::shape>{data_s}};

    migraphx::program p1;
    {
        auto* mm    = p1.get_main_module();
        auto index  = mm->add_parameter("index", idx_s);
        auto loaded = mm->add_instruction(migraphx::make_op("hip::load_scalar"), index);
        auto data   = mm->add_parameter("data", data_s);
        auto* sub   = p1.create_module("sub");
        auto data0  = sub->add_parameter("data", data_s);
        sub->add_return({data0});
        auto smi =
            mm->add_instruction(migraphx::make_op("select_module_index"), {loaded, data}, {sub});
        mm->add_return({smi});
    }
    run_lowering(*p1.get_main_module());

    migraphx::program p2;
    {
        auto* mm    = p2.get_main_module();
        auto index  = mm->add_parameter("index", idx_s);
        auto loaded = mm->add_instruction(migraphx::make_op("hip::load_scalar"), index);
        auto data   = mm->add_parameter("data", data_s);
        auto* sub   = p2.create_module("sub");
        auto data0  = sub->add_parameter("data", data_s);
        sub->add_return({data0});
        auto loaded2 = mm->add_instruction(migraphx::make_op("hip::load_scalar"), loaded);
        auto output  = mm->add_instruction(
            migraphx::make_op("allocate", {{"shape", migraphx::to_value(out_s)}}));
        auto smi = mm->add_instruction(
            migraphx::make_op("select_module_index"), {loaded2, data, output}, {sub});
        mm->add_return({smi});
    }
    EXPECT(p1 == p2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
