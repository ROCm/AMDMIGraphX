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

// dimensions_of lowers to hip::copy_to_gpu of the host op. The select reads
// hip::load_scalar of that copy, placed right after it, so the copy's allocate stays live.
TEST_CASE(select_module_index_lowering_direct_index)
{
    migraphx::shape in_s{migraphx::shape::float_type, {{1, 8}, {4, 4}}};
    migraphx::shape data_s{migraphx::shape::float_type, {2, 2}};
    migraphx::shape idx_s{migraphx::shape::int64_type, {1}};
    migraphx::shape out_s{std::vector<migraphx::shape>{data_s}};
    auto dims = migraphx::make_op("dimensions_of", {{"end", 1}});

    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", in_s);
        auto data  = mm->add_parameter("data", data_s);
        auto* sub  = p1.create_module("sub");
        auto data0 = sub->add_parameter("data", data_s);
        sub->add_return({data0});
        auto index = mm->add_instruction(dims, x);
        auto smi =
            mm->add_instruction(migraphx::make_op("select_module_index"), {index, data}, {sub});
        mm->add_return({smi});
    }
    run_lowering(*p1.get_main_module());

    migraphx::program p2;
    {
        auto* mm   = p2.get_main_module();
        auto x     = mm->add_parameter("x", in_s);
        auto data  = mm->add_parameter("data", data_s);
        auto* sub  = p2.create_module("sub");
        auto data0 = sub->add_parameter("data", data_s);
        sub->add_return({data0});
        auto idx_out = mm->add_instruction(
            migraphx::make_op("allocate", {{"shape", migraphx::to_value(idx_s)}}));
        auto sync   = mm->add_instruction(migraphx::make_op("hip::sync_stream"), x);
        auto host   = mm->add_instruction(dims, sync);
        auto gpu    = mm->add_instruction(migraphx::make_op("hip::copy_to_gpu"), host, idx_out);
        auto loaded = mm->add_instruction(migraphx::make_op("hip::load_scalar"), gpu);
        auto output = mm->add_instruction(
            migraphx::make_op("allocate", {{"shape", migraphx::to_value(out_s)}}));
        auto smi = mm->add_instruction(
            migraphx::make_op("select_module_index"), {loaded, data, output}, {sub});
        mm->add_return({smi});
    }
    EXPECT(p1 == p2);
}

// A squeeze of the device copy is still one element. The select reads
// hip::load_scalar of that squeeze, placed right after it.
TEST_CASE(select_module_index_lowering_squeezed_index)
{
    migraphx::shape in_s{migraphx::shape::float_type, {{1, 8}, {4, 4}}};
    migraphx::shape data_s{migraphx::shape::float_type, {2, 2}};
    migraphx::shape idx_s{migraphx::shape::int64_type, {1}};
    migraphx::shape out_s{std::vector<migraphx::shape>{data_s}};
    auto dims = migraphx::make_op("dimensions_of", {{"end", 1}});

    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", in_s);
        auto data  = mm->add_parameter("data", data_s);
        auto* sub  = p1.create_module("sub");
        auto data0 = sub->add_parameter("data", data_s);
        sub->add_return({data0});
        auto index    = mm->add_instruction(dims, x);
        auto squeezed = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), index);
        auto smi =
            mm->add_instruction(migraphx::make_op("select_module_index"), {squeezed, data}, {sub});
        mm->add_return({smi});
    }
    run_lowering(*p1.get_main_module());

    migraphx::program p2;
    {
        auto* mm   = p2.get_main_module();
        auto x     = mm->add_parameter("x", in_s);
        auto data  = mm->add_parameter("data", data_s);
        auto* sub  = p2.create_module("sub");
        auto data0 = sub->add_parameter("data", data_s);
        sub->add_return({data0});
        auto idx_out = mm->add_instruction(
            migraphx::make_op("allocate", {{"shape", migraphx::to_value(idx_s)}}));
        auto sync     = mm->add_instruction(migraphx::make_op("hip::sync_stream"), x);
        auto host     = mm->add_instruction(dims, sync);
        auto gpu      = mm->add_instruction(migraphx::make_op("hip::copy_to_gpu"), host, idx_out);
        auto squeezed = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), gpu);
        auto loaded   = mm->add_instruction(migraphx::make_op("hip::load_scalar"), squeezed);
        auto output   = mm->add_instruction(
            migraphx::make_op("allocate", {{"shape", migraphx::to_value(out_s)}}));
        auto smi = mm->add_instruction(
            migraphx::make_op("select_module_index"), {loaded, data, output}, {sub});
        mm->add_return({smi});
    }
    EXPECT(p1 == p2);
}

// A reshape of the device copy is still one element. The select reads
// hip::load_scalar of that reshape, placed right after it.
TEST_CASE(select_module_index_lowering_reshaped_index)
{
    migraphx::shape in_s{migraphx::shape::float_type, {{1, 8}, {4, 4}}};
    migraphx::shape data_s{migraphx::shape::float_type, {2, 2}};
    migraphx::shape idx_s{migraphx::shape::int64_type, {1}};
    migraphx::shape out_s{std::vector<migraphx::shape>{data_s}};
    auto dims = migraphx::make_op("dimensions_of", {{"end", 1}});

    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", in_s);
        auto data  = mm->add_parameter("data", data_s);
        auto* sub  = p1.create_module("sub");
        auto data0 = sub->add_parameter("data", data_s);
        sub->add_return({data0});
        auto index    = mm->add_instruction(dims, x);
        auto reshaped = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {1}}}), index);
        auto smi =
            mm->add_instruction(migraphx::make_op("select_module_index"), {reshaped, data}, {sub});
        mm->add_return({smi});
    }
    run_lowering(*p1.get_main_module());

    migraphx::program p2;
    {
        auto* mm   = p2.get_main_module();
        auto x     = mm->add_parameter("x", in_s);
        auto data  = mm->add_parameter("data", data_s);
        auto* sub  = p2.create_module("sub");
        auto data0 = sub->add_parameter("data", data_s);
        sub->add_return({data0});
        auto idx_out = mm->add_instruction(
            migraphx::make_op("allocate", {{"shape", migraphx::to_value(idx_s)}}));
        auto sync     = mm->add_instruction(migraphx::make_op("hip::sync_stream"), x);
        auto host     = mm->add_instruction(dims, sync);
        auto gpu      = mm->add_instruction(migraphx::make_op("hip::copy_to_gpu"), host, idx_out);
        auto reshaped = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {1}}}), gpu);
        auto loaded   = mm->add_instruction(migraphx::make_op("hip::load_scalar"), reshaped);
        auto output   = mm->add_instruction(
            migraphx::make_op("allocate", {{"shape", migraphx::to_value(out_s)}}));
        auto smi = mm->add_instruction(
            migraphx::make_op("select_module_index"), {loaded, data, output}, {sub});
        mm->add_return({smi});
    }
    EXPECT(p1 == p2);
}

// offload_copy is disabled, so a parameter index stays a parameter.
// hip::load_scalar is inserted right after it.
TEST_CASE(select_module_index_lowering_parameter_index)
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
        auto smi =
            mm->add_instruction(migraphx::make_op("select_module_index"), {index, data}, {sub});
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
        auto output = mm->add_instruction(
            migraphx::make_op("allocate", {{"shape", migraphx::to_value(out_s)}}));
        auto smi = mm->add_instruction(
            migraphx::make_op("select_module_index"), {loaded, data, output}, {sub});
        mm->add_return({smi});
    }
    EXPECT(p1 == p2);
}

// Two selects of one index share a single hip::load_scalar, placed right after
// that index.
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
        auto* mm    = p2.get_main_module();
        auto index  = mm->add_parameter("index", idx_s);
        auto loaded = mm->add_instruction(migraphx::make_op("hip::load_scalar"), index);
        auto data   = mm->add_parameter("data", data_s);
        auto* sub   = p2.create_module("sub");
        auto data0  = sub->add_parameter("data", data_s);
        sub->add_return({data0});
        auto output0 = mm->add_instruction(
            migraphx::make_op("allocate", {{"shape", migraphx::to_value(out_s)}}));
        auto smi0 = mm->add_instruction(
            migraphx::make_op("select_module_index"), {loaded, data, output0}, {sub});
        auto output1 = mm->add_instruction(
            migraphx::make_op("allocate", {{"shape", migraphx::to_value(out_s)}}));
        auto smi1 = mm->add_instruction(
            migraphx::make_op("select_module_index"), {loaded, data, output1}, {sub});
        mm->add_return({smi0, smi1});
    }
    EXPECT(p1 == p2);
}

// An index that is already hip::load_scalar is left in place, with no second load.
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
        auto output = mm->add_instruction(
            migraphx::make_op("allocate", {{"shape", migraphx::to_value(out_s)}}));
        auto smi = mm->add_instruction(
            migraphx::make_op("select_module_index"), {loaded, data, output}, {sub});
        mm->add_return({smi});
    }
    EXPECT(p1 == p2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
