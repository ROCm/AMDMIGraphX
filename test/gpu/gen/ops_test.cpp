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
#include <migraphx/shape.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/compile_options.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/gpu/gen/codegen.hpp>
#include <migraphx/gpu/gen/tiling.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/hip.hpp>
#include <test.hpp>

using migraphx::make_op;
using migraphx::shape;

TEST_CASE(pointwise_add_codegen)
{
    // Build a simple pointwise add module, compile and run via gen IR
    migraphx::module m;
    auto x   = m.add_parameter("x", {shape::float_type});
    auto y   = m.add_parameter("y", {shape::float_type});
    auto add = m.add_instruction(make_op("add"), x, y);
    m.add_return({add});

    // Verify codegen produces valid code
    auto code = migraphx::gpu::gen::generate_gen_function(m);
    EXPECT(migraphx::contains(code, "gen_func"));
    EXPECT(migraphx::contains(code, "__device__"));
}

TEST_CASE(gen_kernel_structure)
{
    migraphx::module m;
    auto x   = m.add_parameter("x", {shape::float_type});
    auto y   = m.add_parameter("y", {shape::float_type});
    auto add = m.add_instruction(make_op("add"), x, y);
    m.add_return({add});

    auto code = migraphx::gpu::gen::generate_gen_kernel(m, "test_kernel", 3);
    EXPECT(migraphx::contains(code, "test_kernel"));
    EXPECT(migraphx::contains(code, "make_index"));
    EXPECT(migraphx::contains(code, "make_tensors"));
    EXPECT(migraphx::contains(code, "gen_func"));
    EXPECT(migraphx::contains(code, "migraphx/kernels/gen.hpp"));
}

TEST_CASE(gen_ops_codegen)
{
    // Verify gen IR ops generate the correct device code
    migraphx::module m;
    auto x   = m.add_parameter("x", shape{shape::float_type, {64}});
    auto z   = m.add_parameter("z", shape{shape::float_type, {64}});
    auto gid = m.add_instruction(make_op("gpu::gen::global_id"));
    auto ld  = m.add_instruction(make_op("gpu::gen::vector_load", {{"size", 4}}), x, gid);
    auto st  = m.add_instruction(make_op("gpu::gen::vector_store", {{"size", 4}}), z, gid, ld);
    m.add_return({st});

    auto code = migraphx::gpu::gen::generate_gen_function(m);
    EXPECT(migraphx::contains(code, "idx.global"));
    EXPECT(migraphx::contains(code, "gen::vec_load<4>"));
    EXPECT(migraphx::contains(code, "gen::vec_store<4>"));
}

TEST_CASE(check_op_codegen)
{
    // Verify the check op generates MIGRAPHX_CHECK
    migraphx::module m;
    auto x     = m.add_parameter("x", {shape::float_type});
    auto y     = m.add_parameter("y", {shape::float_type});
    auto add   = m.add_instruction(make_op("add"), x, y);
    auto cond  = m.add_instruction(make_op("greater"), add, x);
    auto check = m.add_instruction(make_op("gpu::gen::check"), cond, add);
    m.add_return({check});

    auto code = migraphx::gpu::gen::generate_gen_function(m);
    EXPECT(migraphx::contains(code, "MIGRAPHX_CHECK"));
}

TEST_CASE(barrier_identity_shape)
{
    migraphx::module m;
    auto x   = m.add_parameter("x", shape{shape::float_type, {16}});
    auto bar = m.add_instruction(make_op("gpu::gen::barrier"), x);
    m.add_return({bar});

    EXPECT(bar->get_shape() == x->get_shape());
}

TEST_CASE(dpp_reduce_preserves_type)
{
    migraphx::module m;
    auto x   = m.add_parameter("x", {shape::float_type});
    auto red = m.add_instruction(make_op("gpu::gen::dpp_reduce", {{"op", "sum"}}), x);
    m.add_return({red});

    EXPECT(red->get_shape() == x->get_shape());
}

TEST_CASE(tile_region_output_shape)
{
    migraphx::module m;
    auto x    = m.add_parameter("x", shape{shape::float_type, {64, 128, 256}});
    auto wgid = m.add_instruction(make_op("gpu::gen::workgroup_id"));
    auto tile = m.add_instruction(
        make_op("gpu::gen::tile_region",
                {{"tile_dims", std::vector<std::size_t>{32, 64}}, {"axis", 1}}),
        x,
        wgid);
    m.add_return({tile});

    EXPECT(tile->get_shape().type() == shape::float_type);
    EXPECT(tile->get_shape().lens()[0] == 64);
    EXPECT(tile->get_shape().lens()[1] == 32);
    EXPECT(tile->get_shape().lens()[2] == 64);
}

TEST_CASE(reduce_waves_output_shape)
{
    migraphx::module m;
    auto x   = m.add_parameter("x", {shape::float_type});
    auto lds = m.add_instruction(
        make_op("gpu::gen::lds_allocate",
                {{"shape", migraphx::to_value(shape{shape::float_type, {8}})}}));
    auto red = m.add_instruction(make_op("gpu::gen::reduce_waves", {{"op", "sum"}}), x, lds);
    m.add_return({red});

    EXPECT(red->get_shape() == x->get_shape());
}

TEST_CASE(strided_load_output_shape)
{
    migraphx::module m;
    auto x  = m.add_parameter("x", shape{shape::float_type, {256}});
    auto id = m.add_instruction(make_op("gpu::gen::global_id"));
    auto ld = m.add_instruction(
        make_op("gpu::gen::strided_load", {{"size", 4}, {"stride", 64}}), x, id);
    m.add_return({ld});

    EXPECT(ld->get_shape().type() == shape::float_type);
    EXPECT(ld->get_shape().lens() == std::vector<std::size_t>{4});
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
