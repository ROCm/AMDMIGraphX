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
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/eliminate_concat.hpp>
#include <migraphx/gpu/concat_gpu_opt.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_op.hpp>
#include <test.hpp>

namespace {

migraphx::operation precompile(const migraphx::operation& op, const migraphx::shape& output)
{
    return migraphx::make_op(
        "gpu::precompile_op",
        {{"op", migraphx::to_value(op)}, {"output_shape", migraphx::to_value(output)}});
}

migraphx::instruction_ref add_alloc(migraphx::module& m, const migraphx::shape& s)
{
    return m.add_instruction(migraphx::make_op("allocate", {{"shape", migraphx::to_value(s)}}));
}

bool is_inner(migraphx::instruction_ref ins, const std::string& name)
{
    if(ins->name() != "gpu::precompile_op")
        return false;
    const auto inner =
        migraphx::from_value<migraphx::operation>(ins->get_operator().to_value().at("op"));
    return inner.name() == name;
}

} // namespace

// A channel-concat slice is not the packed layout a quant_convolution kernel was
// compiled for. Pinning that slice is the output-buffer mismatch.
TEST_CASE(quant_convolution_concat_keeps_native_layout)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape in{migraphx::shape::int8_type, {1, 4, 8, 8}};
    migraphx::shape weight{migraphx::shape::int8_type, {8, 4, 1, 1}};
    migraphx::shape conv_s{migraphx::shape::int32_type, {1, 8, 8, 8}};
    migraphx::shape cat_s{migraphx::shape::int32_type, {1, 16, 8, 8}};

    auto x  = mm->add_parameter("x", in);
    auto w  = mm->add_parameter("w", weight);
    auto y  = mm->add_parameter("y", in);
    auto v  = mm->add_parameter("v", weight);
    auto c1 = mm->add_instruction(precompile(migraphx::make_op("quant_convolution"), conv_s),
                                  x,
                                  w,
                                  add_alloc(*mm, conv_s));
    auto c2 = mm->add_instruction(precompile(migraphx::make_op("quant_convolution"), conv_s),
                                  y,
                                  v,
                                  add_alloc(*mm, conv_s));
    auto cat =
        mm->add_instruction(migraphx::make_op("gpu::precompile_op",
                                              {{"op", migraphx::to_value(migraphx::make_op(
                                                          "concat", {{"axis", 1}}))}}),
                            c1,
                            c2,
                            add_alloc(*mm, cat_s));
    mm->add_return({cat});

    migraphx::program original = p;
    migraphx::run_passes(
        p, {migraphx::eliminate_concat{migraphx::gpu::concat_gpu_optimization{}},
            migraphx::dead_code_elimination{}});
    EXPECT(p == original);
}

// The same slice is a legal output for a JIT pointwise kernel, which reads the
// buffer shape when it is compiled.
TEST_CASE(pointwise_concat_still_pins_slice_layout)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 4}};
    migraphx::shape cat_s{migraphx::shape::float_type, {2, 8}};
    auto x = mm->add_parameter("x", s);
    auto y = mm->add_parameter("y", s);
    auto p1 =
        mm->add_instruction(precompile(migraphx::make_op("pointwise"), s), x, add_alloc(*mm, s));
    auto p2 =
        mm->add_instruction(precompile(migraphx::make_op("pointwise"), s), y, add_alloc(*mm, s));
    auto cat =
        mm->add_instruction(migraphx::make_op("gpu::precompile_op",
                                              {{"op", migraphx::to_value(migraphx::make_op(
                                                          "concat", {{"axis", 1}}))}}),
                            p1,
                            p2,
                            add_alloc(*mm, cat_s));
    mm->add_return({cat});

    migraphx::run_passes(
        p, {migraphx::eliminate_concat{migraphx::gpu::concat_gpu_optimization{}},
            migraphx::dead_code_elimination{}});

    bool pinned = false;
    for(auto ins : migraphx::iterator_for(*p.get_main_module()))
    {
        EXPECT(not is_inner(ins, "concat"));
        if(not is_inner(ins, "pointwise"))
            continue;
        auto out = migraphx::from_value<migraphx::shape>(
            ins->get_operator().to_value().at("output_shape"));
        // Super-buffer {2, 8} has stride 8 on axis 0; the packed input stride is 4.
        if(out.strides().at(0) == 8)
            pinned = true;
    }
    EXPECT(pinned);
}

TEST_CASE(mlir_quant_convolution_output_shape_not_pinnable)
{
    if(not migraphx::has_op("gpu::mlir_op"))
        return;

    migraphx::shape conv_s{migraphx::shape::int32_type, {1, 8, 8, 8}};
    migraphx::shape slice_s{migraphx::shape::int32_type, {1, 8, 8, 8}, {1024, 64, 8, 1}};
    auto qconv = migraphx::make_op("quant_convolution");
    auto mlir  = migraphx::make_op(
        "gpu::mlir_op",
        {{"op", migraphx::to_value(qconv)}, {"tag", "standalone_quant_convolution"}});

    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto alloc = add_alloc(*mm, conv_s);
    auto ins   = mm->add_instruction(precompile(mlir, conv_s), alloc);
    migraphx::gpu::concat_gpu_optimization opt;
    EXPECT(not opt.supports_output_shape(ins, slice_s));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
