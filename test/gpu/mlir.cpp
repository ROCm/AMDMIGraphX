/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/gpu/mlir.hpp>
#include <migraphx/gpu/target.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/write_literals.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/module.hpp>
#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/verify_args.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/functional.hpp>
#include <test.hpp>

struct mlir_gpu_target : migraphx::gpu::target
{
    std::string name() const { return "mlir"; }
    std::vector<migraphx::pass> get_passes(migraphx::context& gctx,
                                           const migraphx::compile_options&) const
    {
        auto& ctx = migraphx::any_cast<migraphx::gpu::context>(gctx);
        return {migraphx::gpu::write_literals{&ctx}};
    }
};

std::string encode(const std::string& s)
{
    std::stringstream ss;
    bool prespace = false;
    for(auto c : s)
    {
        if(std::isspace(c) != 0)
        {
            if(not prespace)
                ss << "  ";
            prespace = true;
        }
        else if(std::isprint(c) != 0)
        {
            ss << c;
            prespace = false;
        }
    }
    return migraphx::trim(ss.str());
}

migraphx::program create_program_from_mlir(const migraphx::module& mmlir)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto names = mmlir.get_parameter_names();
    std::vector<migraphx::instruction_ref> inputs;
    std::transform(names.begin(), names.end(), std::back_inserter(inputs), [&](const auto& name) {
        return mm->add_parameter(name, mmlir.get_parameter_shape(name));
    });
    std::sort(inputs.begin(), inputs.end(), migraphx::by(std::less<>{}, [](auto ins) {
                  return to_string(ins->get_operator());
              }));
    inputs.push_back(mm->add_parameter("output", mmlir.get_output_shapes().front()));

    migraphx::gpu::context ctx;
    migraphx::gpu::insert_mlir(*mm, mm->end(), compile_mlir(ctx, mmlir, inputs, {}), inputs);
    return p;
}

migraphx::parameter_map generate_params(const migraphx::program& p)
{
    migraphx::parameter_map m;
    std::size_t i = 0;
    for(auto&& x : p.get_parameter_shapes())
    {
        // m[x.first] = migraphx::fill_argument(x.second, 1);
        m[x.first] = migraphx::generate_argument(x.second, i++);
    }
    return m;
}

migraphx::argument run_gpu(migraphx::program p, const migraphx::parameter_map& inputs)
{
    mlir_gpu_target t;
    p.compile(t);
    migraphx::parameter_map m;
    for(auto&& input : inputs)
    {
        m[input.first] = t.copy_to(input.second);
    }
    for(auto&& x : p.get_parameter_shapes())
    {
        if(m.count(x.first) == 0)
        {
            m[x.first] = t.allocate(x.second);
        }
    }
    return t.copy_from(p.eval(m).front());
}

migraphx::argument run_ref(migraphx::program p, const migraphx::parameter_map& inputs)
{
    p.compile(migraphx::make_target("ref"));
    return p.eval(inputs).front();
}

bool verify_mlir(const migraphx::module& mmlir)
{
    migraphx::program ref;
    ref.get_main_module()->insert_instructions(ref.get_main_module()->end(), &mmlir);

    auto inputs = generate_params(ref);

    auto mlir = create_program_from_mlir(mmlir);
    return migraphx::verify_args(
        "mlir", run_gpu(mlir, inputs), migraphx::verify::expected{run_ref(ref, inputs)});
}

TEST_CASE(conv)
{
    const std::string mlir_output = R"__migraphx__(
module {
  func.func @mlir_convolution(%arg0: tensor<2x8x3x3xf32>, %arg1: tensor<1x8x4x4xf32>) -> tensor<1x2x2x2xf32> attributes {arch = "", kernel = "mixr", num_cu = 0 : i64} {
    %0 = migraphx.convolution(%arg1, %arg0) {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : (tensor<1x8x4x4xf32>, tensor<2x8x3x3xf32>) -> tensor<1x2x2x2xf32>
    return %0 : tensor<1x2x2x2xf32>
  }
}
)__migraphx__";
    migraphx::module m;
    auto x    = m.add_parameter("x", {migraphx::shape::float_type, {1, 8, 4, 4}});
    auto w    = m.add_parameter("w", {migraphx::shape::float_type, {2, 8, 3, 3}});
    auto conv = m.add_instruction(migraphx::make_op("convolution"), x, w);
    m.add_return({conv});
    auto s = migraphx::gpu::dump_mlir(m);
    // Skip test if MLIR is not enabled
    if(s.empty())
        return;
    CHECK(encode(s) == encode(mlir_output));
    EXPECT(verify_mlir(m));
}

TEST_CASE(conv_add_relu)
{
    const std::string mlir_output = R"__migraphx__(
module {
  func.func @mlir_convolution_add_relu(%arg0: tensor<1x2x2x2xf32>, %arg1: tensor<2x8x3x3xf32>, %arg2: tensor<1x8x4x4xf32>) -> tensor<1x2x2x2xf32> attributes {arch = "", kernel = "mixr", num_cu = 0 : i64} {
    %0 = migraphx.convolution(%arg2, %arg1) {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : (tensor<1x8x4x4xf32>, tensor<2x8x3x3xf32>) -> tensor<1x2x2x2xf32>
    %1 = migraphx.add(%0, %arg0) : (tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
    %2 = migraphx.relu(%1) : (tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
    return %2 : tensor<1x2x2x2xf32>
  }
}
)__migraphx__";
    migraphx::module m;
    auto x    = m.add_parameter("x", {migraphx::shape::float_type, {1, 8, 4, 4}});
    auto w    = m.add_parameter("w", {migraphx::shape::float_type, {2, 8, 3, 3}});
    auto b    = m.add_parameter("b", {migraphx::shape::float_type, {1, 2, 2, 2}});
    auto conv = m.add_instruction(migraphx::make_op("convolution"), x, w);
    auto add  = m.add_instruction(migraphx::make_op("add"), conv, b);
    auto relu = m.add_instruction(migraphx::make_op("relu"), add);
    m.add_return({relu});
    auto s = migraphx::gpu::dump_mlir(m);
    // Skip test if MLIR is not enabled
    if(s.empty())
        return;
    CHECK(encode(s) == encode(mlir_output));
    EXPECT(verify_mlir(m));
}

TEST_CASE(quant_dot_add)
{
    const std::string mlir_output = R"__migraphx__(
module {
  func.func @mlir_quant_dot_add(%arg0: tensor<1x5x4xi8>, %arg1: tensor<1x4x3xi8>, %arg2: tensor<1x5x3xi32>) -> tensor<1x5x3xi32> attributes {arch = "", kernel = "mixr", num_cu = 0 : i64} {
    %0 = migraphx.quant_dot(%arg0, %arg1) : (tensor<1x5x4xi8>, tensor<1x4x3xi8>) -> tensor<1x5x3xi32>
    %1 = migraphx.add(%0, %arg2) : (tensor<1x5x3xi32>, tensor<1x5x3xi32>) -> tensor<1x5x3xi32>
    return %1 : tensor<1x5x3xi32>
  }
}
)__migraphx__";
    migraphx::module m;
    auto arg0 = m.add_parameter("arg0", {migraphx::shape::int8_type, {1, 5, 4}});
    auto arg1 = m.add_parameter("arg1", {migraphx::shape::int8_type, {1, 4, 3}});
    auto arg2 = m.add_parameter("arg2", {migraphx::shape::int32_type, {1, 5, 3}});
    auto conv = m.add_instruction(migraphx::make_op("quant_dot"), arg0, arg1);
    auto add  = m.add_instruction(migraphx::make_op("add"), conv, arg2);
    m.add_return({add});

    auto s = migraphx::gpu::dump_mlir(m);
    // Skip test if MLIR is not enabled
    if(s.empty())
        return;
    CHECK(encode(s) == encode(mlir_output));
    EXPECT(verify_mlir(m));
}

TEST_CASE(dot_add)
{
    const std::string mlir_output = R"__migraphx__(
module {
  func.func @mlir_dot_add(%arg0: tensor<1x5x4xf32>, %arg1: tensor<1x4x3xf32>, %arg2: tensor<1x5x3xf32>) -> tensor<1x5x3xf32> attributes {arch = "", kernel = "mixr", num_cu = 0 : i64} {
    %0 = migraphx.dot(%arg0, %arg1) : (tensor<1x5x4xf32>, tensor<1x4x3xf32>) -> tensor<1x5x3xf32>
    %1 = migraphx.add(%0, %arg2) : (tensor<1x5x3xf32>, tensor<1x5x3xf32>) -> tensor<1x5x3xf32>
    return %1 : tensor<1x5x3xf32>
  }
}
)__migraphx__";
    migraphx::module m;
    auto arg0 = m.add_parameter("arg0", {migraphx::shape::float_type, {1, 5, 4}});
    auto arg1 = m.add_parameter("arg1", {migraphx::shape::float_type, {1, 4, 3}});
    auto arg2 = m.add_parameter("arg2", {migraphx::shape::float_type, {1, 5, 3}});
    auto conv = m.add_instruction(migraphx::make_op("dot"), arg0, arg1);
    auto add  = m.add_instruction(migraphx::make_op("add"), conv, arg2);
    m.add_return({add});
    auto s = migraphx::gpu::dump_mlir(m);
    // Skip test if MLIR is not enabled
    if(s.empty())
        return;
    CHECK(encode(s) == encode(mlir_output));
    EXPECT(verify_mlir(m));
}

TEST_CASE(conv_int8_dequantize_quantize)
{
    const std::string mlir_output = R"__migraphx__(
module {
  func.func @mlir_quant_convolution_dequantizelinear_quantizelinear(%arg0: tensor<2x8x3x3xi8>, %arg1: tensor<1x8x4x4xi8>, %arg2: tensor<1x2x2x2xf32>, %arg3: tensor<1x2x2x2xi32>) -> tensor<1x2x2x2xi32> attributes {arch = "", kernel = "mixr", num_cu = 0 : i64} {
      %0 = migraphx.quant_convolution(%arg1, %arg0) {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : (tensor<1x8x4x4xi8>, tensor<2x8x3x3xi8>) -> tensor<1x2x2x2xi32>
      %1 = migraphx.dequantizelinear(%0, %arg2, %arg3) : (tensor<1x2x2x2xi32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xi32>) -> tensor<1x2x2x2xf32>
      %2 = migraphx.quantizelinear(%1, %arg2, %arg3) : (tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xi32>) -> tensor<1x2x2x2xi32>
      return %2 : tensor<1x2x2x2xi32>
    }
}
)__migraphx__";

    migraphx::module m;
    auto x    = m.add_parameter("x", {migraphx::shape::int8_type, {1, 8, 4, 4}});
    auto w    = m.add_parameter("w", {migraphx::shape::int8_type, {2, 8, 3, 3}});
    auto conv = m.add_instruction(migraphx::make_op("quant_convolution"), x, w);
    migraphx::shape ss{migraphx::shape::float_type, {1, 2, 2, 2}};
    migraphx::shape sz{migraphx::shape::int32_type, {1, 2, 2, 2}};
    auto input2  = m.add_parameter("x_scale", ss);
    auto input3  = m.add_parameter("x_zero_point", sz);
    auto dequant = m.add_instruction(migraphx::make_op("dequantizelinear"), conv, input2, input3);
    auto r       = m.add_instruction(migraphx::make_op("quantizelinear"), dequant, input2, input3);

    m.add_return({r});
    auto s = migraphx::gpu::dump_mlir(m);
    // Skip test if MLIR is not enabled
    if(s.empty())
        return;
    CHECK(encode(s) == encode(mlir_output));
    EXPECT(verify_mlir(m));
}

TEST_CASE(dot_convert)
{
    const std::string mlir_output = R"__migraphx__(
module {
  func.func @mlir_dot_convert(%arg0: tensor<1x5x4xf32>, %arg1: tensor<1x4x3xf32>) -> tensor<1x5x3xf16> attributes {arch = "", kernel = "mixr", num_cu = 0 : i64} {
    %0 = migraphx.dot(%arg0, %arg1) : (tensor<1x5x4xf32>, tensor<1x4x3xf32>) -> tensor<1x5x3xf32>
    %1 = migraphx.convert(%0) {target_type  =  1  :  i64} : (tensor<1x5x3xf32>) -> tensor<1x5x3xf16>
    return %1 : tensor<1x5x3xf16>
  }
}
)__migraphx__";
    migraphx::module m;
    auto arg0  = m.add_parameter("arg0", {migraphx::shape::float_type, {1, 5, 4}});
    auto arg1  = m.add_parameter("arg1", {migraphx::shape::float_type, {1, 4, 3}});
    auto dot   = m.add_instruction(migraphx::make_op("dot"), arg0, arg1);
    auto trunc = m.add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), dot);
    m.add_return({trunc});
    auto s = migraphx::gpu::dump_mlir(m);
    // Skip test if MLIR is not enabled
    if(s.empty())
        return;
    CHECK(encode(s) == encode(mlir_output));
    EXPECT(verify_mlir(m));
}

TEST_CASE(dot_where)
{
    const std::string mlir_output = R"__migraphx__(
module {
  func.func @mlir_dot_where(%arg0: tensor<1x5x4xf32>, %arg1: tensor<1x4x3xf32>, %arg2: tensor<1x5x3xi8>, %arg3: tensor<1x5x3xf32>) -> tensor<1x5x3xf32> attributes {arch = "", kernel = "mixr", num_cu = 0 : i64} {
    %0 = migraphx.dot(%arg0, %arg1) : (tensor<1x5x4xf32>, tensor<1x4x3xf32>) -> tensor<1x5x3xf32>
    %1 = migraphx.where(%arg2, %0, %arg3) : (tensor<1x5x3xi8>, tensor<1x5x3xf32>, tensor<1x5x3xf32>) -> tensor<1x5x3xf32>
    return %1 : tensor<1x5x3xf32>
  }
}
)__migraphx__";
    migraphx::module m;
    auto arg0  = m.add_parameter("arg0", {migraphx::shape::float_type, {1, 5, 4}});
    auto arg1  = m.add_parameter("arg1", {migraphx::shape::float_type, {1, 4, 3}});
    auto arg2  = m.add_parameter("arg2", {migraphx::shape::bool_type, {1, 5, 3}});
    auto arg3  = m.add_parameter("arg3", {migraphx::shape::float_type, {1, 5, 3}});
    auto dot   = m.add_instruction(migraphx::make_op("dot"), arg0, arg1);
    auto where = m.add_instruction(migraphx::make_op("where"), arg2, dot, arg3);
    m.add_return({where});
    auto s = migraphx::gpu::dump_mlir(m);
    // Skip test if MLIR is not enabled
    if(s.empty())
        return;
    CHECK(encode(s) == encode(mlir_output));
    EXPECT(verify_mlir(m));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
