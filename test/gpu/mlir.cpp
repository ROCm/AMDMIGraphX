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
#include <migraphx/gpu/device_name.hpp>
#include <migraphx/gpu/mlir.hpp>
#include <migraphx/gpu/target.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/write_literals.hpp>
#include <migraphx/gpu/prepare_mlir.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/env.hpp>
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

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_MLIR_ENABLE_SPLITK);

struct mlir_gpu_target : migraphx::gpu::target
{
    std::string name() const { return "mlir"; }
    std::vector<migraphx::pass> get_passes(migraphx::context&,
                                           const migraphx::compile_options&) const
    {
        return {migraphx::gpu::write_literals{}};
    }
};

static std::string encode(const std::string& s)
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

static migraphx::module create_mlir_submodule(const migraphx::module& mmlir)
{
    migraphx::module m;
    std::unordered_map<migraphx::instruction_ref, migraphx::instruction_ref> map_ins;
    auto params = mmlir.get_parameter_names();
    for(const auto& name : params)
    {
        auto param     = mmlir.get_parameter(name);
        map_ins[param] = m.add_parameter(name, param->get_shape().as_standard());
    }
    auto y = m.add_instructions(&mmlir, &map_ins);
    m.add_return(y);
    return m;
}

static migraphx::program create_program_from_mlir(const migraphx::module& mmlir)
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
    // A multi-output module writes into a single tuple-shaped output argument
    auto out_shapes = mmlir.get_output_shapes();
    auto out_shape =
        out_shapes.size() == 1 ? out_shapes.front() : migraphx::shape{out_shapes};
    inputs.push_back(mm->add_parameter("output", out_shape));

    migraphx::gpu::context ctx;
    auto shapes = to_shapes(inputs);
    // compile_mlir requires a tuning solution (perfConfig) for the backend pipeline
    auto tc = get_tuning_config_mlir(ctx, create_mlir_submodule(mmlir), shapes, false);
    migraphx::gpu::mlir_code_object mco =
        compile_mlir(ctx, create_mlir_submodule(mmlir), shapes, tc.solutions.front());
    migraphx::gpu::insert_mlir(*mm, mm->end(), mco.cop, inputs);
    return p;
}

static migraphx::parameter_map generate_params(const migraphx::program& p)
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

// Flatten a tuple argument into its sub-arguments so multi-output results
// compare one-to-one with the reference outputs
static std::vector<migraphx::argument> flatten_arguments(const std::vector<migraphx::argument>& args)
{
    std::vector<migraphx::argument> result;
    for(const auto& arg : args)
    {
        auto sub = arg.get_sub_objects();
        if(sub.empty())
            result.push_back(arg);
        else
            result.insert(result.end(), sub.begin(), sub.end());
    }
    return result;
}

static std::vector<migraphx::argument> run_gpu(migraphx::program p,
                                               const migraphx::parameter_map& inputs)
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
    auto results = p.eval(m);
    std::vector<migraphx::argument> outputs;
    std::transform(results.begin(),
                   results.end(),
                   std::back_inserter(outputs),
                   [&](const auto& result) { return t.copy_from(result); });
    return flatten_arguments(outputs);
}

static std::vector<migraphx::argument> run_ref(migraphx::program p,
                                               const migraphx::parameter_map& inputs)
{
    p.compile(migraphx::make_target("ref"));
    return flatten_arguments(p.eval(inputs));
}

static bool verify_mlir(const migraphx::module& mmlir, const migraphx::parameter_map& fixed = {})
{
    migraphx::program ref;
    auto* rm   = ref.get_main_module();
    auto outs  = rm->insert_instructions(rm->end(), &mmlir);
    rm->add_return(outs);

    auto inputs = generate_params(ref);
    for(const auto& input : fixed)
        inputs[input.first] = input.second;

    auto mlir     = create_program_from_mlir(mmlir);
    auto results  = run_gpu(mlir, inputs);
    auto expected = run_ref(ref, inputs);
    return results.size() == expected.size() and
           std::equal(results.begin(),
                      results.end(),
                      expected.begin(),
                      [](const auto& result, const auto& gold) {
                          return migraphx::verify_args_with_tolerance(
                              "mlir", result, migraphx::verify::expected{gold});
                      });
}

static std::string get_attrs()
{
    if(migraphx::enabled(MIGRAPHX_MLIR_ENABLE_SPLITK{}))
    {
        return R"({rock.arch = "", rock.enable_splitk_for_tuning, rock.kernel = "mixr", rock.num_chiplets = 0 : i64, rock.num_cu = 0 : i64})";
    }
    return R"({rock.arch = "", rock.kernel = "mixr", rock.num_chiplets = 0 : i64, rock.num_cu = 0 : i64})";
}

TEST_CASE(conv)
{
    std::string mlir_output = R"__migraphx__(
module {
  func.func @mlir_convolution(%arg0: !migraphx.shaped<2x8x3x3xf32, 72x9x3x1>, %arg1: !migraphx.shaped<1x8x4x4xf32, 128x16x4x1>) -> !migraphx.shaped<1x2x2x2xf32, 8x4x2x1> attributes ${attrs} {
    %0 = migraphx.convolution %arg1, %arg0 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : <1x8x4x4xf32, 128x16x4x1>, <2x8x3x3xf32, 72x9x3x1> -> <1x2x2x2xf32, 8x4x2x1>
    return %0 : !migraphx.shaped<1x2x2x2xf32, 8x4x2x1>
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
    auto mlir_output_with_attrs =
        migraphx::interpolate_string(mlir_output, {{"attrs", get_attrs()}});
    CHECK(encode(s) == encode(mlir_output_with_attrs));
    EXPECT(verify_mlir(m));
}

TEST_CASE(conv_nhwc)
{
    std::string mlir_output = R"__migraphx__(
module {
  func.func @mlir_convolution(%arg0: !migraphx.shaped<2x8x3x3xf32, 72x1x24x8>, %arg1: !migraphx.shaped<1x8x4x4xf32, 128x1x32x8>) -> !migraphx.shaped<1x2x2x2xf32, 8x1x4x2> attributes ${attrs} {
    %0 = migraphx.convolution %arg1, %arg0 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : <1x8x4x4xf32, 128x1x32x8>, <2x8x3x3xf32, 72x1x24x8> -> <1x2x2x2xf32, 8x1x4x2>
    return %0 : !migraphx.shaped<1x2x2x2xf32, 8x1x4x2>
  }
}
)__migraphx__";
    migraphx::module m;
    auto x    = m.add_parameter("x", {migraphx::shape::float_type, {1, 8, 4, 4}, {128, 1, 32, 8}});
    auto w    = m.add_parameter("w", {migraphx::shape::float_type, {2, 8, 3, 3}, {72, 1, 24, 8}});
    auto conv = m.add_instruction(migraphx::make_op("convolution"), x, w);
    m.add_return({conv});
    auto s = migraphx::gpu::dump_mlir(m);
    // Skip test if MLIR is not enabled
    if(s.empty())
        return;
    auto mlir_output_with_attrs =
        migraphx::interpolate_string(mlir_output, {{"attrs", get_attrs()}});
    CHECK(encode(s) == encode(mlir_output_with_attrs));
    EXPECT(verify_mlir(m));
}

TEST_CASE(conv_add_relu)
{
    std::string mlir_output = R"__migraphx__(
module {
  func.func @mlir_convolution_add_relu(%arg0: !migraphx.shaped<1x2x2x2xf32, 8x4x2x1>, %arg1: !migraphx.shaped<2x8x3x3xf32, 72x9x3x1>, %arg2: !migraphx.shaped<1x8x4x4xf32, 128x16x4x1>) -> !migraphx.shaped<1x2x2x2xf32, 8x4x2x1> attributes ${attrs} {
    %0 = migraphx.convolution %arg2, %arg1 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : <1x8x4x4xf32, 128x16x4x1>, <2x8x3x3xf32, 72x9x3x1> -> <1x2x2x2xf32, 8x4x2x1>
    %1 = migraphx.add %0, %arg0 : <1x2x2x2xf32, 8x4x2x1>, <1x2x2x2xf32, 8x4x2x1> -> <1x2x2x2xf32, 8x4x2x1>
    %2 = migraphx.relu %1 : <1x2x2x2xf32, 8x4x2x1> -> <1x2x2x2xf32, 8x4x2x1>
    return %2 : !migraphx.shaped<1x2x2x2xf32, 8x4x2x1>
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
    auto mlir_output_with_attrs =
        migraphx::interpolate_string(mlir_output, {{"attrs", get_attrs()}});
    CHECK(encode(s) == encode(mlir_output_with_attrs));

    EXPECT(verify_mlir(m));
}

TEST_CASE(conv_add_leaky_relu)
{
    std::string mlir_output = R"__migraphx__(
module {
  func.func @mlir_convolution_add_mul_max(%arg0: !migraphx.shaped<1x2x2x2xf32, 8x4x2x1>, %arg1: !migraphx.shaped<2x8x3x3xf32, 72x9x3x1>, %arg2: !migraphx.shaped<1x8x4x4xf32, 128x16x4x1>) -> !migraphx.shaped<1x2x2x2xf32, 8x4x2x1> attributes ${attrs} {
    %0 = migraphx.literal(dense<0.00999999977> : tensor<1xf32>) : <1xf32, 1>
    %1 = migraphx.convolution %arg2, %arg1 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : <1x8x4x4xf32, 128x16x4x1>, <2x8x3x3xf32, 72x9x3x1> -> <1x2x2x2xf32, 8x4x2x1>
    %2 = migraphx.add %1, %arg0 : <1x2x2x2xf32, 8x4x2x1>, <1x2x2x2xf32, 8x4x2x1> -> <1x2x2x2xf32, 8x4x2x1>
    %3 = migraphx.multibroadcast %0 {out_dyn_dims = [], out_lens = [1, 2, 2, 2]} : <1xf32, 1> -> <1x2x2x2xf32, 0x0x0x0>
    %4 = migraphx.mul %2, %3 : <1x2x2x2xf32, 8x4x2x1>, <1x2x2x2xf32, 0x0x0x0> -> <1x2x2x2xf32, 8x4x2x1>
    %5 = migraphx.max %2, %4 : <1x2x2x2xf32, 8x4x2x1>, <1x2x2x2xf32, 8x4x2x1> -> <1x2x2x2xf32, 8x4x2x1>
    return %5 : !migraphx.shaped<1x2x2x2xf32, 8x4x2x1>
  }
}
)__migraphx__";
    migraphx::module m;
    auto x    = m.add_parameter("x", {migraphx::shape::float_type, {1, 8, 4, 4}});
    auto w    = m.add_parameter("w", {migraphx::shape::float_type, {2, 8, 3, 3}});
    auto b    = m.add_parameter("b", {migraphx::shape::float_type, {1, 2, 2, 2}});
    auto conv = m.add_instruction(migraphx::make_op("convolution"), x, w);
    auto add  = m.add_instruction(migraphx::make_op("add"), conv, b);
    auto relu = m.add_instruction(migraphx::make_op("leaky_relu"), add);
    m.add_return({relu});
    auto s = migraphx::gpu::dump_mlir(m);
    // Skip test if MLIR is not enabled
    if(s.empty())
        return;
    auto mlir_output_with_attrs =
        migraphx::interpolate_string(mlir_output, {{"attrs", get_attrs()}});
    CHECK(encode(s) == encode(mlir_output_with_attrs));

    EXPECT(verify_mlir(m));
}

// The following test checks that a dimension -1, within reshape operator is handled properly..
TEST_CASE(conv_reshape_dim_minus_one)
{
    std::string mlir_output = R"__migraphx__(
module {
  func.func @mlir_convolution_reshape(%arg0: !migraphx.shaped<2x8x3x3xf32, 72x9x3x1>, %arg1: !migraphx.shaped<1x8x4x4xf32, 128x16x4x1>) -> !migraphx.shaped<1x4x1x2xf32, 8x2x2x1> attributes ${attrs} {
    %0 = migraphx.convolution %arg1, %arg0 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : <1x8x4x4xf32, 128x16x4x1>, <2x8x3x3xf32, 72x9x3x1> -> <1x2x2x2xf32, 8x4x2x1>
    %1 = migraphx.reshape %0 {dims  =  [1,  4,  1,  2]} : <1x2x2x2xf32, 8x4x2x1> -> <1x4x1x2xf32, 8x2x2x1>
    return %1 : !migraphx.shaped<1x4x1x2xf32, 8x2x2x1>
  }
}
)__migraphx__";
    migraphx::module m;
    auto x       = m.add_parameter("x", {migraphx::shape::float_type, {1, 8, 4, 4}});
    auto w       = m.add_parameter("w", {migraphx::shape::float_type, {2, 8, 3, 3}});
    auto conv    = m.add_instruction(migraphx::make_op("convolution"), x, w);
    auto reshape = m.add_instruction(migraphx::make_op("reshape", {{"dims", {1, -1, 1, 2}}}), conv);
    m.add_return({reshape});
    auto s = migraphx::gpu::dump_mlir(m);
    // Skip test if MLIR is not enabled
    if(s.empty())
        return;
    auto mlir_output_with_attrs =
        migraphx::interpolate_string(mlir_output, {{"attrs", get_attrs()}});
    CHECK(encode(s) == encode(mlir_output_with_attrs));
    EXPECT(verify_mlir(m));
}

TEST_CASE(conv_reduce_sum)
{
    std::string mlir_output = R"__migraphx__(
module {
  func.func @mlir_convolution_reshape_reduce_sum_reshape(%arg0: !migraphx.shaped<2x8x3x3xf32, 72x9x3x1>, %arg1: !migraphx.shaped<1x8x4x4xf32, 128x16x4x1>) -> !migraphx.shaped<1x2x1x1xf32, 2x1x1x1> attributes ${attrs} {
    %0 = migraphx.convolution %arg1, %arg0 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : <1x8x4x4xf32, 128x16x4x1>, <2x8x3x3xf32, 72x9x3x1> -> <1x2x2x2xf32, 8x4x2x1>
    %1 = migraphx.reshape %0 {dims = [1,  2,  4]} : <1x2x2x2xf32, 8x4x2x1> -> <1x2x4xf32, 8x4x1>
    %2 = migraphx.reduce_sum %1 {axes = [2]} : <1x2x4xf32, 8x4x1> -> <1x2x1xf32, 2x1x1>
    %3 = migraphx.reshape %2 {dims = [1, 2, 1, 1]} : <1x2x1xf32, 2x1x1> -> <1x2x1x1xf32, 2x1x1x1>
    return %3 : !migraphx.shaped<1x2x1x1xf32, 2x1x1x1>
  }
}
)__migraphx__";
    migraphx::module m;
    auto x          = m.add_parameter("x", {migraphx::shape::float_type, {1, 8, 4, 4}});
    auto w          = m.add_parameter("w", {migraphx::shape::float_type, {2, 8, 3, 3}});
    auto conv       = m.add_instruction(migraphx::make_op("convolution"), x, w);
    auto reduce_sum = m.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2, 3}}}), conv);
    m.add_return({reduce_sum});
    auto s = migraphx::gpu::dump_mlir(m);
    // Skip test if MLIR is not enabled
    if(s.empty())
        return;
    auto mlir_output_with_attrs =
        migraphx::interpolate_string(mlir_output, {{"attrs", get_attrs()}});
    CHECK(encode(s) == encode(mlir_output_with_attrs));
    // EXPECT(verify_mlir(m));
}

TEST_CASE(conv_backwards)
{
    std::string mlir_output = R"__migraphx__(
module {
  func.func @mlir_convolution_backwards(%arg0: !migraphx.shaped<1x1x3x3xf32, 9x9x3x1>, %arg1: !migraphx.shaped<1x1x3x3xf32, 9x9x3x1>) -> !migraphx.shaped<1x1x5x5xf32, 25x25x5x1> attributes ${attrs} {
    %0 = migraphx.backwards_data_convolution %arg1, %arg0 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : <1x1x3x3xf32, 9x9x3x1>, <1x1x3x3xf32, 9x9x3x1> -> <1x1x5x5xf32, 25x25x5x1>
    return %0 : !migraphx.shaped<1x1x5x5xf32, 25x25x5x1>
  }
}
)__migraphx__";

    migraphx::module m;
    auto x      = m.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 1, 3, 3}});
    auto w      = m.add_parameter("w", migraphx::shape{migraphx::shape::float_type, {1, 1, 3, 3}});
    auto conv_b = m.add_instruction(migraphx::make_op("convolution_backwards"), x, w);
    m.add_return({conv_b});

    auto s = migraphx::gpu::dump_mlir(m);
    // Skip test if MLIR is not enabled
    if(s.empty())
        return;
    auto mlir_output_with_attrs =
        migraphx::interpolate_string(mlir_output, {{"attrs", get_attrs()}});
    CHECK(encode(s) == encode(mlir_output_with_attrs));
    EXPECT(verify_mlir(m));
}

TEST_CASE(grouped_conv1d)
{
    std::string mlir_output = R"__migraphx__(
module {
  func.func @mlir_convolution(%arg0: !migraphx.shaped<4x1x3xf32, 3x3x1>, %arg1: !migraphx.shaped<1x4x16xf32, 64x16x1>) -> !migraphx.shaped<1x4x14xf32, 56x14x1> attributes ${attrs} {
    %0 = migraphx.convolution %arg1, %arg0 {dilation = [1], group = 4 : i64, padding = [0, 0], padding_mode = 0 : i64, stride = [1]} : <1x4x16xf32, 64x16x1>, <4x1x3xf32, 3x3x1> -> <1x4x14xf32, 56x14x1>
    return %0 : !migraphx.shaped<1x4x14xf32, 56x14x1>
  }
}
)__migraphx__";

    migraphx::module m;
    auto input   = m.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 4, 16}});
    auto weights = m.add_parameter("w", migraphx::shape{migraphx::shape::float_type, {4, 1, 3}});
    auto group_conv = m.add_instruction(
        migraphx::make_op("convolution",
                          {{"group", 4}, {"padding", {0}}, {"stride", {1}}, {"dilation", {1}}}),
        input,
        weights);
    m.add_return({group_conv});
    auto s = migraphx::gpu::dump_mlir(m);
    // Skip test if MLIR is not enabled
    if(s.empty())
        return;
    auto mlir_output_with_attrs =
        migraphx::interpolate_string(mlir_output, {{"attrs", get_attrs()}});
    CHECK(encode(s) == encode(mlir_output_with_attrs));
    EXPECT(verify_mlir(m));
}

TEST_CASE(grouped_conv2d)
{
    std::string mlir_output = R"__migraphx__(
module {
  func.func @mlir_convolution(%arg0: !migraphx.shaped<4x1x3x3xf32, 9x9x3x1>, %arg1: !migraphx.shaped<1x4x16x16xf32, 1024x256x16x1>) -> !migraphx.shaped<1x4x14x14xf32, 784x196x14x1> attributes ${attrs} {
    %0 = migraphx.convolution %arg1, %arg0 {dilation = [1, 1], group = 4 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : <1x4x16x16xf32, 1024x256x16x1>, <4x1x3x3xf32, 9x9x3x1> -> <1x4x14x14xf32, 784x196x14x1>
    return %0 : !migraphx.shaped<1x4x14x14xf32, 784x196x14x1>
  }
}
)__migraphx__";

    migraphx::module m;
    auto input = m.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 4, 16, 16}});
    auto weights = m.add_parameter("w", migraphx::shape{migraphx::shape::float_type, {4, 1, 3, 3}});
    auto group_conv =
        m.add_instruction(migraphx::make_op("convolution", {{"group", 4}}), input, weights);
    m.add_return({group_conv});
    auto s = migraphx::gpu::dump_mlir(m);
    // Skip test if MLIR is not enabled
    if(s.empty())
        return;
    auto mlir_output_with_attrs =
        migraphx::interpolate_string(mlir_output, {{"attrs", get_attrs()}});
    CHECK(encode(s) == encode(mlir_output_with_attrs));
    EXPECT(verify_mlir(m));
}

TEST_CASE(grouped_conv3d)
{
    std::string mlir_output = R"__migraphx__(
module {
  func.func @mlir_convolution(%arg0: !migraphx.shaped<4x1x3x3x3xf32, 27x27x9x3x1>, %arg1: !migraphx.shaped<1x4x16x16x16xf32, 16384x4096x256x16x1>) -> !migraphx.shaped<1x4x14x14x14xf32, 10976x2744x196x14x1> attributes ${attrs} {
    %0 = migraphx.convolution %arg1, %arg0 {dilation = [1, 1, 1], group = 4 : i64, padding = [0, 0, 0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1, 1]} : <1x4x16x16x16xf32, 16384x4096x256x16x1>, <4x1x3x3x3xf32, 27x27x9x3x1> -> <1x4x14x14x14xf32, 10976x2744x196x14x1>
    return %0 : !migraphx.shaped<1x4x14x14x14xf32, 10976x2744x196x14x1>
  }
}
)__migraphx__";

    migraphx::module m;
    auto input =
        m.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 4, 16, 16, 16}});
    auto weights =
        m.add_parameter("w", migraphx::shape{migraphx::shape::float_type, {4, 1, 3, 3, 3}});
    auto group_conv = m.add_instruction(
        migraphx::make_op(
            "convolution",
            {{"group", 4}, {"padding", {0, 0, 0}}, {"stride", {1, 1, 1}}, {"dilation", {1, 1, 1}}}),
        input,
        weights);
    m.add_return({group_conv});
    auto s = migraphx::gpu::dump_mlir(m);
    // Skip test if MLIR is not enabled
    if(s.empty())
        return;
    auto mlir_output_with_attrs =
        migraphx::interpolate_string(mlir_output, {{"attrs", get_attrs()}});
    CHECK(encode(s) == encode(mlir_output_with_attrs));
    EXPECT(verify_mlir(m));
}

TEST_CASE(quant_dot_add)
{
    std::string mlir_output = R"__migraphx__(
module {
  func.func @mlir_quant_dot_add(%arg0: !migraphx.shaped<1x5x4xsi8, 20x4x1>, %arg1: !migraphx.shaped<1x4x3xsi8, 12x3x1>, %arg2: !migraphx.shaped<1x5x3xsi32, 15x3x1>) -> !migraphx.shaped<1x5x3xsi32, 15x3x1> attributes ${attrs} {
    %0 = migraphx.quant_dot %arg0, %arg1 : <1x5x4xsi8, 20x4x1>, <1x4x3xsi8, 12x3x1> -> <1x5x3xsi32, 15x3x1>
    %1 = migraphx.add %0, %arg2 : <1x5x3xsi32, 15x3x1>, <1x5x3xsi32, 15x3x1> -> <1x5x3xsi32, 15x3x1>
    return %1 : !migraphx.shaped<1x5x3xsi32, 15x3x1>
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
    auto mlir_output_with_attrs =
        migraphx::interpolate_string(mlir_output, {{"attrs", get_attrs()}});
    CHECK(encode(s) == encode(mlir_output_with_attrs));
    EXPECT(verify_mlir(m));
}

TEST_CASE(dot_add)
{
    std::string mlir_output = R"__migraphx__(
module {
  func.func @mlir_dot_add(%arg0: !migraphx.shaped<1x5x4xf32, 20x4x1>, %arg1: !migraphx.shaped<1x4x3xf32, 12x3x1>, %arg2: !migraphx.shaped<1x5x3xf32, 15x3x1>) -> !migraphx.shaped<1x5x3xf32, 15x3x1> attributes ${attrs} {
    %0 = migraphx.dot %arg0, %arg1 : <1x5x4xf32, 20x4x1>, <1x4x3xf32, 12x3x1> -> <1x5x3xf32, 15x3x1>
    %1 = migraphx.add %0, %arg2 : <1x5x3xf32, 15x3x1>, <1x5x3xf32, 15x3x1> -> <1x5x3xf32, 15x3x1>
    return %1 : !migraphx.shaped<1x5x3xf32, 15x3x1>
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
    auto mlir_output_with_attrs =
        migraphx::interpolate_string(mlir_output, {{"attrs", get_attrs()}});
    CHECK(encode(s) == encode(mlir_output_with_attrs));
    EXPECT(verify_mlir(m));
}

TEST_CASE(unsqueeze_dot_add)
{
    std::string mlir_output = R"__migraphx__(
module {
  func.func @mlir_unsqueeze_dot_add(%arg0: !migraphx.shaped<5x4xf32, 4x1>, %arg1: !migraphx.shaped<1x4x3xf32, 12x3x1>, %arg2: !migraphx.shaped<1x5x3xf32, 15x3x1>) -> !migraphx.shaped<1x5x3xf32, 15x3x1> attributes ${attrs} {
    %0 = migraphx.reshape %arg0 {dims = [1, 5, 4]} : <5x4xf32, 4x1> -> <1x5x4xf32, 20x4x1>
    %1 = migraphx.dot %0, %arg1 : <1x5x4xf32, 20x4x1>, <1x4x3xf32, 12x3x1> -> <1x5x3xf32, 15x3x1>
    %2 = migraphx.add %1, %arg2 : <1x5x3xf32, 15x3x1>, <1x5x3xf32, 15x3x1> -> <1x5x3xf32, 15x3x1>
    return %2 : !migraphx.shaped<1x5x3xf32, 15x3x1>
  }
}
)__migraphx__";
    migraphx::module m;
    auto arg0      = m.add_parameter("arg0", {migraphx::shape::float_type, {5, 4}});
    auto arg1      = m.add_parameter("arg1", {migraphx::shape::float_type, {1, 4, 3}});
    auto arg2      = m.add_parameter("arg2", {migraphx::shape::float_type, {1, 5, 3}});
    auto unsqueeze = m.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), arg0);
    auto dot       = m.add_instruction(migraphx::make_op("dot"), unsqueeze, arg1);
    auto add       = m.add_instruction(migraphx::make_op("add"), dot, arg2);
    m.add_return({add});
    auto s = migraphx::gpu::dump_mlir(m);
    // Skip test if MLIR is not enabled
    if(s.empty())
        return;
    auto mlir_output_with_attrs =
        migraphx::interpolate_string(mlir_output, {{"attrs", get_attrs()}});
    CHECK(encode(s) == encode(mlir_output_with_attrs));
    EXPECT(verify_mlir(m));
}

TEST_CASE(conv_int8_dequantize_quantize)
{
    std::string mlir_output = R"__migraphx__(
module {
  func.func @mlir_quant_convolution_dequantizelinear_quantizelinear(%arg0: !migraphx.shaped<2x8x3x3xsi8, 72x9x3x1>, %arg1: !migraphx.shaped<1x8x4x4xsi8, 128x16x4x1>, %arg2: !migraphx.shaped<1x2x2x2xf32, 8x4x2x1>, %arg3: !migraphx.shaped<1x2x2x2xsi32, 8x4x2x1>) -> !migraphx.shaped<1x2x2x2xsi32, 8x4x2x1> attributes ${attrs} {
      %0 = migraphx.quant_convolution %arg1, %arg0 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : <1x8x4x4xsi8, 128x16x4x1>, <2x8x3x3xsi8, 72x9x3x1> -> <1x2x2x2xsi32, 8x4x2x1>
      %1 = migraphx.dequantizelinear %0, %arg2, %arg3 : <1x2x2x2xsi32, 8x4x2x1>, <1x2x2x2xf32, 8x4x2x1>, !migraphx.shaped<1x2x2x2xsi32, 8x4x2x1> -> <1x2x2x2xf32, 8x4x2x1>
      %2 = migraphx.quantizelinear %1, %arg2, %arg3 : <1x2x2x2xf32, 8x4x2x1>, <1x2x2x2xf32, 8x4x2x1>, !migraphx.shaped<1x2x2x2xsi32, 8x4x2x1> -> <1x2x2x2xsi32, 8x4x2x1>
      return %2 : !migraphx.shaped<1x2x2x2xsi32, 8x4x2x1>
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
    auto mlir_output_with_attrs =
        migraphx::interpolate_string(mlir_output, {{"attrs", get_attrs()}});
    CHECK(encode(s) == encode(mlir_output_with_attrs));
    EXPECT(verify_mlir(m));
}

TEST_CASE(dot_convert)
{
    std::string mlir_output = R"__migraphx__(
module {
  func.func @mlir_dot_convert(%arg0: !migraphx.shaped<1x5x4xf32, 20x4x1>, %arg1: !migraphx.shaped<1x4x3xf32, 12x3x1>) -> !migraphx.shaped<1x5x3xf16, 15x3x1> attributes ${attrs} {
    %0 = migraphx.dot %arg0, %arg1 : <1x5x4xf32, 20x4x1>, <1x4x3xf32, 12x3x1> -> <1x5x3xf32, 15x3x1>
    %1 = migraphx.convert %0 {target_type  =  1  :  i64} : <1x5x3xf32, 15x3x1> to <1x5x3xf16, 15x3x1>
    return %1 : !migraphx.shaped<1x5x3xf16, 15x3x1>
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
    auto mlir_output_with_attrs =
        migraphx::interpolate_string(mlir_output, {{"attrs", get_attrs()}});
    CHECK(encode(s) == encode(mlir_output_with_attrs));
    EXPECT(verify_mlir(m));
}

TEST_CASE(dot_where)
{
    std::string mlir_output = R"__migraphx__(
module {
  func.func @mlir_dot_where(%arg0: !migraphx.shaped<1x5x4xf32, 20x4x1>, %arg1: !migraphx.shaped<1x4x3xf32, 12x3x1>, %arg2: !migraphx.shaped<1x5x3xsi8, 15x3x1>, %arg3: !migraphx.shaped<1x5x3xf32, 15x3x1>) -> !migraphx.shaped<1x5x3xf32, 15x3x1> attributes ${attrs} {
    %0 = migraphx.dot %arg0, %arg1 : <1x5x4xf32, 20x4x1>, <1x4x3xf32, 12x3x1> -> <1x5x3xf32, 15x3x1>
    %1 = migraphx.where %arg2, %0, %arg3 : <1x5x3xsi8, 15x3x1>, <1x5x3xf32, 15x3x1>, <1x5x3xf32, 15x3x1> -> <1x5x3xf32, 15x3x1>
    return %1 : !migraphx.shaped<1x5x3xf32, 15x3x1>
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
    auto mlir_output_with_attrs =
        migraphx::interpolate_string(mlir_output, {{"attrs", get_attrs()}});
    CHECK(encode(s) == encode(mlir_output_with_attrs));

    EXPECT(verify_mlir(m));
}

TEST_CASE(int4_unpack_ir)
{
    std::string mlir_output = R"__migraphx__(
module {
  func.func @mlir_unpack_int4(%arg0: !migraphx.shaped<2x1xsi8, 1x1>) -> !migraphx.shaped<2x2xsi8, 2x1> attributes ${attrs} {
    %0 = migraphx.unpack %arg0 {axis = 1 : i64} : <2x1xsi8, 1x1> -> <2x2xsi8, 2x1>
    return %0 : !migraphx.shaped<2x2xsi8, 2x1>
  }
}
)__migraphx__";
    migraphx::module m;
    auto arg0 = m.add_parameter("arg0", {migraphx::shape::int8_type, {2, 1}});
    auto unpk = m.add_instruction(migraphx::make_op("unpack_int4"), arg0);
    m.add_return({unpk});
    auto s = migraphx::gpu::dump_mlir(m);

    // Skip test if MLIR is not enabled
    if(s.empty())
        return;

    auto mlir_output_with_attrs =
        migraphx::interpolate_string(mlir_output, {{"attrs", get_attrs()}});

    CHECK(encode(s) == encode(mlir_output_with_attrs));
}

TEST_CASE(int4_unpack_conv)
{
    std::string mlir_output = R"__migraphx__(
module {
  func.func @mlir_unpack_int4_quant_convolution(%arg0: !migraphx.shaped<2x8x2x1xsi8, 16x2x1x1>, %arg1: !migraphx.shaped<1x8x4x4xsi8, 128x16x4x1>) -> !migraphx.shaped<1x2x3x3xsi32, 18x9x3x1> attributes ${attrs} {
    %0 = migraphx.unpack %arg0 {axis = 3 : i64} : <2x8x2x1xsi8, 16x2x1x1> -> <2x8x2x2xsi8, 32x4x2x1>
    %1 = migraphx.quant_convolution %arg1, %0 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : <1x8x4x4xsi8, 128x16x4x1>, <2x8x2x2xsi8, 32x4x2x1> -> <1x2x3x3xsi32, 18x9x3x1>
    return %1 : !migraphx.shaped<1x2x3x3xsi32, 18x9x3x1>
  }
}
)__migraphx__";
    migraphx::module m;
    auto x    = m.add_parameter("x", {migraphx::shape::int8_type, {1, 8, 4, 4}});
    auto pk_w = m.add_parameter("w", {migraphx::shape::int8_type, {2, 8, 2, 1}});
    auto w    = m.add_instruction(migraphx::make_op("unpack_int4"), pk_w);
    auto conv = m.add_instruction(migraphx::make_op("quant_convolution"), x, w); // w: {2,8,2,2}
    m.add_return({conv});
    auto s = migraphx::gpu::dump_mlir(m);
    // Skip test if MLIR is not enabled
    if(s.empty())
        return;
    auto mlir_output_with_attrs =
        migraphx::interpolate_string(mlir_output, {{"attrs", get_attrs()}});
    CHECK(encode(s) == encode(mlir_output_with_attrs));
    EXPECT(verify_mlir(m));
}

TEST_CASE(int4_unpack_dequantizelinear)
{
    std::string mlir_output = R"__migraphx__(
module {
  func.func @mlir_unsqueeze_reshape_slice_unsqueeze_reshape_slice_unpack_int4_dequantizelinear_dot(%arg0: !migraphx.shaped<2x3x5xf32, 15x5x1>, %arg1: !migraphx.shaped<2x5x1xsi8, 5x1x1>, %arg2: !migraphx.shaped<2x2x2xf32, 4x2x1>, %arg3: !migraphx.shaped<2x2x2xsi8, 4x2x1>) -> !migraphx.shaped<2x3x2xf32, 6x2x1> attributes ${attrs} {
    %0 = migraphx.reshape %arg2 {dims = [2, 2, 1, 2]} : <2x2x2xf32, 4x2x1> -> <2x2x1x2xf32, 4x2x2x1>
    %1 = migraphx.multibroadcast %0 {out_dyn_dims = [], out_lens = [2, 2, 3, 2]} : <2x2x1x2xf32, 4x2x2x1> -> <2x2x3x2xf32, 4x2x0x1>
    %2 = migraphx.reshape %1 {dims = [2, 6, 2]} : <2x2x3x2xf32, 4x2x0x1> -> <2x6x2xf32, 12x2x1>
    %3 = migraphx.slice %2 {axes = [1], ends = [5], starts = [0]} : <2x6x2xf32, 12x2x1> -> <2x5x2xf32, 12x2x1>
    %4 = migraphx.reshape %arg3 {dims = [2, 2, 1, 2]} : <2x2x2xsi8, 4x2x1> -> <2x2x1x2xsi8, 4x2x2x1>
    %5 = migraphx.multibroadcast %4 {out_dyn_dims = [], out_lens = [2, 2, 3, 2]} : <2x2x1x2xsi8, 4x2x2x1> -> <2x2x3x2xsi8, 4x2x0x1>
    %6 = migraphx.reshape %5 {dims = [2, 6, 2]} : <2x2x3x2xsi8, 4x2x0x1> -> <2x6x2xsi8, 12x2x1>
    %7 = migraphx.slice %6 {axes = [1], ends = [5], starts = [0]} : <2x6x2xsi8, 12x2x1> -> <2x5x2xsi8, 12x2x1>
    %8 = migraphx.unpack %arg1 {axis = 2 : i64} : <2x5x1xsi8, 5x1x1> -> <2x5x2xsi8, 10x2x1>
    %9 = migraphx.dequantizelinear %8, %3, %7 : <2x5x2xsi8, 10x2x1>, <2x5x2xf32, 12x2x1>, !migraphx.shaped<2x5x2xsi8, 12x2x1> -> <2x5x2xf32, 10x2x1>
    %10 = migraphx.dot %arg0, %9 : <2x3x5xf32, 15x5x1>, <2x5x2xf32, 10x2x1> -> <2x3x2xf32, 6x2x1>
    return %10 : !migraphx.shaped<2x3x2xf32, 6x2x1>
  }
}
)__migraphx__";
    migraphx::module m;
    auto x0 = m.add_parameter("x0", migraphx::shape{migraphx::shape::float_type, {2, 3, 5}});
    auto x1 = m.add_parameter("x1", migraphx::shape{migraphx::shape::int8_type, {2, 5, 1}});
    auto x2 = m.add_parameter("x2", migraphx::shape{migraphx::shape::float_type, {2, 2, 2}});
    auto x3 = m.add_parameter("x3", migraphx::shape{migraphx::shape::int8_type, {2, 2, 2}});

    auto unsqueeze1 = m.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), x2);
    auto broadcast1 = m.add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 3, 2}}}), unsqueeze1);
    auto reshape1 =
        m.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 6, 2}}}), broadcast1);
    auto scale = m.add_instruction(
        migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {5}}}), reshape1);

    auto unsqueeze2 = m.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), x3);
    auto broadcast2 = m.add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 3, 2}}}), unsqueeze2);
    auto reshape2 =
        m.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 6, 2}}}), broadcast2);
    auto zp = m.add_instruction(
        migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {5}}}), reshape2);

    auto unpack = m.add_instruction(migraphx::make_op("unpack_int4"), x1);
    auto dq     = m.add_instruction(migraphx::make_op("dequantizelinear"), unpack, scale, zp);
    auto dot    = m.add_instruction(migraphx::make_op("dot"), x0, dq);
    m.add_return({dot});
    auto s = migraphx::gpu::dump_mlir(m);
    // Skip test if MLIR is not enabled
    if(s.empty())
        return;
    auto mlir_output_with_attrs =
        migraphx::interpolate_string(mlir_output, {{"attrs", get_attrs()}});
    CHECK(encode(s) == encode(mlir_output_with_attrs));
    EXPECT(verify_mlir(m));
}

TEST_CASE(uint4_unpack_dequantizelinear)
{
    std::string mlir_output = R"__migraphx__(
module {
  func.func @mlir_unsqueeze_reshape_slice_unsqueeze_reshape_slice_unpack_int4_dequantizelinear_dot(%arg0: !migraphx.shaped<2x3x5xf32, 15x5x1>, %arg1: !migraphx.shaped<2x5x1xui8, 5x1x1>, %arg2: !migraphx.shaped<2x2x2xf32, 4x2x1>, %arg3: !migraphx.shaped<2x2x2xui8, 4x2x1>) -> !migraphx.shaped<2x3x2xf32, 6x2x1> attributes ${attrs} {
    %0 = migraphx.reshape %arg2 {dims = [2, 2, 1, 2]} : <2x2x2xf32, 4x2x1> -> <2x2x1x2xf32, 4x2x2x1>
    %1 = migraphx.multibroadcast %0 {out_dyn_dims = [], out_lens = [2, 2, 3, 2]} : <2x2x1x2xf32, 4x2x2x1> -> <2x2x3x2xf32, 4x2x0x1>
    %2 = migraphx.reshape %1 {dims = [2, 6, 2]} : <2x2x3x2xf32, 4x2x0x1> -> <2x6x2xf32, 12x2x1>
    %3 = migraphx.slice %2 {axes = [1], ends = [5], starts = [0]} : <2x6x2xf32, 12x2x1> -> <2x5x2xf32, 12x2x1>
    %4 = migraphx.reshape %arg3 {dims = [2, 2, 1, 2]} : <2x2x2xui8, 4x2x1> -> <2x2x1x2xui8, 4x2x2x1>
    %5 = migraphx.multibroadcast %4 {out_dyn_dims = [], out_lens = [2, 2, 3, 2]} : <2x2x1x2xui8, 4x2x2x1> -> <2x2x3x2xui8, 4x2x0x1>
    %6 = migraphx.reshape %5 {dims = [2, 6, 2]} : <2x2x3x2xui8, 4x2x0x1> -> <2x6x2xui8, 12x2x1>
    %7 = migraphx.slice %6 {axes = [1], ends = [5], starts = [0]} : <2x6x2xui8, 12x2x1> -> <2x5x2xui8, 12x2x1>
    %8 = migraphx.unpack %arg1 {axis = 2 : i64} : <2x5x1xui8, 5x1x1> -> <2x5x2xui8, 10x2x1>
    %9 = migraphx.dequantizelinear %8, %3, %7 : <2x5x2xui8, 10x2x1>, <2x5x2xf32, 12x2x1>, !migraphx.shaped<2x5x2xui8, 12x2x1> -> <2x5x2xf32, 10x2x1>
    %10 = migraphx.dot %arg0, %9 : <2x3x5xf32, 15x5x1>, <2x5x2xf32, 10x2x1> -> <2x3x2xf32, 6x2x1>
    return %10 : !migraphx.shaped<2x3x2xf32, 6x2x1>
  }
}
)__migraphx__";
    migraphx::module m;
    auto x0 = m.add_parameter("x0", migraphx::shape{migraphx::shape::float_type, {2, 3, 5}});
    auto x1 = m.add_parameter("x1", migraphx::shape{migraphx::shape::uint8_type, {2, 5, 1}});
    auto x2 = m.add_parameter("x2", migraphx::shape{migraphx::shape::float_type, {2, 2, 2}});
    auto x3 = m.add_parameter("x3", migraphx::shape{migraphx::shape::uint8_type, {2, 2, 2}});

    auto unsqueeze1 = m.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), x2);
    auto broadcast1 = m.add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 3, 2}}}), unsqueeze1);
    auto reshape1 =
        m.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 6, 2}}}), broadcast1);
    auto scale = m.add_instruction(
        migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {5}}}), reshape1);

    auto unsqueeze2 = m.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), x3);
    auto broadcast2 = m.add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 3, 2}}}), unsqueeze2);
    auto reshape2 =
        m.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 6, 2}}}), broadcast2);
    auto zp = m.add_instruction(
        migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {5}}}), reshape2);

    auto unpack = m.add_instruction(migraphx::make_op("unpack_int4"), x1);
    auto dq     = m.add_instruction(migraphx::make_op("dequantizelinear"), unpack, scale, zp);
    auto dot    = m.add_instruction(migraphx::make_op("dot"), x0, dq);
    m.add_return({dot});
    auto s = migraphx::gpu::dump_mlir(m);
    // Skip test if MLIR is not enabled
    if(s.empty())
        return;
    auto mlir_output_with_attrs =
        migraphx::interpolate_string(mlir_output, {{"attrs", get_attrs()}});
    CHECK(encode(s) == encode(mlir_output_with_attrs));
    EXPECT(verify_mlir(m));
}

TEST_CASE(mxfp4_gemm)
{
    std::string mlir_output = R"__migraphx__(
module {
  func.func @mlir_unpack_fp4_unpack_fp4_transpose_reshape_reshape_quant_dot_add(%arg0: !migraphx.shaped<1x2048xf4E2M1FN, 2048x1>, %arg1: !migraphx.shaped<1000x2048xf4E2M1FN, 2048x1>, %arg2: !migraphx.shaped<1x64x1xf32, 64x1x1>, %arg3: !migraphx.shaped<64x1x1000xf32, 1x1x64>, %arg4: !migraphx.shaped<1x1000xf32, 1000x1>) -> !migraphx.shaped<1x1000xf32, 1000x1> attributes ${attrs} {
    %0 = migraphx.transpose %arg1 {permutation = [1, 0]} : <1000x2048xf4E2M1FN, 2048x1> -> <2048x1000xf4E2M1FN, 1x2048>
    %1 = migraphx.multibroadcast %arg2 {out_dyn_dims = [], out_lens = [1, 64, 32]} : <1x64x1xf32, 64x1x1> -> <1x64x32xf32, 64x1x0>
    %2 = migraphx.reshape %1 {dims = [1, 2048]} : <1x64x32xf32, 64x1x0> -> <1x2048xf32, 2048x1>
    %3 = migraphx.multibroadcast %arg3 {out_dyn_dims = [], out_lens = [64, 32, 1000]} : <64x1x1000xf32, 1x1x64> -> <64x32x1000xf32, 1x0x64>
    %4 = migraphx.reshape %3 {dims = [2048, 1000]} : <64x32x1000xf32, 1x0x64> -> <2048x1000xf32, 1000x1>
    %5 = migraphx.quant_dot %arg0 scaled by %2, %0 scaled by %4 : <1x2048xf4E2M1FN, 2048x1> scaled by !migraphx.shaped<1x2048xf32, 2048x1>, <2048x1000xf4E2M1FN, 1x2048> scaled by !migraphx.shaped<2048x1000xf32, 1000x1> -> <1x1000xf32, 1000x1>
    %6 = migraphx.add %5, %arg4 : <1x1000xf32, 1000x1>, <1x1000xf32, 1000x1> -> <1x1000xf32, 1000x1>
    return %6 : !migraphx.shaped<1x1000xf32, 1000x1>
  }
}
)__migraphx__";
    migraphx::module m;
    auto x5      = m.add_parameter("x5", {migraphx::shape::float_type, {1, 1000}});
    auto x4      = m.add_parameter("x4", {migraphx::shape::float_type, {64, 1, 1000}, {1, 1, 64}});
    auto x3      = m.add_parameter("x3", {migraphx::shape::float_type, {1, 64, 1}});
    auto x2      = m.add_parameter("x2", {migraphx::shape::fp4x2_type, {1000, 1024}});
    auto x1      = m.add_parameter("x1", {migraphx::shape::fp4x2_type, {1, 1024}});
    auto unpack1 = m.add_instruction(migraphx::make_op("unpack_fp4", {{"axis", 1}}), x1);
    auto unpack2 = m.add_instruction(migraphx::make_op("unpack_fp4", {{"axis", 1}}), x2);
    auto trans =
        m.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), unpack2);
    auto mbcast1 =
        m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {1, 64, 32}}}), x3);
    auto reshape1 = m.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 2048}}}), mbcast1);
    auto mbcast2 =
        m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {64, 32, 1000}}}), x4);
    auto reshape2 =
        m.add_instruction(migraphx::make_op("reshape", {{"dims", {2048, 1000}}}), mbcast2);
    auto qdot =
        m.add_instruction(migraphx::make_op("quant_dot"), unpack1, trans, reshape1, reshape2);
    auto add = m.add_instruction(migraphx::make_op("add"), qdot, x5);
    m.add_return({add});

    auto s = migraphx::gpu::dump_mlir(m);
    // Skip test if MLIR is not enabled
    if(s.empty())
        return;
    auto mlir_output_with_attrs =
        migraphx::interpolate_string(mlir_output, {{"attrs", get_attrs()}});
    CHECK(encode(s) == encode(mlir_output_with_attrs));
    // Don't verify here. Tests with a verify test instead.
}

TEST_CASE(mlir_lds_usage_fits_arch)
{
    const auto device_name = migraphx::gpu::get_device_name();
    const auto gfx_name    = migraphx::gpu::get_gfx_name(device_name);
    EXPECT(
        migraphx::gpu::mlir_lds_usage_fits_arch(64, gfx_name, migraphx::shape::type_t::half_type));
    EXPECT(not migraphx::gpu::mlir_lds_usage_fits_arch(
        8192, gfx_name, migraphx::shape::type_t::half_type));
    EXPECT(migraphx::gpu::mlir_lds_usage_fits_arch(
        64, device_name, migraphx::shape::type_t::half_type));
}

// prepare_mlir rewrites a non-standard-strided constant (as folded from a transposed literal) to
// a standard shape so the emitted MLIR literal is accepted.
TEST_CASE(dot_nonstandard_literal)
{
    std::string mlir_output = R"__migraphx__(
module {
  func.func @mlir_dot(%arg0: !migraphx.shaped<1x2x3xf32, 6x3x1>) -> !migraphx.shaped<1x2x2xf32, 4x2x1> attributes ${attrs} {
    %0 = migraphx.literal(dense<[[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00], [5.000000e+00, 6.000000e+00]]]> : tensor<1x3x2xf32>) : <1x3x2xf32, 6x2x1>
    %1 = migraphx.dot %arg0, %0 : <1x2x3xf32, 6x3x1>, <1x3x2xf32, 6x2x1> -> <1x2x2xf32, 4x2x1>
    return %1 : !migraphx.shaped<1x2x2xf32, 4x2x1>
  }
}
)__migraphx__";
    migraphx::module m;
    auto arg0 = m.add_parameter("arg0", {migraphx::shape::float_type, {1, 2, 3}});
    migraphx::shape ws{migraphx::shape::float_type, {1, 3, 2}, {6, 1, 3}};
    auto lit = m.add_literal(migraphx::literal{ws, {1, 2, 3, 4, 5, 6}});
    auto dot = m.add_instruction(migraphx::make_op("dot"), arg0, lit);
    m.add_return({dot});
    migraphx::run_passes(m, {migraphx::gpu::prepare_mlir{}});

    auto s = migraphx::gpu::dump_mlir(m);
    // Skip test if MLIR is not enabled
    if(s.empty())
        return;
    auto mlir_output_with_attrs =
        migraphx::interpolate_string(mlir_output, {{"attrs", get_attrs()}});
    CHECK(encode(s) == encode(mlir_output_with_attrs));
}

// rocMLIR accumulates a reduction into its output buffer, so it asks for that buffer to be
// zero-initialized through a rock.prefill attribute typed after the buffer's element type.
// hip::fill only takes an integer value, so compile_mlir has to convert both the float
// attribute a float reduction produces and the integer one an i32 reduction produces.
TEST_CASE_SKIP(prefill_float_reduce, "temporarily disabled")
{
    migraphx::module m;
    auto a      = m.add_parameter("a", {migraphx::shape::float_type, {1, 5, 4}});
    auto b      = m.add_parameter("b", {migraphx::shape::float_type, {1, 4, 3}});
    auto dot    = m.add_instruction(migraphx::make_op("dot"), a, b);
    auto reduce = m.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), dot);
    m.add_return({reduce});
    // Skip test if MLIR is not enabled
    if(migraphx::gpu::dump_mlir(m).empty())
        return;

    // compile_mlir takes the parameter shapes in sorted-name order with the output shape last.
    std::vector<migraphx::shape> shapes = {a->get_shape(), b->get_shape(), reduce->get_shape()};
    migraphx::gpu::context ctx;
    auto tc = get_tuning_config_mlir(ctx, create_mlir_submodule(m), shapes, false);
    EXPECT(not tc.solutions.empty());
    auto mco = compile_mlir(ctx, create_mlir_submodule(m), shapes, tc.solutions.front());

    EXPECT(not mco.prefill_values.empty());
    EXPECT(mco.prefill_indices.size() == mco.prefill_values.size());
    EXPECT(migraphx::all_of(mco.prefill_values, [](const migraphx::value& v) {
        return v.is_int64() and v.to<int>() == 0;
    }));
}

TEST_CASE_SKIP(prefill_integer_reduce, "temporarily disabled")
{
    migraphx::module m;
    auto a      = m.add_parameter("a", {migraphx::shape::int8_type, {1, 5, 4}});
    auto b      = m.add_parameter("b", {migraphx::shape::int8_type, {1, 4, 3}});
    auto dot    = m.add_instruction(migraphx::make_op("quant_dot"), a, b);
    auto reduce = m.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), dot);
    m.add_return({reduce});
    EXPECT(reduce->get_shape().type() == migraphx::shape::int32_type);
    // Skip test if MLIR is not enabled
    if(migraphx::gpu::dump_mlir(m).empty())
        return;

    std::vector<migraphx::shape> shapes = {a->get_shape(), b->get_shape(), reduce->get_shape()};
    migraphx::gpu::context ctx;
    auto tc = get_tuning_config_mlir(ctx, create_mlir_submodule(m), shapes, false);
    EXPECT(not tc.solutions.empty());
    auto mco = compile_mlir(ctx, create_mlir_submodule(m), shapes, tc.solutions.front());

    EXPECT(not mco.prefill_values.empty());
    EXPECT(mco.prefill_indices.size() == mco.prefill_values.size());
    EXPECT(migraphx::all_of(mco.prefill_values, [](const migraphx::value& v) {
        return v.is_int64() and v.to<int>() == 0;
    }));
}

// Decode-shaped kv-cache attention module with a scalar sequence-length mask
// and GQA broadcasting, as produced by find_kv_cache_attention
static migraphx::module make_kv_cache_attention_module(bool with_lse)
{
    migraphx::module m;
    migraphx::shape s_q{migraphx::shape::half_type, {1, 4, 1, 4}};
    migraphx::shape s_kv{migraphx::shape::half_type, {1, 2, 8, 4}};
    migraphx::shape s_scalar{migraphx::shape::half_type, {1}};
    std::vector<std::size_t> mask_lens{1, 4, 1, 8};

    auto scale = m.add_literal(migraphx::literal{s_scalar, {0.125}});
    auto ninf =
        m.add_literal(migraphx::literal{s_scalar, {-std::numeric_limits<float>::infinity()}});
    auto range = m.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::int32_type, {8}},
                                                 {0, 1, 2, 3, 4, 5, 6, 7}});
    auto q     = m.add_parameter("q", s_q);
    auto k     = m.add_parameter("k", s_kv);
    auto v     = m.add_parameter("v", s_kv);
    auto seq_len = m.add_parameter("seq_len", {migraphx::shape::int32_type, {1}});

    auto unsq_k = m.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), k);
    auto tsp_k  = m.add_instruction(
        migraphx::make_op("transpose", {{"permutation", {0, 1, 2, 4, 3}}}), unsq_k);
    auto bc_k = m.add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", {1, 2, 2, 4, 8}}}), tsp_k);
    auto rsp_k  = m.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 4, 4, 8}}}), bc_k);
    auto unsq_v = m.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), v);
    auto bc_v   = m.add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", {1, 2, 2, 8, 4}}}), unsq_v);
    auto rsp_v = m.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 4, 8, 4}}}), bc_v);
    auto gemm1 = m.add_instruction(migraphx::make_op("dot"), q, rsp_k);
    auto bc_scale =
        m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", mask_lens}}), scale);
    auto scaled = m.add_instruction(migraphx::make_op("mul"), gemm1, bc_scale);
    auto bc_range =
        m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", mask_lens}}), range);
    auto rsp_sl =
        m.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 1, 1, 1}}}), seq_len);
    auto bc_sl =
        m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", mask_lens}}), rsp_sl);
    auto grt  = m.add_instruction(migraphx::make_op("greater"), bc_range, bc_sl);
    auto cond = m.add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::bool_type}}), grt);
    auto bc_ninf =
        m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", mask_lens}}), ninf);
    auto mask = m.add_instruction(migraphx::make_op("where"), cond, bc_ninf, scaled);
    auto rmax = m.add_instruction(migraphx::make_op("reduce_max", {{"axes", {3}}}), mask);
    auto bc_rm =
        m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", mask_lens}}), rmax);
    auto sub  = m.add_instruction(migraphx::make_op("sub"), mask, bc_rm);
    auto exp  = m.add_instruction(migraphx::make_op("exp"), sub);
    auto rsum = m.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {3}}}), exp);
    auto bc_rs =
        m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", mask_lens}}), rsum);
    auto div   = m.add_instruction(migraphx::make_op("div"), exp, bc_rs);
    auto gemm2 = m.add_instruction(migraphx::make_op("dot"), div, rsp_v);
    auto tsp_out =
        m.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), gemm2);
    auto rsp_out = m.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 1, 16}}}), tsp_out);
    if(with_lse)
    {
        auto log_sum = m.add_instruction(migraphx::make_op("log"), rsum);
        auto lse     = m.add_instruction(migraphx::make_op("add"), rmax, log_sum);
        m.add_return({rsp_out, lse});
    }
    else
    {
        m.add_return({rsp_out});
    }
    return m;
}

// The scalar sequence-length mask must be emitted with the sequence length
// broadcast over the leading {batch, heads} dims in a separate step
// (find_kv_cache_mask_seq_len) so rocMLIR can bind a currentSeqLen matching
// the attention batch even when Q is a plain input.
TEST_CASE(kv_cache_attention_seq_len_mask)
{
    std::string mlir_output = R"__migraphx__(
module {
  func.func @mlir_unsqueeze_transpose_reshape_unsqueeze_reshape_dot_mul_reshape_unsqueeze_greater_convert_where_reshape_reduce_max_reshape_sub_exp_reshape_reduce_sum_reshape_div_dot_transpose_reshape(%arg0: !migraphx.shaped<1x2x8x4xf16, 64x32x4x1>, %arg1: !migraphx.shaped<1x4x1x4xf16, 16x4x4x1>, %arg2: !migraphx.shaped<1xsi32, 1>, %arg3: !migraphx.shaped<1x2x8x4xf16, 64x32x4x1>) -> !migraphx.shaped<1x1x16xf16, 16x16x1> attributes ${attrs} {
    %0 = migraphx.literal(dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xsi32>) : <8xsi32, 1>
    %1 = migraphx.literal(dense<0xFC00> : tensor<1xf16>) : <1xf16, 1>
    %2 = migraphx.literal(dense<1.250000e-01> : tensor<1xf16>) : <1xf16, 1>
    %3 = migraphx.reshape %arg0 {dims = [1, 2, 1, 8, 4]} : <1x2x8x4xf16, 64x32x4x1> -> <1x2x1x8x4xf16, 64x32x32x4x1>
    %4 = migraphx.transpose %3 {permutation = [0, 1, 2, 4, 3]} : <1x2x1x8x4xf16, 64x32x32x4x1> -> <1x2x1x4x8xf16, 64x32x32x1x4>
    %5 = migraphx.multibroadcast %4 {out_dyn_dims = [], out_lens = [1, 2, 2, 4, 8]} : <1x2x1x4x8xf16, 64x32x32x1x4> -> <1x2x2x4x8xf16, 64x32x0x1x4>
    %6 = migraphx.reshape %5 {dims = [1, 4, 4, 8]} : <1x2x2x4x8xf16, 64x32x0x1x4> -> <1x4x4x8xf16, 128x32x8x1>
    %7 = migraphx.reshape %arg3 {dims = [1, 2, 1, 8, 4]} : <1x2x8x4xf16, 64x32x4x1> -> <1x2x1x8x4xf16, 64x32x32x4x1>
    %8 = migraphx.multibroadcast %7 {out_dyn_dims = [], out_lens = [1, 2, 2, 8, 4]} : <1x2x1x8x4xf16, 64x32x32x4x1> -> <1x2x2x8x4xf16, 64x32x0x4x1>
    %9 = migraphx.reshape %8 {dims = [1, 4, 8, 4]} : <1x2x2x8x4xf16, 64x32x0x4x1> -> <1x4x8x4xf16, 128x32x4x1>
    %10 = migraphx.dot %arg1, %6 : <1x4x1x4xf16, 16x4x4x1>, <1x4x4x8xf16, 128x32x8x1> -> <1x4x1x8xf16, 32x8x8x1>
    %11 = migraphx.multibroadcast %2 {out_dyn_dims = [], out_lens = [1, 4, 1, 8]} : <1xf16, 1> -> <1x4x1x8xf16, 0x0x0x0>
    %12 = migraphx.mul %10, %11 : <1x4x1x8xf16, 32x8x8x1>, <1x4x1x8xf16, 0x0x0x0> -> <1x4x1x8xf16, 32x8x8x1>
    %13 = migraphx.multibroadcast %0 {out_dyn_dims = [], out_lens = [1, 4, 1, 8]} : <8xsi32, 1> -> <1x4x1x8xsi32, 0x0x0x1>
    %14 = migraphx.reshape %arg2 {dims = [1]} : <1xsi32, 1> -> <1xsi32, 1>
    %15 = migraphx.multibroadcast %14 {out_dyn_dims = [], out_lens = [1, 4]} : <1xsi32, 1> -> <1x4xsi32, 0x0>
    %16 = migraphx.reshape %15 {dims = [1, 4, 1, 1]} : <1x4xsi32, 0x0> -> <1x4x1x1xsi32, 0x0x1x1>
    %17 = migraphx.multibroadcast %16 {out_dyn_dims = [], out_lens = [1, 4, 1, 8]} : <1x4x1x1xsi32, 0x0x1x1> -> <1x4x1x8xsi32, 0x0x1x0>
    %18 = migraphx.greater %13, %17 : <1x4x1x8xsi32, 0x0x0x1>, <1x4x1x8xsi32, 0x0x1x0> -> <1x4x1x8xsi32, 32x8x8x1>
    %19 = migraphx.convert %18 {target_type = 0 : i64} : <1x4x1x8xsi32, 32x8x8x1> to <1x4x1x8xsi8, 32x8x8x1>
    %20 = migraphx.multibroadcast %1 {out_dyn_dims = [], out_lens = [1, 4, 1, 8]} : <1xf16, 1> -> <1x4x1x8xf16, 0x0x0x0>
    %21 = migraphx.where %19, %20, %12 : <1x4x1x8xsi8, 32x8x8x1>, <1x4x1x8xf16, 0x0x0x0>, <1x4x1x8xf16, 32x8x8x1> -> <1x4x1x8xf16, 32x8x8x1>
    %22 = migraphx.reshape %21 {dims = [1, 4, 1, 8]} : <1x4x1x8xf16, 32x8x8x1> -> <1x4x1x8xf16, 32x8x8x1>
    %23 = migraphx.reduce_max %22 {axes = [3]} : <1x4x1x8xf16, 32x8x8x1> -> <1x4x1x1xf16, 4x1x1x1>
    %24 = migraphx.reshape %23 {dims = [1, 4, 1, 1]} : <1x4x1x1xf16, 4x1x1x1> -> <1x4x1x1xf16, 4x1x1x1>
    %25 = migraphx.multibroadcast %24 {out_dyn_dims = [], out_lens = [1, 4, 1, 8]} : <1x4x1x1xf16, 4x1x1x1> -> <1x4x1x8xf16, 4x1x1x0>
    %26 = migraphx.sub %21, %25 : <1x4x1x8xf16, 32x8x8x1>, <1x4x1x8xf16, 4x1x1x0> -> <1x4x1x8xf16, 32x8x8x1>
    %27 = migraphx.exp %26 : <1x4x1x8xf16, 32x8x8x1> -> <1x4x1x8xf16, 32x8x8x1>
    %28 = migraphx.reshape %27 {dims = [1, 4, 1, 8]} : <1x4x1x8xf16, 32x8x8x1> -> <1x4x1x8xf16, 32x8x8x1>
    %29 = migraphx.reduce_sum %28 {axes = [3]} : <1x4x1x8xf16, 32x8x8x1> -> <1x4x1x1xf16, 4x1x1x1>
    %30 = migraphx.reshape %29 {dims = [1, 4, 1, 1]} : <1x4x1x1xf16, 4x1x1x1> -> <1x4x1x1xf16, 4x1x1x1>
    %31 = migraphx.multibroadcast %30 {out_dyn_dims = [], out_lens = [1, 4, 1, 8]} : <1x4x1x1xf16, 4x1x1x1> -> <1x4x1x8xf16, 4x1x1x0>
    %32 = migraphx.div %27, %31 : <1x4x1x8xf16, 32x8x8x1>, <1x4x1x8xf16, 4x1x1x0> -> <1x4x1x8xf16, 32x8x8x1>
    %33 = migraphx.dot %32, %9 : <1x4x1x8xf16, 32x8x8x1>, <1x4x8x4xf16, 128x32x4x1> -> <1x4x1x4xf16, 16x4x4x1>
    %34 = migraphx.transpose %33 {permutation = [0, 2, 1, 3]} : <1x4x1x4xf16, 16x4x4x1> -> <1x1x4x4xf16, 16x4x4x1>
    %35 = migraphx.reshape %34 {dims = [1, 1, 16]} : <1x1x4x4xf16, 16x4x4x1> -> <1x1x16xf16, 16x16x1>
    return %35 : !migraphx.shaped<1x1x16xf16, 16x16x1>
  }
}
)__migraphx__";
    auto m                  = make_kv_cache_attention_module(false);
    auto s                  = migraphx::gpu::dump_mlir(m);
    // Skip test if MLIR is not enabled
    if(s.empty())
        return;
    auto mlir_output_with_attrs =
        migraphx::interpolate_string(mlir_output, {{"attrs", get_attrs()}});
    CHECK(encode(s) == encode(mlir_output_with_attrs));
    // A random sequence length can mask every column, making the softmax nan
    auto seq_len = migraphx::fill_argument({migraphx::shape::int32_type, {1}}, 5);
    EXPECT(verify_mlir(m, {{"seq_len", seq_len}}));
}

// Attention sinks return {output, lse} from the fused module
// (find_attention_sinks + find_kv_cache_attention); the module must lower to a
// two-result function with the log/add lse chain intact.
TEST_CASE(kv_cache_attention_sinks_lse)
{
    std::string mlir_output = R"__migraphx__(
module {
  func.func @mlir_unsqueeze_transpose_reshape_unsqueeze_reshape_dot_mul_reshape_unsqueeze_greater_convert_where_reshape_reduce_max_reshape_sub_exp_reshape_reduce_sum_reshape_div_dot_transpose_reshape_log_add(%arg0: !migraphx.shaped<1x2x8x4xf16, 64x32x4x1>, %arg1: !migraphx.shaped<1x4x1x4xf16, 16x4x4x1>, %arg2: !migraphx.shaped<1xsi32, 1>, %arg3: !migraphx.shaped<1x2x8x4xf16, 64x32x4x1>) -> (!migraphx.shaped<1x1x16xf16, 16x16x1>, !migraphx.shaped<1x4x1x1xf16, 4x1x1x1>) attributes ${attrs} {
    %0 = migraphx.literal(dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xsi32>) : <8xsi32, 1>
    %1 = migraphx.literal(dense<0xFC00> : tensor<1xf16>) : <1xf16, 1>
    %2 = migraphx.literal(dense<1.250000e-01> : tensor<1xf16>) : <1xf16, 1>
    %3 = migraphx.reshape %arg0 {dims = [1, 2, 1, 8, 4]} : <1x2x8x4xf16, 64x32x4x1> -> <1x2x1x8x4xf16, 64x32x32x4x1>
    %4 = migraphx.transpose %3 {permutation = [0, 1, 2, 4, 3]} : <1x2x1x8x4xf16, 64x32x32x4x1> -> <1x2x1x4x8xf16, 64x32x32x1x4>
    %5 = migraphx.multibroadcast %4 {out_dyn_dims = [], out_lens = [1, 2, 2, 4, 8]} : <1x2x1x4x8xf16, 64x32x32x1x4> -> <1x2x2x4x8xf16, 64x32x0x1x4>
    %6 = migraphx.reshape %5 {dims = [1, 4, 4, 8]} : <1x2x2x4x8xf16, 64x32x0x1x4> -> <1x4x4x8xf16, 128x32x8x1>
    %7 = migraphx.reshape %arg3 {dims = [1, 2, 1, 8, 4]} : <1x2x8x4xf16, 64x32x4x1> -> <1x2x1x8x4xf16, 64x32x32x4x1>
    %8 = migraphx.multibroadcast %7 {out_dyn_dims = [], out_lens = [1, 2, 2, 8, 4]} : <1x2x1x8x4xf16, 64x32x32x4x1> -> <1x2x2x8x4xf16, 64x32x0x4x1>
    %9 = migraphx.reshape %8 {dims = [1, 4, 8, 4]} : <1x2x2x8x4xf16, 64x32x0x4x1> -> <1x4x8x4xf16, 128x32x4x1>
    %10 = migraphx.dot %arg1, %6 : <1x4x1x4xf16, 16x4x4x1>, <1x4x4x8xf16, 128x32x8x1> -> <1x4x1x8xf16, 32x8x8x1>
    %11 = migraphx.multibroadcast %2 {out_dyn_dims = [], out_lens = [1, 4, 1, 8]} : <1xf16, 1> -> <1x4x1x8xf16, 0x0x0x0>
    %12 = migraphx.mul %10, %11 : <1x4x1x8xf16, 32x8x8x1>, <1x4x1x8xf16, 0x0x0x0> -> <1x4x1x8xf16, 32x8x8x1>
    %13 = migraphx.multibroadcast %0 {out_dyn_dims = [], out_lens = [1, 4, 1, 8]} : <8xsi32, 1> -> <1x4x1x8xsi32, 0x0x0x1>
    %14 = migraphx.reshape %arg2 {dims = [1]} : <1xsi32, 1> -> <1xsi32, 1>
    %15 = migraphx.multibroadcast %14 {out_dyn_dims = [], out_lens = [1, 4]} : <1xsi32, 1> -> <1x4xsi32, 0x0>
    %16 = migraphx.reshape %15 {dims = [1, 4, 1, 1]} : <1x4xsi32, 0x0> -> <1x4x1x1xsi32, 0x0x1x1>
    %17 = migraphx.multibroadcast %16 {out_dyn_dims = [], out_lens = [1, 4, 1, 8]} : <1x4x1x1xsi32, 0x0x1x1> -> <1x4x1x8xsi32, 0x0x1x0>
    %18 = migraphx.greater %13, %17 : <1x4x1x8xsi32, 0x0x0x1>, <1x4x1x8xsi32, 0x0x1x0> -> <1x4x1x8xsi32, 32x8x8x1>
    %19 = migraphx.convert %18 {target_type = 0 : i64} : <1x4x1x8xsi32, 32x8x8x1> to <1x4x1x8xsi8, 32x8x8x1>
    %20 = migraphx.multibroadcast %1 {out_dyn_dims = [], out_lens = [1, 4, 1, 8]} : <1xf16, 1> -> <1x4x1x8xf16, 0x0x0x0>
    %21 = migraphx.where %19, %20, %12 : <1x4x1x8xsi8, 32x8x8x1>, <1x4x1x8xf16, 0x0x0x0>, <1x4x1x8xf16, 32x8x8x1> -> <1x4x1x8xf16, 32x8x8x1>
    %22 = migraphx.reshape %21 {dims = [1, 4, 1, 8]} : <1x4x1x8xf16, 32x8x8x1> -> <1x4x1x8xf16, 32x8x8x1>
    %23 = migraphx.reduce_max %22 {axes = [3]} : <1x4x1x8xf16, 32x8x8x1> -> <1x4x1x1xf16, 4x1x1x1>
    %24 = migraphx.reshape %23 {dims = [1, 4, 1, 1]} : <1x4x1x1xf16, 4x1x1x1> -> <1x4x1x1xf16, 4x1x1x1>
    %25 = migraphx.multibroadcast %24 {out_dyn_dims = [], out_lens = [1, 4, 1, 8]} : <1x4x1x1xf16, 4x1x1x1> -> <1x4x1x8xf16, 4x1x1x0>
    %26 = migraphx.sub %21, %25 : <1x4x1x8xf16, 32x8x8x1>, <1x4x1x8xf16, 4x1x1x0> -> <1x4x1x8xf16, 32x8x8x1>
    %27 = migraphx.exp %26 : <1x4x1x8xf16, 32x8x8x1> -> <1x4x1x8xf16, 32x8x8x1>
    %28 = migraphx.reshape %27 {dims = [1, 4, 1, 8]} : <1x4x1x8xf16, 32x8x8x1> -> <1x4x1x8xf16, 32x8x8x1>
    %29 = migraphx.reduce_sum %28 {axes = [3]} : <1x4x1x8xf16, 32x8x8x1> -> <1x4x1x1xf16, 4x1x1x1>
    %30 = migraphx.reshape %29 {dims = [1, 4, 1, 1]} : <1x4x1x1xf16, 4x1x1x1> -> <1x4x1x1xf16, 4x1x1x1>
    %31 = migraphx.multibroadcast %30 {out_dyn_dims = [], out_lens = [1, 4, 1, 8]} : <1x4x1x1xf16, 4x1x1x1> -> <1x4x1x8xf16, 4x1x1x0>
    %32 = migraphx.div %27, %31 : <1x4x1x8xf16, 32x8x8x1>, <1x4x1x8xf16, 4x1x1x0> -> <1x4x1x8xf16, 32x8x8x1>
    %33 = migraphx.dot %32, %9 : <1x4x1x8xf16, 32x8x8x1>, <1x4x8x4xf16, 128x32x4x1> -> <1x4x1x4xf16, 16x4x4x1>
    %34 = migraphx.transpose %33 {permutation = [0, 2, 1, 3]} : <1x4x1x4xf16, 16x4x4x1> -> <1x1x4x4xf16, 16x4x4x1>
    %35 = migraphx.reshape %34 {dims = [1, 1, 16]} : <1x1x4x4xf16, 16x4x4x1> -> <1x1x16xf16, 16x16x1>
    %36 = migraphx.log %30 : <1x4x1x1xf16, 4x1x1x1> -> <1x4x1x1xf16, 4x1x1x1>
    %37 = migraphx.add %24, %36 : <1x4x1x1xf16, 4x1x1x1>, <1x4x1x1xf16, 4x1x1x1> -> <1x4x1x1xf16, 4x1x1x1>
    return %35, %37 : !migraphx.shaped<1x1x16xf16, 16x16x1>, !migraphx.shaped<1x4x1x1xf16, 4x1x1x1>
  }
}
)__migraphx__";
    auto m                  = make_kv_cache_attention_module(true);
    auto s                  = migraphx::gpu::dump_mlir(m);
    // Skip test if MLIR is not enabled
    if(s.empty())
        return;
    auto mlir_output_with_attrs =
        migraphx::interpolate_string(mlir_output, {{"attrs", get_attrs()}});
    CHECK(encode(s) == encode(mlir_output_with_attrs));
    // A random sequence length can mask every column, making the softmax nan
    auto seq_len = migraphx::fill_argument({migraphx::shape::int32_type, {1}}, 5);
    EXPECT(verify_mlir(m, {{"seq_len", seq_len}}));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
