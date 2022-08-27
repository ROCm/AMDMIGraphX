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
#include <test.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/program.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/gpu/kernel.hpp>
#include <migraphx/gpu/target.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/device_name.hpp>
#include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/gpu/compiler.hpp>

// NOLINTNEXTLINE
const std::string write_2s = R"__migraphx__(
#include <hip/hip_runtime.h>

extern "C" {
__global__ void write(int8_t* data) 
{
    int num = threadIdx.x + blockDim.x * blockIdx.x;
    data[num] = 2;
}
    
}

int main() {}

)__migraphx__";

// NOLINTNEXTLINE
const std::string add_2s_binary = R"__migraphx__(
#include <hip/hip_runtime.h>

extern "C" {
__global__ void add_2(std::int8_t* x, std::int8_t* y) 
{
    int num = threadIdx.x + blockDim.x * blockIdx.x;
    y[num] = x[num] + 2;
}
    
}

int main() {}

)__migraphx__";

// NOLINTNEXTLINE
const std::string simple_pointwise_increment = R"__migraphx__(
#include <migraphx/kernels/index.hpp>
#include <args.hpp>

using namespace migraphx;

extern "C" {
__global__ void kernel(void* x, void* y) 
{
    make_tensors()(x, y)([](auto xt, auto yt) __device__ {
        auto idx = make_index();
        const auto stride = idx.nglobal();
        for(index_int i = idx.global; i < xt.get_shape().elements(); i += stride)
        {
            yt[i] = xt[i] + 1;
        }
    });
}
    
}

int main() {}

)__migraphx__";

// NOLINTNEXTLINE
const std::string check_define = R"__migraphx__(

#ifndef __DEFINE__
#error __DEFINE__ was not defined
#endif

int main() {}

)__migraphx__";

// NOLINTNEXTLINE
const std::string unused_param = R"__migraphx__(

extern "C" {
__global__ void kernel(void* x, void* y) 
{}
}

int main() {}

)__migraphx__";

// NOLINTNEXTLINE
const std::string incorrect_program = R"__migraphx__(

extern "C" {
__global__ void kernel(void* x) 
{
    x += y;
}
}

int main() {}

)__migraphx__";

// NOLINTNEXTLINE
const std::string math_template = R"__migraphx__(
#include <migraphx/kernels/pointwise.hpp>
#include <migraphx/kernels/math.hpp>

extern "C" {
__global__ void kernel(${type}* p) 
{
    auto x = *p;
    *p = migraphx::implicit_conversion(migraphx::${invoke});

}
}

int main() {}

)__migraphx__";

migraphx::src_file make_src_file(const std::string& name, const std::string& content)
{
    return {name, std::make_pair(content.data(), content.data() + content.size())};
}

TEST_CASE(simple_compile_hip)
{
    auto binaries = migraphx::gpu::compile_hip_src(
        {make_src_file("main.cpp", write_2s)}, "", migraphx::gpu::get_device_name());
    EXPECT(binaries.size() == 1);

    migraphx::argument input{{migraphx::shape::int8_type, {5}}};
    auto ginput = migraphx::gpu::to_gpu(input);
    migraphx::gpu::kernel k{binaries.front(), "write"};
    k.launch(nullptr, input.get_shape().elements(), 1024)(ginput.cast<std::int8_t>());
    auto output = migraphx::gpu::from_gpu(ginput);

    EXPECT(output != input);
    auto data = output.get<std::int8_t>();
    EXPECT(migraphx::all_of(data, [](auto x) { return x == 2; }));
}

auto check_target(const std::string& arch)
{
    auto define  = "__" + arch + "__";
    auto content = migraphx::replace_string(check_define, "__DEFINE__", define);
    return migraphx::gpu::compile_hip_src({make_src_file("main.cpp", content)}, "", arch);
}

TEST_CASE(compile_target)
{
    EXPECT(not check_target("gfx900").empty());
    EXPECT(not check_target("gfx906").empty());
}

TEST_CASE(compile_errors)
{
    EXPECT(test::throws([&] {
        migraphx::gpu::compile_hip_src(
            {make_src_file("main.cpp", incorrect_program)}, "", migraphx::gpu::get_device_name());
    }));
}

TEST_CASE(compile_warnings)
{
    auto compile = [](const std::string& params) {
        return migraphx::gpu::compile_hip_src(
            {make_src_file("main.cpp", unused_param)}, params, migraphx::gpu::get_device_name());
    };

    EXPECT(not compile("").empty());
    EXPECT(not compile("-Wunused-parameter -Wno-error").empty());
    EXPECT(not compile("-Wno-unused-parameter -Werror").empty());
    EXPECT(test::throws([&] { compile("-Werror=unused-parameter"); }));
    EXPECT(test::throws([&] { compile("-Wunused-parameter -Werror"); }));
}

TEST_CASE(code_object_hip)
{
    auto binaries = migraphx::gpu::compile_hip_src(
        {make_src_file("main.cpp", add_2s_binary)}, "", migraphx::gpu::get_device_name());
    EXPECT(binaries.size() == 1);

    migraphx::shape input{migraphx::shape::int8_type, {5}};

    std::vector<migraphx::shape> expected_inputs = {input, input};
    auto co                                      = migraphx::make_op("gpu::code_object",
                                {{"code_object", migraphx::value::binary{binaries.front()}},
                                 {"symbol_name", "add_2"},
                                 {"global", input.elements()},
                                 {"local", 1024},
                                 {"expected_inputs", migraphx::to_value(expected_inputs)},
                                 {"output", migraphx::to_value(input)}});

    migraphx::program p;
    auto* mm            = p.get_main_module();
    auto input_literal  = migraphx::generate_literal(input);
    auto output_literal = migraphx::transform(input_literal, [](auto x) { return x + 2; });
    auto x              = mm->add_literal(input_literal);
    auto y              = mm->add_parameter("output", input);
    mm->add_instruction(co, x, y);
    migraphx::compile_options options;
    p.compile(migraphx::gpu::target{}, options);

    auto result =
        migraphx::gpu::from_gpu(p.eval({{"output", migraphx::gpu::allocate_gpu(input)}}).front());

    EXPECT(result == output_literal.get_argument());
}

TEST_CASE(compile_code_object_hip)
{
    migraphx::shape input{migraphx::shape::float_type, {5, 2}};
    migraphx::gpu::hip_compile_options options;
    options.global = 256 * 1024;
    options.local  = 1024;
    options.inputs = {input, input};
    options.output = input;

    auto co = migraphx::gpu::compile_hip_code_object(simple_pointwise_increment, options);

    migraphx::program p;
    auto* mm            = p.get_main_module();
    auto input_literal  = migraphx::generate_literal(input);
    auto output_literal = migraphx::transform(input_literal, [](auto x) { return x + 1; });
    auto x              = mm->add_literal(input_literal);
    auto y              = mm->add_parameter("output", input);
    mm->add_instruction(co, x, y);
    p.compile(migraphx::gpu::target{}, migraphx::compile_options{});

    auto result =
        migraphx::gpu::from_gpu(p.eval({{"output", migraphx::gpu::allocate_gpu(input)}}).front());

    EXPECT(result == output_literal.get_argument());
}

TEST_CASE(compile_pointwise)
{
    migraphx::shape input{migraphx::shape::float_type, {5, 2}};

    migraphx::gpu::context ctx;
    auto co = migraphx::gpu::compile_op(
        "pointwise", ctx, {input, input}, {{"lambda", "[](auto x) { return x + 1; }"}});

    migraphx::program p;
    auto* mm            = p.get_main_module();
    auto input_literal  = migraphx::generate_literal(input);
    auto output_literal = migraphx::transform(input_literal, [](auto x) { return x + 1; });
    auto x              = mm->add_literal(input_literal);
    auto y              = mm->add_parameter("output", input);
    mm->add_instruction(co, x, y);
    p.compile(migraphx::gpu::target{}, migraphx::compile_options{});

    auto result =
        migraphx::gpu::from_gpu(p.eval({{"output", migraphx::gpu::allocate_gpu(input)}}).front());

    EXPECT(result == output_literal.get_argument());
}

TEST_CASE(compile_math)
{
    std::vector<std::string> math_invoke = {
        // clang-format off
        "abs(x)",
        "acos(x)",
        "acosh(x)",
        "asin(x)",
        "asinh(x)",
        "atan(x)",
        "atanh(x)",
        "ceil(x)",
        "cos(x)",
        "cosh(x)",
        "erf(x)",
        "exp(x)",
        "floor(x)",
        "isnan(x)",
        "log(x)",
        "max(x, x)",
        "min(x, x)",
        "pow(x, 0)",
        "pow(x, x)",
        "round(x)",
        "rsqrt(x)",
        "sin(x)",
        "sinh(x)",
        "sqrt(x)",
        "tan(x)",
        "tanh(x)",
        "where(true, x, x)",
        // clang-format on
    };
    std::vector<std::string> data_types;
    auto vec_sizes = {2, 4, 6};
    for(auto&& t : migraphx::shape::types())
    {
        if(contains({migraphx::shape::bool_type, migraphx::shape::tuple_type}, t))
            continue;
        auto name = migraphx::shape::cpp_type(t);
        if(t == migraphx::shape::half_type)
            name.insert(0, "migraphx::");
        data_types.push_back(name);
        migraphx::transform(vec_sizes, std::back_inserter(data_types), [&](auto i) {
            return "migraphx::vec<" + name + ", " + std::to_string(i) + ">";
        });
    }
    migraphx::shape input{migraphx::shape::float_type, {5, 2}};
    migraphx::gpu::hip_compile_options options;
    options.global = 1024;
    options.local  = 1024;
    options.inputs = {input};
    options.output = input;
    migraphx::par_for(math_invoke.size() * data_types.size(), 1, [&](auto i) {
        const auto& t      = data_types[i % data_types.size()];
        const auto& invoke = math_invoke[i / data_types.size()];
        auto src = migraphx::interpolate_string(math_template, {{"type", t}, {"invoke", invoke}});
        auto co  = migraphx::gpu::compile_hip_code_object(src, options);
        (void)co;
    });
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
