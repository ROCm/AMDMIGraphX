#include <test.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/program.hpp>
#include <migraphx/gpu/kernel.hpp>
#include <migraphx/gpu/target.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/device_name.hpp>
#include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/gpu/compile_pointwise.hpp>

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
    auto co = migraphx::gpu::compile_pointwise(ctx, {input, input}, "[](auto x) { return x + 1; }");

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

int main(int argc, const char* argv[]) { test::run(argc, argv); }
