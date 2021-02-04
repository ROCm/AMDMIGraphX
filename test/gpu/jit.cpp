#include <test.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/program.hpp>
#include <migraphx/gpu/kernel.hpp>
#include <migraphx/gpu/target.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/gpu/compile_hip.hpp>

// NOLINTNEXTLINE
const std::string write_2s = R"migraphx(
#include <hip/hip_runtime.h>

extern "C" {
__global__ void write(int* data) 
{
    int num = threadIdx.x + blockDim.x * blockIdx.x;
    data[num] = 2;
}
    
}

int main() {}

)migraphx";

// NOLINTNEXTLINE
const std::string add_2s_binary = R"migraphx(
#include <hip/hip_runtime.h>

extern "C" {
__global__ void add_2(std::int32_t* x, std::int32_t* y) 
{
    int num = threadIdx.x + blockDim.x * blockIdx.x;
    y[num] = x[num] + 2;
}
    
}

int main() {}

)migraphx";

migraphx::gpu::src_file make_src_file(const std::string& name, const std::string& content)
{
    return {name, std::make_pair(content.data(), content.data() + content.size())};
}

std::string get_device_name()
{
    hipDeviceProp_t props{};
    int device;
    EXPECT(hipGetDevice(&device) == hipSuccess);
    EXPECT(hipGetDeviceProperties(&props, device) == hipSuccess);
    return "gfx" + std::to_string(props.gcnArch);
}

TEST_CASE(simple_compile_hip)
{
    auto binaries = migraphx::gpu::compile_hip_src(
        {make_src_file("main.cpp", write_2s)}, "", get_device_name());
    EXPECT(binaries.size() == 1);

    migraphx::argument input{{migraphx::shape::int32_type, {5}}};
    auto ginput = migraphx::gpu::to_gpu(input);
    migraphx::gpu::kernel k{binaries.front(), "write"};
    k.launch(nullptr, input.get_shape().elements(), 1024)(ginput.cast<int>());
    auto output = migraphx::gpu::from_gpu(ginput);

    EXPECT(output != input);
    auto data = output.get<int>();
    EXPECT(migraphx::all_of(data, [](auto x) { return x == 2; }));
}

TEST_CASE(code_object_hip)
{
    auto binaries = migraphx::gpu::compile_hip_src(
        {make_src_file("main.cpp", add_2s_binary)}, "", get_device_name());
    EXPECT(binaries.size() == 1);

    migraphx::shape input{migraphx::shape::int32_type, {5}};

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
    auto y              = mm->add_instruction(
        migraphx::make_op("hip::allocate", {{"shape", migraphx::to_value(input)}}));
    mm->add_instruction(co, x, y);
    migraphx::compile_options options;
    p.compile(migraphx::gpu::target{}, options);

    auto result = migraphx::gpu::from_gpu(p.eval({}).front());

    EXPECT(result == output_literal.get_argument());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
