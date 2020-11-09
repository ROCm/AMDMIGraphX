#include <test.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/gpu/kernel.hpp>
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

int main(int argc, const char* argv[]) { test::run(argc, argv); }
