//  MIGraphX C++ API
#include <algorithm>
#include <hip/hip_runtime.h>
#include <migraphx/migraphx.hpp>
#include <numeric>

#define MIGRAPHX_HIP_ASSERT(x) (assert((x) == hipSuccess))
/*
 * Square each element in the array A and write to array C.
 */
template <typename T>
__global__ void vector_square(T* C_d, const T* A_d, size_t N)
{
    size_t offset = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    size_t stride = hipBlockDim_x * hipGridDim_x;

    for(size_t i = offset; i < N; i += stride)
    {
        C_d[i] = A_d[i] * A_d[i];
    }
}

struct square_custom_op final : migraphx::experimental_custom_op_base
{
    virtual std::string name() const override { return "square_custom_op"; }
    virtual migraphx::argument
    compute(migraphx::context ctx, migraphx::shape, migraphx::arguments inputs) const override
    {
        float* d_output;
        auto* h_output   = reinterpret_cast<float*>(inputs[1].data());
        auto input_bytes = inputs[0].get_shape().bytes();
        MIGRAPHX_HIP_ASSERT(hipSetDevice(0));
        MIGRAPHX_HIP_ASSERT(hipMalloc(&d_output, input_bytes));
        MIGRAPHX_HIP_ASSERT(hipMemcpyAsync(d_output,
                                           inputs[0].data(),
                                           input_bytes,
                                           hipMemcpyHostToDevice,
                                           ctx.get_queue<hipStream_t>()));
        const unsigned blocks          = 512;
        const unsigned threadsPerBlock = 256;
        hipLaunchKernelGGL(
            vector_square, dim3(blocks), dim3(threadsPerBlock), 0, 0, d_output, d_output, 8192);
        MIGRAPHX_HIP_ASSERT(hipMemcpy(h_output, d_output, input_bytes, hipMemcpyDeviceToHost));
        MIGRAPHX_HIP_ASSERT(hipFree(d_output));
        return inputs[1];
    }
    virtual migraphx::shape compute_shape(migraphx::shapes inputs) const override
    {
        assert(inputs.size() == 2);
        assert(bool{inputs[0] == inputs[1]});
        return inputs.back();
    }
};

int main(int argc, const char* argv[])
{
    square_custom_op square_op;
    migraphx::register_experimental_custom_op(square_op);
    migraphx::program p;
    migraphx::shape s{migraphx_shape_float_type, {32, 256}};
    migraphx::module m = p.get_main_module();
    auto x             = m.add_parameter("x", s);
    auto alloc         = m.add_instruction(
        migraphx::operation(
            "allocate", R"({"shape":{"type":"float_type","lens":[32, 256], "strides":[256, 1]}})"),
        {});
    auto custom_kernel = m.add_instruction(migraphx::operation("square_custom_op"), {x, alloc});
    m.add_return({custom_kernel});
    p.compile(migraphx::target("gpu"));
    migraphx::program_parameters pp;
    std::vector<float> x_data(32 * 256);
    std::iota(x_data.begin(), x_data.end(), 0);
    std::vector<float> ret_data(32 * 256, -1);
    pp.add("x", migraphx::argument(s, x_data.data()));
    pp.add("main:#output_0", migraphx::argument(s, ret_data.data()));
    auto results                       = p.eval(pp);
    auto result                        = results[0];
    std::vector<float> expected_result = x_data;
    std::transform(expected_result.begin(),
                   expected_result.end(),
                   expected_result.begin(),
                   [](auto i) { return std::pow(i, 2); });
    assert(bool{result == migraphx::argument(s, expected_result.data())});
    auto* result_ptr = reinterpret_cast<float*>(result.data());
    std::vector<float> result_vec(result_ptr, result_ptr + 8192);
    assert(std::equal(ret_data.begin(), ret_data.end(), result_vec.begin(), result_vec.end()));
    return 0;
}
