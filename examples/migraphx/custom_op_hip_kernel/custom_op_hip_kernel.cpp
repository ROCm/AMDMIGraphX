#include <algorithm>
#include <hip/hip_runtime.h>
#include <migraphx/migraphx.hpp> // MIGraphX's C++ API
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
        // if compile options has offload_copy = true then, parameters and outputs will be
        // automatically copied to and from GPUs' memory. Here assume that `inputs` arguments are
        // already in the GPU, so no need to do Malloc, Free or Memcpy last element in the `inputs`
        // is output argument, so it should be returned from compute method.
        auto* input_buffer  = reinterpret_cast<float*>(inputs[0].data());
        auto* output_buffer = reinterpret_cast<float*>(inputs[1].data());
        size_t n_elements   = inputs[0].get_shape().bytes() / sizeof(inputs[0].get_shape().type());
        MIGRAPHX_HIP_ASSERT(hipSetDevice(0));
        const unsigned blocks            = 512;
        const unsigned threads_per_block = 256;
        hipLaunchKernelGGL(vector_square,
                           dim3(blocks),
                           dim3(threads_per_block),
                           0,
                           ctx.get_queue<hipStream_t>(),
                           output_buffer,
                           input_buffer,
                           n_elements);
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
    auto neg_ins       = m.add_instruction(migraphx::operation("neg"), x);
    // do allocation for the output buffer for the custom_kernel
    auto alloc = m.add_allocation(s);
    auto custom_kernel =
        m.add_instruction(migraphx::operation("square_custom_op"), {neg_ins, alloc});
    auto relu_ins = m.add_instruction(migraphx::operation("relu"), {custom_kernel});
    m.add_return({relu_ins});
    migraphx::compile_options options;
    // set offload copy to true for GPUs
    options.set_offload_copy();
    p.compile(migraphx::target("gpu"), options);
    migraphx::program_parameters pp;
    std::vector<float> x_data(s.bytes() / sizeof(s.type()));
    std::iota(x_data.begin(), x_data.end(), 0);
    pp.add("x", migraphx::argument(s, x_data.data()));
    auto results                       = p.eval(pp);
    auto result                        = results[0];
    std::vector<float> expected_result = x_data;
    std::transform(expected_result.begin(),
                   expected_result.end(),
                   expected_result.begin(),
                   [](auto i) { return std::pow(i, 2); });
    assert(bool{result == migraphx::argument(s, expected_result.data())});
    return 0;
}
