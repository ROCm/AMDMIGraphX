#include <algorithm>
#include <hip/hip_runtime.h>
#include <rocblas.h>
#include <migraphx/migraphx.h>
#include <migraphx/migraphx.hpp> // MIGraphX's C++ API
#include <numeric>

#define MIGRAPHX_ROCBLAS_ASSERT(x) (assert((x) == rocblas_status::rocblas_status_success))
#define MIGRAPHX_HIP_ASSERT(x) (assert((x) == hipSuccess))

rocblas_handle create_rocblas_handle_ptr()
{
    rocblas_handle handle;
    MIGRAPHX_ROCBLAS_ASSERT(rocblas_create_handle(&handle));
    return rocblas_handle{handle};
}

rocblas_handle create_rocblas_handle_ptr(migraphx::context& ctx)
{
    MIGRAPHX_HIP_ASSERT(hipSetDevice(0));
    rocblas_handle rb = create_rocblas_handle_ptr();
    auto* stream = ctx.get_queue<hipStream_t>();
    MIGRAPHX_ROCBLAS_ASSERT(rocblas_set_stream(rb, stream));
    return rb;
}

struct sscal_custom_op final : migraphx::experimental_custom_op_base
{
    virtual std::string name() const override { return "sscal_custom_op"; }
    virtual migraphx::argument compute(migraphx::context ctx,
                                       migraphx::shape output_shape,
                                       migraphx::arguments args) const override
    {
        // create MIOpen stream handle
        auto rocblas_handle = create_rocblas_handle_ptr(ctx);
        rocblas_int n = args[1].get_shape().lengths()[0];
        float* alpha = reinterpret_cast<float*>(args[0].data());
        float* vec_ptr = reinterpret_cast<float*>(args[1].data());
        // make miopen activation descriptor
        MIGRAPHX_ROCBLAS_ASSERT(rocblas_sscal(rocblas_handle, n, alpha, vec_ptr, 1));
        return args[1];
    }

    virtual migraphx::shape compute_shape(migraphx::shapes inputs) const override
    {
        assert(inputs.size() == 2);
        assert(inputs[0].lengths().size() == 1);
        assert(inputs[0].lengths()[0] == 1);
        assert(inputs[1].lengths().size() == 1);
        return inputs.back();
    }
};

int main(int argc, const char* argv[])
{
    // computes ReLU(neg(x) * scale)
    sscal_custom_op sscal_op;
    migraphx::register_experimental_custom_op(sscal_op);
    migraphx::program p;
    migraphx::shape x_shape{migraphx_shape_float_type, {8192}};
    migraphx::shape scale_shape{migraphx_shape_float_type, {1}};
    migraphx::module m = p.get_main_module();
    auto x             = m.add_parameter("x", x_shape);
    auto scale = m.add_parameter("scale", scale_shape);
    auto neg_ins       = m.add_instruction(migraphx::operation("neg"), {x});
    // do allocation for the output buffer for the custom_kernel
    auto custom_kernel = m.add_instruction(migraphx::operation("sscal_custom_op"), {scale, neg_ins});
    auto relu_ins      = m.add_instruction(migraphx::operation("relu"), {custom_kernel});
    m.add_return({relu_ins});

    migraphx::compile_options options;
    // set offload copy to true for GPUs
    options.set_offload_copy();
    p.compile(migraphx::target("gpu"), options);
    migraphx::program_parameters pp;
    std::vector<float> x_data(x_shape.bytes() / sizeof(x_shape.type()));
    std::vector<float> scale_data{-1};
    std::iota(x_data.begin(), x_data.end(), 0);
    pp.add("x", migraphx::argument(x_shape, x_data.data()));
    pp.add("scale", migraphx::argument(scale_shape, scale_data.data()));
    auto results                       = p.eval(pp);
    auto result                        = results[0];
    std::vector<float> expected_result = x_data;
    assert(bool{result == migraphx::argument(x_shape, expected_result.data())});
    return 0;
}
