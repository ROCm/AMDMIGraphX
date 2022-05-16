#include <algorithm>
#include <hip/hip_runtime.h>
#include <migraphx/migraphx.h>
#include <miopen/miopen.h>
#include <migraphx/migraphx.hpp> // MIGraphX's C++ API
#include <numeric>

#define MIGRAPHX_MIOPEN_ASSERT(x) (assert((x) == miopenStatusSuccess))
#define MIGRAPHX_HIP_ASSERT(x) (assert((x) == hipSuccess))

inline miopenTensorDescriptor_t make_miopen_tensor(const migraphx::shape& s, bool pack = false)
{
    // TODO: normalize_standard shape for scalar 
    miopenTensorDescriptor_t t;
    MIGRAPHX_MIOPEN_ASSERT(miopenCreateTensorDescriptor(&t));
    // Convert to ints
    std::vector<int> lens(s.lengths().begin(), s.lengths().end());
    std::vector<int> strides(s.strides().begin(), s.strides().end());
    miopenDataType_t d;
    if(s.type() == migraphx_shape_float_type)
        d = miopenFloat;
    else if(s.type() == migraphx_shape_half_type)
        d = miopenHalf;
    else if(s.type() == migraphx_shape_int32_type)
        d = miopenInt32;
    else if(s.type() == migraphx_shape_int8_type)
    {
        if(pack)
        {
            // update the lens and corresponding strides
            d          = miopenInt8x4;
            lens[1]    = ((lens[1] + 3) / 4) * 4;
            strides[0] = strides[1] * lens[1];
        }
        else
        {
            d = miopenInt8;
        }
    }
    else
    {
        throw("MAKE_TENSOR: unsupported type");
    }
    miopenSetTensorDescriptor(t, d, s.lengths().size(), lens.data(), strides.data());
    return t;
}

// create MIOpen stream handle
inline auto make_miopen_handle(migraphx::context& ctx) {
    MIGRAPHX_HIP_ASSERT(hipSetDevice(0));
    auto* stream  = ctx.get_queue<hipStream_t>();
    miopenHandle_t out;
    MIGRAPHX_MIOPEN_ASSERT(miopenCreateWithStream(&out, stream));
    return out;
}

inline auto make_activation_descriptor(miopenActivationMode_t mode, double alpha=0, double beta=0, double gamma=0) {
    miopenActivationDescriptor_t ad;
    MIGRAPHX_MIOPEN_ASSERT(miopenCreateActivationDescriptor(&ad));
    miopenSetActivationDescriptor(ad, mode, alpha, beta, gamma);
    return ad;
}

struct abs_custom_op final : migraphx::experimental_custom_op_base
{
    virtual std::string name() const override { return "abs_custom_op"; }
    virtual migraphx::argument
    compute(migraphx::context ctx, migraphx::shape output_shape, migraphx::arguments args) const override
    {
        float alpha = 1;
        float beta  = 0;
        auto x_desc = make_miopen_tensor(args[0].get_shape());
        auto y_desc = make_miopen_tensor(output_shape);
        // create MIOpen stream handle
        auto miopen_handle = make_miopen_handle(ctx);
        // make miopen activation descriptor 
        auto ad = make_activation_descriptor(miopenActivationABS, 0, 0, 0);
        miopenActivationForward(miopen_handle,
                                ad,
                                &alpha,
                                x_desc,
                                args[0].data(),
                                &beta,
                                y_desc,
                                args[1].data());
        return args[1];
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
    abs_custom_op abs_op;
    migraphx::register_experimental_custom_op(abs_op);
    migraphx::program p;
    migraphx::shape s{migraphx_shape_float_type, {32, 256}};
    migraphx::module m = p.get_main_module();
    auto x             = m.add_parameter("x", s);
    // do allocation for the output buffer for the custom_kernel
    auto alloc = m.add_allocation(s);
    auto custom_kernel =
        m.add_instruction(migraphx::operation("abs_custom_op"), {x, alloc});
    m.add_return({custom_kernel});
    migraphx::compile_options options;
    // set offload copy to true for GPUs
    options.set_offload_copy();
    p.compile(migraphx::target("gpu"), options);
    p.print();
    migraphx::program_parameters pp;
    std::vector<float> x_data(s.bytes() / sizeof(s.type()));
    std::iota(x_data.begin(), x_data.end(), -(32 * 256));
    pp.add("x", migraphx::argument(s, x_data.data()));
    auto results                       = p.eval(pp);
    auto result                        = results[0];
    std::vector<float> expected_result = x_data;
    std::transform(expected_result.begin(),
                   expected_result.end(),
                   expected_result.begin(),
                   [](auto i) { return std::abs(i); });
    assert(bool{result == migraphx::argument(s, expected_result.data())});
    return 0;
}
