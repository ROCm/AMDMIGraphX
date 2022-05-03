
#include <hip/hip_runtime_api.h>
#include <migraphx/migraphx.h>
#include <migraphx/migraphx.hpp>
#include "test.hpp"

#define HIP_ASSERT(x) (EXPECT((x) == hipSuccess))
struct simple_custom_op final : migraphx::experimental_custom_op_base
{
    virtual std::string name() const override { return "simple_custom_op"; }
    virtual migraphx::argument compute(migraphx::context ctx,
                                       migraphx::shape output_shape,
                                       migraphx::arguments inputs) const override
    {
        // sets first half size_bytes of the input 0, and rest of the half bytes are copied.
        float* d_output;
        float* h_output{new float[12]};
        auto input_bytes = inputs[0].get_shape().bytes();
        auto copy_bytes  = input_bytes / 2;
        HIP_ASSERT(hipSetDevice(0));
        HIP_ASSERT(hipMalloc(&d_output, input_bytes));
        HIP_ASSERT(hipMemcpyAsync(d_output,
                                  inputs[0].data(),
                                  input_bytes,
                                  hipMemcpyHostToDevice,
                                  ctx.get_queue<hipStream_t>()));
        HIP_ASSERT(hipMemset(d_output, 0, copy_bytes));
        HIP_ASSERT(hipMemcpy(h_output, d_output, input_bytes, hipMemcpyDeviceToHost));
        HIP_ASSERT(hipFree(d_output));
        return {output_shape, h_output};
    }

    virtual migraphx::shape compute_shape(migraphx::shapes inputs) const override
    {
        return inputs.front();
    }
};

TEST_CASE(run_simple_custom_op)
{
    simple_custom_op simple_op;
    migraphx::register_experimental_custom_op(simple_op);
    migraphx::program p;
    migraphx::shape s{migraphx_shape_float_type, {4, 3}};
    migraphx::module m = p.get_main_module();
    auto x             = m.add_parameter("x", s);
    auto custom_kernel = m.add_instruction(migraphx::operation("simple_custom_op"), {x});
    m.add_return({custom_kernel});
    p.compile(migraphx::target("gpu"));
    migraphx::program_parameters pp;
    std::vector<float> x_data(12, 1);
    pp.add("x", migraphx::argument(s, x_data.data()));
    auto results = p.eval(pp);
    auto result  = results[0];
    std::vector<float> expected_result(12, 0);
    std::fill(expected_result.begin() + 6, expected_result.end(), 1);
    EXPECT(bool{result == migraphx::argument(s, expected_result.data())});
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
