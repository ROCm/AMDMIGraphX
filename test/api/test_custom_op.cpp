
#include <hip/hip_runtime_api.h>
#include <migraphx/migraphx.h>
#include <migraphx/migraphx.hpp>
#include "test.hpp"

struct simple_custom_op final : migraphx::experimental_custom_op_base
{
    virtual std::string name() const override { return "simple_custom_op"; }
    virtual migraphx::argument compute(migraphx::context ctx,
                                       migraphx::shape output_shape,
                                       migraphx::arguments inputs) const override
    {
        float* d_output;
        float* h_output{new float[9]};
        auto input_bytes  = inputs[0].get_shape().bytes();
        auto copy_bytes   = input_bytes / 2;
        auto memset_bytes = input_bytes - copy_bytes;
        auto res          = hipSetDevice(0);
        EXPECT(res == hipSuccess);
        res = hipMalloc(&d_output, input_bytes);
        EXPECT(res == hipSuccess);
        res = hipMemcpyAsync(d_output,
                             inputs[0].data(),
                             input_bytes,
                             hipMemcpyHostToDevice,
                             ctx.get_queue<hipStream_t>());
        EXPECT(res == hipSuccess);
        res = hipMemset(d_output, 0, memset_bytes);
        EXPECT(res == hipSuccess);
        res = hipMemcpy(h_output, d_output, input_bytes, hipMemcpyDeviceToHost);
        EXPECT(res == hipSuccess);
        return migraphx::argument(output_shape, h_output);
    }

    virtual migraphx::shape compute_shape(migraphx::shapes inputs) const override
    {
        return inputs.front();
    }
};

TEST_CASE(register_custom_op)
{
    simple_custom_op simple_op;
    migraphx::register_experimental_custom_op(simple_op);

    auto op = migraphx::operation("simple_custom_op");
    EXPECT(op.name() == "simple_custom_op");
}

TEST_CASE(run_simple_custom_op)
{
    simple_custom_op simple_op;
    migraphx::register_experimental_custom_op(simple_op);
    migraphx::program p;
    migraphx::shape s{migraphx_shape_float_type, {3, 3}};
    migraphx::module m = p.get_main_module();
    auto x             = m.add_parameter("x", s);
    auto custom_kernel = m.add_instruction(migraphx::operation("simple_custom_op"), {x});
    m.add_return({custom_kernel});
    p.compile(migraphx::target("gpu"));
    migraphx::program_parameters pp;
    std::vector<float> x_data(9, 1);
    pp.add("x", migraphx::argument(s, x_data.data()));
    auto results = p.eval(pp);
    auto result  = results[0];
    std::vector<float> expected_result(9, 0);
    for(size_t i = 4; i < 9; i++)
    {
        expected_result[i] = 1;
    }
    EXPECT(bool{result == migraphx::argument(s, expected_result.data())});
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
