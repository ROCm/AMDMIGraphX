#include <migraphx/migraphx.h>
#include <migraphx/migraphx.hpp>
#include "test.hpp"

struct simple_custom_op final : migraphx::experimental_custom_op_base
{
    virtual std::string name() const override { return "simple_custom_op"; }
    virtual migraphx::argument
    compute(migraphx::context, migraphx::shape, migraphx::arguments inputs) const override
    {
        return inputs[0];
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

int main(int argc, const char* argv[]) { test::run(argc, argv); }
