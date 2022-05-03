#include <migraphx/migraphx.h>
#include <migraphx/migraphx.hpp>
#include "test.hpp"

TEST_CASE(add_op)
{
    auto add_op = migraphx::operation("add");
    EXPECT(add_op.name() == "add");
}

TEST_CASE(reduce_mean_without_quotes)
{
    auto rm = migraphx::operation("reduce_mean", "{axes : [1, 2, 3, 4]}");
    EXPECT(rm.name() == "reduce_mean");
}

TEST_CASE(reduce_mean)
{
    auto rm = migraphx::operation("reduce_mean", "{\"axes\" : [1, 2, 3, 4]}");
    EXPECT(rm.name() == "reduce_mean");
}

TEST_CASE(reduce_mean_with_format)
{
    auto rm = migraphx::operation("reduce_mean", "{axes : [%i, %i, %i, %i]}", 1, 2, 3, 4);
    EXPECT(rm.name() == "reduce_mean");
}

// TODO: Enable this after Khalique's PR is merged.
// TEST_CASE(allocate_with_nested_json_str)
// {
//     auto alloc = migraphx::operation(
//         "allocate", "{\"shape\":{\"type\":\"float_type\",\"lens\":[3, 3], \"strides\":[3, 1]}}");
//     EXPECT(alloc.name() == "allocate");
// }

struct simple_custom_op final : migraphx::experimental_custom_op_base
{
    virtual std::string name() const override { return "simple_custom_op"; }
    virtual migraphx::argument compute(migraphx::context,
                                       migraphx::shape,
                                       migraphx::arguments inputs) const override
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
