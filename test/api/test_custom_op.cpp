#include <migraphx/migraphx.h>
#include <migraphx/migraphx.hpp>
#include "test.hpp"

struct my_custom_op final : migraphx::experimental_custom_op_base
{
    virtual std::string name() const override { return "my_custom_op"; }
    virtual migraphx::shape compute_shape(migraphx::shapes inputs) const override
    {
        return inputs.front();
    }
};

TEST_CASE(construct_custom_op)
{
    my_custom_op op;
    migraphx::experimental_custom_op cop{op};
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
