#include <migraphx/register_op.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/op/convolution.hpp>
#include <migraphx/op/rnn_variable_seq_lens.hpp>
#include <sstream>
#include <string>
#include "test.hpp"

TEST_CASE(load_op)
{
    for(const auto& name : migraphx::get_operators())
    {
        auto op = migraphx::load_op(name);
        CHECK(op.name() == name);
    }
}

TEST_CASE(make_op)
{
    for(const auto& name : migraphx::get_operators())
    {
        auto op = migraphx::load_op(name);
        CHECK(op == migraphx::make_op(name));
    }
}

TEST_CASE(make_op_from_value1)
{
    migraphx::operation x = migraphx::make_op(
        "convolution", {{"padding", {1, 1}}, {"stride", {2, 2}}, {"dilation", {2, 2}}});
    migraphx::operation y = migraphx::op::convolution{{1, 1}, {2, 2}, {2, 2}};
    EXPECT(x == y);
}

TEST_CASE(make_op_from_value2)
{
    migraphx::operation x = migraphx::make_op("convolution", {{"padding", {1, 1}}});
    migraphx::operation y = migraphx::op::convolution{{1, 1}};
    EXPECT(x == y);
}

TEST_CASE(make_rnn_op_from_value)
{
    migraphx::op::rnn_direction dirct = migraphx::op::rnn_direction::reverse;
    migraphx::operation x             = migraphx::make_op(
        "rnn_var_sl_shift_output", {{"output_name", "hidden_states"}, {"direction", dirct}});
    migraphx::operation y = migraphx::op::rnn_var_sl_shift_output{"hidden_states", dirct};
    EXPECT(x == y);
}

TEST_CASE(make_op_invalid_key)
{
    EXPECT(test::throws([] { migraphx::make_op("convolution", {{"paddings", {1, 1}}}); }));
}

TEST_CASE(ops)
{
    auto names = migraphx::get_operators();
    EXPECT(names.size() > 1);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
