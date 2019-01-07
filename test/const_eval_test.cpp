#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <sstream>
#include "test.hpp"
#include <basic_ops.hpp>

struct sum_cf_op
{
    std::string name() const { return "sum_cf"; }
    migraphx::argument compute(const migraphx::shape&, std::vector<migraphx::argument> args) const
    {
        migraphx::argument result;
        if(args.size() != 2)
            MIGRAPHX_THROW("Wrong args");
        if(args[0].get_shape() != args[1].get_shape())
            MIGRAPHX_THROW("Wrong args");
        if(args[0].get_shape().lens().size() != 1)
            MIGRAPHX_THROW("Wrong args");
        if(args[0].get_shape().lens().front() != 1)
            MIGRAPHX_THROW("Wrong args");

        args[0].visit_at([&](auto x) {
            args[1].visit_at([&](auto y) { result = migraphx::literal{x + y}.get_argument(); });
        });
        return result;
    }

    migraphx::shape compute_shape(std::vector<migraphx::shape> inputs) const
    {
        if(inputs.size() != 2)
            MIGRAPHX_THROW("Wrong inputs");
        return inputs.front();
    }
};

TEST_CASE(literal_test)
{
    migraphx::program p;
    auto lit = p.add_literal(1);
    CHECK(lit->eval() == migraphx::literal{1});
}

TEST_CASE(param_test)
{
    migraphx::program p;
    auto lit = p.add_parameter("param", migraphx::shape{migraphx::shape::float_type, {1}});
    CHECK(lit->eval().empty());
}

TEST_CASE(op_test1)
{
    migraphx::program p;
    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    auto sum = p.add_instruction(sum_cf_op{}, one, two);
    CHECK(sum->eval() == migraphx::literal{3});
}

TEST_CASE(op_test2)
{
    migraphx::program p;
    auto x = p.add_parameter("param", migraphx::shape{migraphx::shape::float_type, {1}});
    ;
    auto two = p.add_literal(2);
    auto sum = p.add_instruction(sum_cf_op{}, x, two);
    CHECK(sum->eval().empty());
}

TEST_CASE(op_test3)
{
    migraphx::program p;
    auto one  = p.add_literal(1);
    auto two  = p.add_literal(2);
    auto sum1 = p.add_instruction(sum_op{}, one, two);
    auto sum2 = p.add_instruction(sum_cf_op{}, sum1, two);
    CHECK(sum2->eval().empty());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
