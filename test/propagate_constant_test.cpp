#include <migraphx/propagate_constant.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <basic_ops.hpp>
#include <migraphx/make_op.hpp>

#include <test.hpp>

void run_pass(migraphx::module& m)
{
    migraphx::run_passes(m, {migraphx::propagate_constant{}, migraphx::dead_code_elimination{}});
}

TEST_CASE(const_add)
{
    migraphx::module m1;
    auto one = m1.add_literal(1);
    auto two = m1.add_literal(2);
    auto sum = m1.add_instruction(migraphx::make_op("add"), one, two);
    m1.add_instruction(pass_op{}, sum);
    run_pass(m1);

    migraphx::module m2;
    auto total = m2.add_literal(3);
    m2.add_instruction(pass_op{}, total);
    EXPECT(m1 == m2);
}

TEST_CASE(const_add_parameter)
{
    migraphx::module m1;
    auto one = m1.add_parameter("one", {migraphx::shape::int32_type, {1}});
    auto two = m1.add_literal(2);
    auto sum = m1.add_instruction(migraphx::make_op("add"), one, two);
    m1.add_instruction(pass_op{}, sum);
    run_pass(m1);

    migraphx::module m2;
    auto total = m2.add_literal(3);
    m2.add_instruction(pass_op{}, total);
    EXPECT(m1 != m2);
}

TEST_CASE(const_multiadd)
{
    migraphx::module m1;
    auto one  = m1.add_literal(1);
    auto two  = m1.add_literal(2);
    auto sum1 = m1.add_instruction(migraphx::make_op("add"), one, two);
    auto sum2 = m1.add_instruction(migraphx::make_op("add"), sum1, two);
    m1.add_instruction(pass_op{}, sum2);
    run_pass(m1);

    migraphx::module m2;
    auto total = m2.add_literal(5);
    m2.add_instruction(pass_op{}, total);
    EXPECT(m1 == m2);
}

TEST_CASE(const_add_mul)
{
    migraphx::module m1;
    auto one  = m1.add_literal(1);
    auto two  = m1.add_literal(2);
    auto mul  = m1.add_instruction(migraphx::make_op("mul"), two, two);
    auto sum1 = m1.add_instruction(migraphx::make_op("add"), one, mul);
    auto sum2 = m1.add_instruction(migraphx::make_op("add"), sum1, two);
    m1.add_instruction(pass_op{}, sum2);
    run_pass(m1);

    migraphx::module m2;
    auto total = m2.add_literal(7);
    m2.add_instruction(pass_op{}, total);
    EXPECT(m1 == m2);
}

TEST_CASE(const_add_scalar)
{
    migraphx::module m1;
    auto one = m1.add_instruction(migraphx::make_op("scalar", {{"scalar_bcst_dims", {2, 2}}}),
                                  m1.add_literal(1));
    auto two = m1.add_instruction(migraphx::make_op("scalar", {{"scalar_bcst_dims", {2, 2}}}),
                                  m1.add_literal(2));
    auto sum = m1.add_instruction(migraphx::make_op("add"), one, two);
    m1.add_instruction(pass_op{}, sum);
    run_pass(m1);

    migraphx::module m2;
    auto total =
        m2.add_literal(migraphx::literal{{migraphx::shape::int32_type, {2, 2}}, {3, 3, 3, 3}});
    m2.add_instruction(pass_op{}, total);
    EXPECT(m1 == m2);
}

TEST_CASE(const_scalar)
{
    migraphx::module m1;
    {
        auto one = m1.add_instruction(migraphx::make_op("scalar", {{"scalar_bcst_dims", {2, 2}}}),
                                      m1.add_literal(1));
        m1.add_instruction(pass_op{}, one);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto one = m2.add_instruction(migraphx::make_op("scalar", {{"scalar_bcst_dims", {2, 2}}}),
                                      m2.add_literal(1));
        m2.add_instruction(pass_op{}, one);
    }
    EXPECT(m1 == m2);
}

TEST_CASE(const_dot)
{
    migraphx::module m1;
    {
        migraphx::shape s{migraphx::shape::float_type, {2, 2}};
        std::vector<float> vec = {1.0f, 2.0f, 1.0f, 2.0f};

        auto l  = m1.add_literal(migraphx::literal(s, vec));
        auto dl = m1.add_instruction(migraphx::make_op("dot"), l, l);
        auto x  = m1.add_parameter("x", s);
        auto r  = m1.add_instruction(migraphx::make_op("add"), dl, x);
        m1.add_return({r});
    }

    run_pass(m1);

    migraphx::module m2;
    {
        migraphx::shape s{migraphx::shape::float_type, {2, 2}};
        std::vector<float> vec = {3.0f, 6.0f, 3.0f, 6.0f};

        auto x = m2.add_parameter("x", s);
        auto l = m2.add_literal(migraphx::literal(s, vec));
        auto r = m2.add_instruction(migraphx::make_op("add"), l, x);
        m2.add_return({r});
    }
    EXPECT(m1 == m2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
