
#include <migraphx/program.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/instruction.hpp>
#include <sstream>
#include "test.hpp"
#include <basic_ops.hpp>

struct id_target
{
    std::string name() const { return "id"; }
    std::vector<migraphx::pass> get_passes(migraphx::context&) const { return {}; }
    migraphx::context get_context() const { return {}; }
};

struct reverse_pass
{
    std::string name() const { return "reverse_pass"; }

    void apply(migraphx::program& p) const
    {
        for(auto ins : migraphx::iterator_for(p))
        {
            if(ins->name() == "sum")
            {
                p.replace_instruction(ins, minus_op{}, ins->inputs());
            }
            else if(ins->name() == "minus")
            {
                p.replace_instruction(ins, sum_op{}, ins->inputs());
            }
        }
    }
};

struct reverse_target
{
    std::string name() const { return "reverse"; }
    std::vector<migraphx::pass> get_passes(migraphx::context&) const { return {reverse_pass{}}; }
    migraphx::context get_context() const { return {}; }
};

struct double_reverse_target
{
    std::string name() const { return "double_reverse"; }
    std::vector<migraphx::pass> get_passes(migraphx::context&) const
    {
        return {reverse_pass{}, reverse_pass{}};
    }
    migraphx::context get_context() const { return {}; }
};

TEST_CASE(literal_test1)
{
    migraphx::program p;

    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    p.add_instruction(sum_op{}, one, two);
    auto result = p.eval({});
    EXPECT(result == migraphx::literal{3});
    EXPECT(result != migraphx::literal{4});
}

TEST_CASE(literal_test2)
{
    migraphx::program p;

    auto one  = p.add_literal(1);
    auto two  = p.add_literal(2);
    auto sum1 = p.add_instruction(sum_op{}, one, two);
    p.add_instruction(sum_op{}, sum1, two);

    auto result = p.eval({});
    EXPECT(result == migraphx::literal{5});
    EXPECT(result != migraphx::literal{3});
}

TEST_CASE(print_test)
{
    migraphx::program p;

    auto x   = p.add_parameter("x", {migraphx::shape::int64_type});
    auto two = p.add_literal(2);
    p.add_instruction(sum_op{}, x, two);

    std::stringstream ss;
    ss << p;
    std::string s = ss.str();
    EXPECT(!s.empty());
}

TEST_CASE(param_test)
{
    migraphx::program p;

    auto x = p.add_parameter("x", {migraphx::shape::int64_type});
    auto y = p.add_parameter("y", {migraphx::shape::int64_type});

    p.add_instruction(sum_op{}, x, y);
    auto result = p.eval(
        {{"x", migraphx::literal{1}.get_argument()}, {"y", migraphx::literal{2}.get_argument()}});
    EXPECT(result == migraphx::literal{3});
    EXPECT(result != migraphx::literal{4});
}

TEST_CASE(param_error_test)
{
    migraphx::program p;

    auto x = p.add_parameter("x", {migraphx::shape::int64_type});
    auto y = p.add_parameter("y", {migraphx::shape::int64_type});

    p.add_instruction(sum_op{}, x, y);
    EXPECT(test::throws<migraphx::exception>(
        [&] {
            p.eval({{"x", migraphx::literal{1}.get_argument()}});
        },
        "Parameter not found: y"));
}

TEST_CASE(replace_test)
{
    migraphx::program p;

    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    auto sum = p.add_instruction(sum_op{}, one, two);
    p.replace_instruction(sum, minus_op{}, two, one);
    EXPECT(bool{p.validate() == p.end()});

    auto result = p.eval({});
    EXPECT(result == migraphx::literal{1});
    EXPECT(result != migraphx::literal{3});
}

TEST_CASE(replace_ins_test)
{
    migraphx::program p;

    auto one   = p.add_literal(1);
    auto two   = p.add_literal(2);
    auto sum   = p.add_instruction(sum_op{}, one, two);
    auto minus = p.add_instruction(minus_op{}, two, one);
    p.replace_instruction(sum, minus);
    EXPECT(bool{p.validate() == p.end()});

    auto result = p.eval({});
    EXPECT(result == migraphx::literal{1});
    EXPECT(result != migraphx::literal{3});
}

TEST_CASE(replace_ins_test2)
{
    migraphx::program p;

    auto one   = p.add_literal(1);
    auto two   = p.add_literal(2);
    auto sum   = p.add_instruction(sum_op{}, one, two);
    auto minus = p.add_instruction(minus_op{}, two, one);
    p.add_instruction(pass_op{}, minus);
    p.replace_instruction(two, sum);
    EXPECT(bool{p.validate() == p.end()});

    auto result = p.eval({});
    EXPECT(result == migraphx::literal{2});
    EXPECT(result != migraphx::literal{3});
}

TEST_CASE(insert_replace_test)
{
    migraphx::program p;

    auto one  = p.add_literal(1);
    auto two  = p.add_literal(2);
    auto sum1 = p.add_instruction(sum_op{}, one, two);
    p.add_instruction(sum_op{}, sum1, two);

    auto sum0 = p.insert_instruction(sum1, sum_op{}, two, two);
    p.replace_instruction(sum1, minus_op{}, sum0, two);
    EXPECT(bool{p.validate() == p.end()});

    auto result = p.eval({});
    EXPECT(result == migraphx::literal{4});
    EXPECT(result != migraphx::literal{5});
}

TEST_CASE(target_test)
{
    migraphx::program p;

    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    p.add_instruction(sum_op{}, one, two);
    p.compile(id_target{});
    auto result = p.eval({});
    EXPECT(result == migraphx::literal{3});
    EXPECT(result != migraphx::literal{4});
}

TEST_CASE(reverse_target_test)
{
    migraphx::program p;

    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    p.add_instruction(sum_op{}, two, one);
    p.compile(reverse_target{});
    auto result = p.eval({});
    EXPECT(result == migraphx::literal{1});
    EXPECT(result != migraphx::literal{4});
}

TEST_CASE(double_reverse_target_test)
{
    migraphx::program p;

    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    p.add_instruction(sum_op{}, two, one);
    p.compile(double_reverse_target{});
    auto result = p.eval({});
    EXPECT(result == migraphx::literal{3});
    EXPECT(result != migraphx::literal{4});
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
