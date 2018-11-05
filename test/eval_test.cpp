
#include <migraph/program.hpp>
#include <migraph/iterator_for.hpp>
#include <migraph/instruction.hpp>
#include <sstream>
#include "test.hpp"
#include <basic_ops.hpp>

struct id_target
{
    std::string name() const { return "id"; }
    std::vector<migraph::pass> get_passes(migraph::context&) const { return {}; }
    migraph::context get_context() const { return {}; }
};

struct reverse_pass
{
    std::string name() const { return "reverse_pass"; }

    void apply(migraph::program& p) const
    {
        for(auto ins : migraph::iterator_for(p))
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
    std::vector<migraph::pass> get_passes(migraph::context&) const { return {reverse_pass{}}; }
    migraph::context get_context() const { return {}; }
};

struct double_reverse_target
{
    std::string name() const { return "double_reverse"; }
    std::vector<migraph::pass> get_passes(migraph::context&) const
    {
        return {reverse_pass{}, reverse_pass{}};
    }
    migraph::context get_context() const { return {}; }
};

void literal_test1()
{
    migraph::program p;

    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    p.add_instruction(sum_op{}, one, two);
    auto result = p.eval({});
    EXPECT(result == migraph::literal{3});
    EXPECT(result != migraph::literal{4});
}

void literal_test2()
{
    migraph::program p;

    auto one  = p.add_literal(1);
    auto two  = p.add_literal(2);
    auto sum1 = p.add_instruction(sum_op{}, one, two);
    p.add_instruction(sum_op{}, sum1, two);

    auto result = p.eval({});
    EXPECT(result == migraph::literal{5});
    EXPECT(result != migraph::literal{3});
}

void print_test()
{
    migraph::program p;

    auto x   = p.add_parameter("x", {migraph::shape::int64_type});
    auto two = p.add_literal(2);
    p.add_instruction(sum_op{}, x, two);

    std::stringstream ss;
    ss << p;
    std::string s = ss.str();
    EXPECT(!s.empty());
}

void param_test()
{
    migraph::program p;

    auto x = p.add_parameter("x", {migraph::shape::int64_type});
    auto y = p.add_parameter("y", {migraph::shape::int64_type});

    p.add_instruction(sum_op{}, x, y);
    auto result = p.eval(
        {{"x", migraph::literal{1}.get_argument()}, {"y", migraph::literal{2}.get_argument()}});
    EXPECT(result == migraph::literal{3});
    EXPECT(result != migraph::literal{4});
}

void param_error_test()
{
    migraph::program p;

    auto x = p.add_parameter("x", {migraph::shape::int64_type});
    auto y = p.add_parameter("y", {migraph::shape::int64_type});

    p.add_instruction(sum_op{}, x, y);
    EXPECT(test::throws<migraph::exception>(
        [&] {
            p.eval({{"x", migraph::literal{1}.get_argument()}});
        },
        "Parameter not found: y"));
}

void replace_test()
{
    migraph::program p;

    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    auto sum = p.add_instruction(sum_op{}, one, two);
    p.replace_instruction(sum, minus_op{}, two, one);
    EXPECT(bool{p.validate() == p.end()});

    auto result = p.eval({});
    EXPECT(result == migraph::literal{1});
    EXPECT(result != migraph::literal{3});
}

void replace_ins_test()
{
    migraph::program p;

    auto one   = p.add_literal(1);
    auto two   = p.add_literal(2);
    auto sum   = p.add_instruction(sum_op{}, one, two);
    auto minus = p.add_instruction(minus_op{}, two, one);
    p.replace_instruction(sum, minus);
    EXPECT(bool{p.validate() == p.end()});

    auto result = p.eval({});
    EXPECT(result == migraph::literal{1});
    EXPECT(result != migraph::literal{3});
}

void replace_ins_test2()
{
    migraph::program p;

    auto one   = p.add_literal(1);
    auto two   = p.add_literal(2);
    auto sum   = p.add_instruction(sum_op{}, one, two);
    auto minus = p.add_instruction(minus_op{}, two, one);
    p.add_instruction(pass_op{}, minus);
    p.replace_instruction(two, sum);
    EXPECT(bool{p.validate() == p.end()});

    auto result = p.eval({});
    EXPECT(result == migraph::literal{2});
    EXPECT(result != migraph::literal{3});
}

void insert_replace_test()
{
    migraph::program p;

    auto one  = p.add_literal(1);
    auto two  = p.add_literal(2);
    auto sum1 = p.add_instruction(sum_op{}, one, two);
    p.add_instruction(sum_op{}, sum1, two);

    auto sum0 = p.insert_instruction(sum1, sum_op{}, two, two);
    p.replace_instruction(sum1, minus_op{}, sum0, two);
    EXPECT(bool{p.validate() == p.end()});

    auto result = p.eval({});
    EXPECT(result == migraph::literal{4});
    EXPECT(result != migraph::literal{5});
}

void target_test()
{
    migraph::program p;

    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    p.add_instruction(sum_op{}, one, two);
    p.compile(id_target{});
    auto result = p.eval({});
    EXPECT(result == migraph::literal{3});
    EXPECT(result != migraph::literal{4});
}

void reverse_target_test()
{
    migraph::program p;

    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    p.add_instruction(sum_op{}, two, one);
    p.compile(reverse_target{});
    auto result = p.eval({});
    EXPECT(result == migraph::literal{1});
    EXPECT(result != migraph::literal{4});
}

void double_reverse_target_test()
{
    migraph::program p;

    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    p.add_instruction(sum_op{}, two, one);
    p.compile(double_reverse_target{});
    auto result = p.eval({});
    EXPECT(result == migraph::literal{3});
    EXPECT(result != migraph::literal{4});
}

int main()
{
    literal_test1();
    literal_test2();
    print_test();
    param_test();
    param_error_test();
    replace_test();
    replace_ins_test();
    replace_ins_test2();
    insert_replace_test();
    target_test();
    reverse_target_test();
}
