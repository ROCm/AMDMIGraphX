
#include <migraph/program.hpp>
#include <migraph/argument.hpp>
#include <migraph/shape.hpp>
#include <sstream>
#include "test.hpp"

struct sum_op
{
    std::string name() const { return "sum"; }
    migraph::argument compute(migraph::context&, migraph::shape, std::vector<migraph::argument> args) const
    {
        migraph::argument result;
        if(args.size() != 2)
            MIGRAPH_THROW("Wrong args");
        if(args[0].get_shape() != args[1].get_shape())
            MIGRAPH_THROW("Wrong args");
        if(args[0].get_shape().lens().size() != 1)
            MIGRAPH_THROW("Wrong args");
        if(args[0].get_shape().lens().front() != 1)
            MIGRAPH_THROW("Wrong args");

        args[0].visit_at([&](auto x) {
            args[1].visit_at([&](auto y) { result = migraph::literal{x + y}.get_argument(); });
        });
        return result;
    }

    migraph::shape compute_shape(std::vector<migraph::shape> inputs) const
    {
        if(inputs.size() != 2)
            MIGRAPH_THROW("Wrong inputs");
        return inputs.front();
    }
};

struct minus_op
{
    std::string name() const { return "minus"; }
    migraph::argument compute(migraph::context&, migraph::shape, std::vector<migraph::argument> args) const
    {
        migraph::argument result;
        if(args.size() != 2)
            MIGRAPH_THROW("Wrong args");
        if(args[0].get_shape() != args[1].get_shape())
            MIGRAPH_THROW("Wrong args");
        if(args[0].get_shape().lens().size() != 1)
            MIGRAPH_THROW("Wrong args");
        if(args[0].get_shape().lens().front() != 1)
            MIGRAPH_THROW("Wrong args");

        args[0].visit_at([&](auto x) {
            args[1].visit_at([&](auto y) { result = migraph::literal{x - y}.get_argument(); });
        });
        return result;
    }

    migraph::shape compute_shape(std::vector<migraph::shape> inputs) const
    {
        if(inputs.size() != 2)
            MIGRAPH_THROW("Wrong inputs");
        return inputs.front();
    }
};

struct id_target
{
    std::string name() const { return "id"; }
    void apply(migraph::program&) const {}
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
    auto result =
        p.eval({{"x", migraph::literal{1}.get_argument()}, {"y", migraph::literal{2}.get_argument()}});
    EXPECT(result == migraph::literal{3});
    EXPECT(result != migraph::literal{4});
}

void replace_test()
{
    migraph::program p;

    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    auto sum = p.add_instruction(sum_op{}, one, two);
    p.replace_instruction(sum, minus_op{}, two, one);

    auto result = p.eval({});
    EXPECT(result == migraph::literal{1});
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

int main()
{
    literal_test1();
    literal_test2();
    print_test();
    param_test();
    replace_test();
    insert_replace_test();
    target_test();
}
