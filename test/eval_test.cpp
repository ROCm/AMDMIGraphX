
#include <rtg/program.hpp>
#include <rtg/argument.hpp>
#include <rtg/shape.hpp>
#include <sstream>
#include "test.hpp"

struct sum_op
{
    std::string name() const { return "sum"; }
    rtg::argument compute(std::vector<rtg::argument> args) const
    {
        rtg::argument result;
        if(args.size() != 2)
            RTG_THROW("Wrong args");
        if(args[0].get_shape() != args[1].get_shape())
            RTG_THROW("Wrong args");
        if(args[0].get_shape().lens().size() != 1)
            RTG_THROW("Wrong args");
        if(args[0].get_shape().lens().front() != 1)
            RTG_THROW("Wrong args");

        args[0].visit_at([&](auto x) {
            args[1].visit_at([&](auto y) { result = rtg::literal{x + y}.get_argument(); });
        });
        return result;
    }

    rtg::shape compute_shape(std::vector<rtg::shape> inputs) const
    {
        if(inputs.size() != 2)
            RTG_THROW("Wrong inputs");
        return inputs.front();
    }
};

struct minus_op
{
    std::string name() const { return "minus"; }
    rtg::argument compute(std::vector<rtg::argument> args) const
    {
        rtg::argument result;
        if(args.size() != 2)
            RTG_THROW("Wrong args");
        if(args[0].get_shape() != args[1].get_shape())
            RTG_THROW("Wrong args");
        if(args[0].get_shape().lens().size() != 1)
            RTG_THROW("Wrong args");
        if(args[0].get_shape().lens().front() != 1)
            RTG_THROW("Wrong args");

        args[0].visit_at([&](auto x) {
            args[1].visit_at([&](auto y) { result = rtg::literal{x - y}.get_argument(); });
        });
        return result;
    }

    rtg::shape compute_shape(std::vector<rtg::shape> inputs) const
    {
        if(inputs.size() != 2)
            RTG_THROW("Wrong inputs");
        return inputs.front();
    }
};

struct id_target
{
    std::string name() const { return "id"; }
    void apply(rtg::program&) const {}
};

void literal_test1()
{
    rtg::program p;

    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    p.add_instruction(sum_op{}, one, two);
    auto result = p.eval({});
    EXPECT(result == rtg::literal{3});
    EXPECT(result != rtg::literal{4});
}

void literal_test2()
{
    rtg::program p;

    auto one  = p.add_literal(1);
    auto two  = p.add_literal(2);
    auto sum1 = p.add_instruction(sum_op{}, one, two);
    p.add_instruction(sum_op{}, sum1, two);

    auto result = p.eval({});
    EXPECT(result == rtg::literal{5});
    EXPECT(result != rtg::literal{3});
}

void print_test()
{
    rtg::program p;

    auto x = p.add_parameter("x", {rtg::shape::int64_type});
    auto two = p.add_literal(2);
    p.add_instruction(sum_op{}, x, two);

    std::stringstream ss;
    ss << p;
    std::string s = ss.str();
    EXPECT(!s.empty());
}

void param_test()
{
    rtg::program p;

    auto x = p.add_parameter("x", {rtg::shape::int64_type});
    auto y = p.add_parameter("y", {rtg::shape::int64_type});

    p.add_instruction(sum_op{}, x, y);
    auto result =
        p.eval({{"x", rtg::literal{1}.get_argument()}, {"y", rtg::literal{2}.get_argument()}});
    EXPECT(result == rtg::literal{3});
    EXPECT(result != rtg::literal{4});
}

void replace_test()
{
    rtg::program p;

    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    auto sum = p.add_instruction(sum_op{}, one, two);
    p.replace_instruction(sum, minus_op{}, two, one);

    auto result = p.eval({});
    EXPECT(result == rtg::literal{1});
    EXPECT(result != rtg::literal{3});
}

void insert_replace_test()
{
    rtg::program p;

    auto one  = p.add_literal(1);
    auto two  = p.add_literal(2);
    auto sum1 = p.add_instruction(sum_op{}, one, two);
    p.add_instruction(sum_op{}, sum1, two);

    auto sum0 = p.insert_instruction(sum1, sum_op{}, two, two);
    p.replace_instruction(sum1, minus_op{}, sum0, two);

    auto result = p.eval({});
    EXPECT(result == rtg::literal{4});
    EXPECT(result != rtg::literal{5});
}


void target_test()
{
    rtg::program p;

    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    p.add_instruction(sum_op{}, one, two);
    p.compile(id_target{});
    auto result = p.eval({});
    EXPECT(result == rtg::literal{3});
    EXPECT(result != rtg::literal{4});
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
