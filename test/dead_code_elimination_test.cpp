#include <migraph/dead_code_elimination.hpp>
#include <basic_ops.hpp>
#include <test.hpp>

struct dce_target
{
    std::string name() const { return "dce"; }
    std::vector<migraph::pass> get_passes(migraph::context&) const
    {
        return {migraph::dead_code_elimination{}};
    }
    migraph::context get_context() const { return {}; }
};

void simple_test()
{
    migraph::program p;

    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    p.add_instruction(sum_op{}, one, two);
    auto count = std::distance(p.begin(), p.end());
    p.compile(dce_target{});
    EXPECT(std::distance(p.begin(), p.end()) == count);
    auto result = p.eval({});
    EXPECT(result == migraph::literal{3});
    EXPECT(result != migraph::literal{4});
}

void duplicate_test1()
{
    migraph::program p;

    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    p.add_instruction(sum_op{}, one, two);
    p.add_instruction(sum_op{}, one, two);
    auto count = std::distance(p.begin(), p.end());
    p.compile(dce_target{});
    EXPECT(std::distance(p.begin(), p.end()) == (count - 1));
    auto result = p.eval({});
    EXPECT(result == migraph::literal{3});
    EXPECT(result != migraph::literal{4});
}

void duplicate_test2()
{
    migraph::program p;

    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    p.add_instruction(sum_op{}, one, two);
    p.add_instruction(minus_op{}, one, two);
    p.add_instruction(sum_op{}, one, two);
    auto count = std::distance(p.begin(), p.end());
    p.compile(dce_target{});
    EXPECT(std::distance(p.begin(), p.end()) == (count - 2));
    auto result = p.eval({});
    EXPECT(result == migraph::literal{3});
    EXPECT(result != migraph::literal{4});
}

int main()
{
    simple_test();
    duplicate_test1();
    duplicate_test2();
}
