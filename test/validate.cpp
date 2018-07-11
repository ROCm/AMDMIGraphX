#include <migraph/program.hpp>
#include <basic_ops.hpp>
#include <test.hpp>

void simple_test()
{
    migraph::program p;

    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    p.add_instruction(sum_op{}, one, two);
    std::cout << std::distance(p.begin(), p.validate()) << std::endl;
    EXPECT(bool{p.validate() == p.end()});
    auto result = p.eval({});
    EXPECT(result == migraph::literal{3});
    EXPECT(result != migraph::literal{4});
}

void out_of_order()
{
    migraph::program p;

    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    auto ins = p.add_instruction(sum_op{}, one, two);
    p.move_instruction(two, p.end());
    EXPECT(bool{p.validate() == ins});
}

int main()
{
    simple_test();
    out_of_order();
}
