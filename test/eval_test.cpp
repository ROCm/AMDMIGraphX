
#include <rtg/program.hpp>
#include <rtg/argument.hpp>
#include <rtg/shape.hpp>
#include "test.hpp"


struct sum_op
{
    std::string name() const
    {
        return "sum";
    }
    rtg::argument compute(std::vector<rtg::argument> args) const
    {
        rtg::argument result;
        if(args.size() != 2) throw "Wrong args";
        if(args[0].get_shape() != args[1].get_shape()) throw "Wrong args";
        if(args[0].get_shape().lens().size() != 1) throw "Wrong args";
        if(args[0].get_shape().lens().front() != 1) throw "Wrong args";

        args[0].visit_at([&](auto x) {
            args[1].visit_at([&](auto y) {
                result = rtg::literal{x + y}.get_argument();
            });
        });
        return result;
    }

    rtg::shape compute_shape(std::vector<rtg::shape> inputs) const
    {
        if(inputs.size() != 2) throw "Wrong inputs";
        return inputs.front();
    }
};

void literal_test() {
    rtg::program p;

    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    p.add_instruction(sum_op{}, one, two);
    auto result = p.eval({});
    EXPECT(result == rtg::literal{3});
    EXPECT(result != rtg::literal{4});
}

void param_test() {
    rtg::program p;

    auto x = p.add_parameter("x", {rtg::shape::int64_type});
    auto y = p.add_parameter("y", {rtg::shape::int64_type});

    p.add_instruction(sum_op{}, x, y);
    auto result = p.eval({
        {"x", rtg::literal{1}.get_argument()}, 
        {"y", rtg::literal{2}.get_argument()}
    });
    EXPECT(result == rtg::literal{3});
    EXPECT(result != rtg::literal{4});
}

int main() {
    literal_test();
    param_test();

}
