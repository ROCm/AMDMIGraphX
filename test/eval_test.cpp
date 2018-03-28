
#include <rtg/program.hpp>
#include <rtg/argument.hpp>
#include <rtg/shape.hpp>
#include "test.hpp"

int main() {

    rtg::program p;
    p.add_operator("sum", 
        [](std::vector<rtg::argument> args) {
            rtg::argument result;
            if(args.size() != 2) throw "Wrong args";
            if(args[0].s != args[1].s) throw "Wrong args";
            if(args[0].s.lens().size() != 1) throw "Wrong args";
            if(args[0].s.lens().front() != 1) throw "Wrong args";

            args[0].visit([&](auto x) {
                args[1].visit([&](auto y) {
                    result = rtg::literal{x + y}.get_argument();
                });
            });
            return result;
        },
        [](std::vector<rtg::shape> inputs) {
            if(inputs.size() != 2) throw "Wrong inputs";
            return inputs.front();
        }
    );

    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    p.add_instruction("sum", one, two);
    EXPECT(p.eval() == rtg::literal{3});
}
