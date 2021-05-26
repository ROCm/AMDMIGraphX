#include <migraphx/dom_info.hpp>
#include <migraphx/program.hpp>
#include <basic_ops.hpp>
#include <test.hpp>

TEST_CASE(dom1)
{
    migraphx::module mm;
    auto ins1 = mm.add_parameter("entry", {migraphx::shape::float_type});
    auto ins2 = mm.add_instruction(pass_op{}, ins1);
    auto ins3 = mm.add_instruction(pass_op{}, ins2);
    auto ins4 = mm.add_instruction(pass_op{}, ins2);
    auto ins5 = mm.add_instruction(pass_op{}, ins3, ins4);
    auto ins6 = mm.add_instruction(pass_op{}, ins2);

    auto dom = migraphx::compute_dominator(mm);
    EXPECT(dom.strictly_dominate(ins1, ins2));
    EXPECT(dom.strictly_dominate(ins2, ins3));
    EXPECT(dom.strictly_dominate(ins2, ins4));
    EXPECT(dom.strictly_dominate(ins2, ins5));
    EXPECT(dom.strictly_dominate(ins2, ins6));

    EXPECT(not dom.strictly_dominate(ins3, ins6));
    EXPECT(not dom.strictly_dominate(ins4, ins6));
    EXPECT(not dom.strictly_dominate(ins3, ins5));
    EXPECT(not dom.strictly_dominate(ins4, ins5));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
