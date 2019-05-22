#include <migraphx/horizontal_fusion.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/program.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/instruction.hpp>
#include <basic_ops.hpp>
#include <test.hpp>

struct horizontal_fusion_target
{
    std::string name() const { return "horizontal fusion"; }
    std::vector<migraphx::pass> get_passes(migraphx::context&) const
    {
        return {migraphx::horizontal_fusion{}};
    }
    migraphx::context get_context() const { return {}; }
};

TEST_CASE(test1)
{
    migraphx::program p;
    int size = 192 * 35 * 35;
    std::vector<float> a(size);
    for(auto i = 0; i < size; i++)
        a[i] = 1.0 * i;
    size = 64 * 192;
    std::vector<float> b(size);
    for(auto i = 0; i < size; i++)
        b[i] = 0.5 * i;
    size = 64 * 35 * 35;
    std::vector<float> c(size);
    for(auto i = 0; i < size; i++)
        c[i] = 0.1 * i;
    size = 48 * 192;
    std::vector<float> d(size);
    for(auto i = 0; i < size; i++)
        d[i] = 0.2 * i;
    size = 48 * 35 * 35;
    std::vector<float> e(size);
    for(auto i = 0; i < size; i++)
        e[i] = 0.3 * i;

    migraphx::shape a_shape{migraphx::shape::float_type, {1, 192, 35, 35}};
    auto l1 = p.add_literal(migraphx::literal{a_shape, a});
    auto p1 = p.add_instruction(migraphx::op::pooling{}, l1);
    migraphx::shape b_shape{migraphx::shape::float_type, {64, 192, 1, 1}};
    auto l2 = p.add_literal(migraphx::literal{b_shape, b});
    auto c1 = p.add_instruction(migraphx::op::convolution{}, p1, l2);
    migraphx::shape c_shape{migraphx::shape::float_type, {1, 64, 35, 35}};
    auto l3 = p.add_literal(migraphx::literal{c_shape, c});
    auto a1 = p.add_instruction(migraphx::op::add{}, c1, l3);
    auto r1 = p.add_instruction(migraphx::op::relu{}, a1);
    migraphx::shape d_shape{migraphx::shape::float_type, {48, 192, 1, 1}};
    auto l4 = p.add_literal(migraphx::literal{d_shape, d});
    auto c2 = p.add_instruction(migraphx::op::convolution{}, p1, l4);
    migraphx::shape e_shape{migraphx::shape::float_type, {1, 48, 35, 35}};
    auto l5 = p.add_literal(migraphx::literal{e_shape, e});
    auto a2 = p.add_instruction(migraphx::op::add{}, c2, l5);
    auto r2 = p.add_instruction(migraphx::op::relu{}, a2);
    auto p2 = p.add_instruction(migraphx::op::pooling{}, r2);
    auto l6 = p.add_literal(migraphx::literal{b_shape, b});
    auto c3 = p.add_instruction(migraphx::op::convolution{}, p1, l6);
    auto l7 = p.add_literal(migraphx::literal{c_shape, c});
    auto a3 = p.add_instruction(migraphx::op::add{}, c3, l7);
    auto r3 = p.add_instruction(migraphx::op::relu{}, a3);
    auto p3 = p.add_instruction(migraphx::op::pooling{}, r3);
    p.add_instruction(migraphx::op::concat{1}, r1, p2, p3);
    auto count = std::distance(p.begin(), p.end());
    p.compile(horizontal_fusion_target{});

    EXPECT(std::distance(p.begin(), p.end()) == count - 6);
    EXPECT(std::count_if(p.begin(), p.end(), [](const migraphx::instruction& ins) {
               return ins.name() == "convolution";
           }) == 1);
    EXPECT(std::count_if(p.begin(), p.end(), [](const migraphx::instruction& ins) {
               return ins.name() == "relu";
           }) == 1);
    EXPECT(std::count_if(p.begin(), p.end(), [](const migraphx::instruction& ins) {
               return ins.name() == "add";
           }) == 1);
    EXPECT(std::count_if(p.begin(), p.end(), [](const migraphx::instruction& ins) {
               return ins.name() == "split";
           }) == 2);
    EXPECT(std::count_if(p.begin(), p.end(), [](const migraphx::instruction& ins) {
               return ins.name() == "load";
           }) == 2);
    migraphx::instruction& instr =
        (*std::find_if(p.begin(), p.end(), [](const migraphx::instruction& ins) {
            return ins.name() == "convolution";
        }));
    auto lens = instr.get_shape().lens();
    EXPECT((lens.at(0) == 1) && (lens.at(1) == 176) && (lens.at(2) == 35) && (lens.at(3) == 35));
    lens = instr.inputs().at(1)->get_shape().lens();
    EXPECT((lens.at(0) == 176) && (lens.at(1) == 192) && (lens.at(2) == 1) && (lens.at(3) == 1));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
