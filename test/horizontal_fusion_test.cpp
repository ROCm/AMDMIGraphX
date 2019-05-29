#include <migraphx/horizontal_fusion.hpp>
#include <migraphx/propagate_constant.hpp>
#include <migraphx/dead_code_elimination.hpp>
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
        return {migraphx::horizontal_fusion{},
                migraphx::propagate_constant{},
                migraphx::dead_code_elimination{}};
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

    EXPECT(std::count_if(p.begin(), p.end(), [](const migraphx::instruction& ins) {
               return ins.name() == "concat";
           }) == 1);

    migraphx::instruction& instr =
        (*std::find_if(p.begin(), p.end(), [](const migraphx::instruction& ins) {
            return ins.name() == "convolution";
        }));
    auto lens = instr.get_shape().lens();
    EXPECT((lens.at(0) == 1) && (lens.at(1) == 176) && (lens.at(2) == 35) && (lens.at(3) == 35));
    lens = instr.inputs().at(1)->get_shape().lens();
    EXPECT((lens.at(0) == 176) && (lens.at(1) == 192) && (lens.at(2) == 1) && (lens.at(3) == 1));
}

// case for non-unique outputs, i.e., the last layer feeds into the same consumer.  Due to the
// limitation of current IR design, the last layer can't be fused.
TEST_CASE(test2)
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
    auto l6 = p.add_literal(migraphx::literal{b_shape, b});
    auto c3 = p.add_instruction(migraphx::op::convolution{}, p1, l6);
    auto l7 = p.add_literal(migraphx::literal{c_shape, c});
    auto a3 = p.add_instruction(migraphx::op::add{}, c3, l7);
    auto r3 = p.add_instruction(migraphx::op::relu{}, a3);
    p.add_instruction(migraphx::op::concat{1}, r1, r2, r3);
    p.compile(horizontal_fusion_target{});
    EXPECT(std::count_if(p.begin(), p.end(), [](const migraphx::instruction& ins) {
               return ins.name() == "convolution";
           }) == 1);
    EXPECT(std::count_if(p.begin(), p.end(), [](const migraphx::instruction& ins) {
               return ins.name() == "relu";
           }) == 3);

    EXPECT(std::count_if(p.begin(), p.end(), [](const migraphx::instruction& ins) {
               return ins.name() == "add";
           }) == 1);

    EXPECT(std::count_if(p.begin(), p.end(), [](const migraphx::instruction& ins) {
               return ins.name() == "split";
           }) == 1);
}

// case for breaking split at the end.
TEST_CASE(test3)
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
    p.add_instruction(migraphx::op::concat{1}, p2, p3, r1);
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

    EXPECT(std::count_if(p.begin(), p.end(), [](const migraphx::instruction& ins) {
               return ins.name() == "concat";
           }) == 1);

    migraphx::instruction& instr =
        (*std::find_if(p.begin(), p.end(), [](const migraphx::instruction& ins) {
            return ins.name() == "convolution";
        }));
    auto lens = instr.get_shape().lens();
    EXPECT((lens.at(0) == 1) && (lens.at(1) == 176) && (lens.at(2) == 35) && (lens.at(3) == 35));
    lens = instr.inputs().at(1)->get_shape().lens();
    EXPECT((lens.at(0) == 176) && (lens.at(1) == 192) && (lens.at(2) == 1) && (lens.at(3) == 1));
}

// case where literal does not directly feed into layers to be fused.

TEST_CASE(test4)
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
    auto l1   = p.add_literal(migraphx::literal{a_shape, a});
    auto l1_c = p.add_instruction(migraphx::op::contiguous{}, l1);
    auto p1   = p.add_instruction(migraphx::op::pooling{}, l1_c);
    migraphx::shape b_shape{migraphx::shape::float_type, {64, 192, 1, 1}};
    auto l2   = p.add_literal(migraphx::literal{b_shape, b});
    auto l2_c = p.add_instruction(migraphx::op::contiguous{}, l2);
    auto c1   = p.add_instruction(migraphx::op::convolution{}, p1, l2_c);
    migraphx::shape c_shape{migraphx::shape::float_type, {1, 64, 35, 35}};
    auto l3   = p.add_literal(migraphx::literal{c_shape, c});
    auto l3_c = p.add_instruction(migraphx::op::contiguous{}, l3);
    auto a1   = p.add_instruction(migraphx::op::add{}, c1, l3_c);
    auto r1   = p.add_instruction(migraphx::op::relu{}, a1);
    migraphx::shape d_shape{migraphx::shape::float_type, {48, 192, 1, 1}};
    auto l4   = p.add_literal(migraphx::literal{d_shape, d});
    auto l4_c = p.add_instruction(migraphx::op::contiguous{}, l4);
    auto c2   = p.add_instruction(migraphx::op::convolution{}, p1, l4_c);
    migraphx::shape e_shape{migraphx::shape::float_type, {1, 48, 35, 35}};
    auto l5   = p.add_literal(migraphx::literal{e_shape, e});
    auto l5_c = p.add_instruction(migraphx::op::contiguous{}, l5);
    auto a2   = p.add_instruction(migraphx::op::add{}, c2, l5_c);
    auto r2   = p.add_instruction(migraphx::op::relu{}, a2);
    auto p2   = p.add_instruction(migraphx::op::pooling{}, r2);
    auto l6   = p.add_literal(migraphx::literal{b_shape, b});
    auto l6_c = p.add_instruction(migraphx::op::contiguous{}, l6);
    auto c3   = p.add_instruction(migraphx::op::convolution{}, p1, l6_c);
    auto l7   = p.add_literal(migraphx::literal{c_shape, c});
    auto l7_c = p.add_instruction(migraphx::op::contiguous{}, l7);
    auto a3   = p.add_instruction(migraphx::op::add{}, c3, l7_c);
    auto r3   = p.add_instruction(migraphx::op::relu{}, a3);
    auto p3   = p.add_instruction(migraphx::op::pooling{}, r3);
    p.add_instruction(migraphx::op::concat{1}, r1, p2, p3);
    p.compile(horizontal_fusion_target{});
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

    EXPECT(std::count_if(p.begin(), p.end(), [](const migraphx::instruction& ins) {
               return ins.name() == "concat";
           }) == 1);
}

// case where different paddings prevents this transformation.
TEST_CASE(test5)
{
    migraphx::program p;
    int size = 384 * 8 * 8;
    std::vector<float> a(size);
    for(auto i = 0; i < size; i++)
        a[i] = 1.0 * i;
    size = 384 * 384 * 1 * 3;
    std::vector<float> b(size);
    for(auto i = 0; i < size; i++)
        b[i] = 0.5 * i;
    size = 384 * 384 * 3;
    std::vector<float> c(size);
    for(auto i = 0; i < size; i++)
        c[i] = 0.1 * i;

    migraphx::shape a_shape{migraphx::shape::float_type, {1, 384, 8, 8}};
    auto l1 = p.add_literal(migraphx::literal{a_shape, a});
    migraphx::shape b_shape{migraphx::shape::float_type, {384, 384, 1, 3}};
    auto l2 = p.add_literal(migraphx::literal{b_shape, b});
    auto c1 = p.add_instruction(migraphx::op::convolution{{0, 1}, {1, 1}, {1, 1}}, l1, l2);
    migraphx::shape c_shape{migraphx::shape::float_type, {384, 384, 3, 1}};
    auto l3 = p.add_literal(migraphx::literal{c_shape, c});
    auto c2 = p.add_instruction(migraphx::op::convolution{{1, 0}, {1, 1}, {1, 1}}, l1, l3);

    p.add_instruction(migraphx::op::concat{1}, c1, c2);
    auto count1 = std::distance(p.begin(), p.end());
    p.compile(horizontal_fusion_target{});
    auto count2 = std::distance(p.begin(), p.end());

    EXPECT(count1 == count2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
