#include <migraph/auto_contiguous.hpp>
#include <migraph/operators.hpp>
#include <basic_ops.hpp>
#include <test.hpp>

struct contiguous_target
{
    std::string name() const { return "contiguous"; }
    std::vector<migraph::pass> get_passes(migraph::context&) const
    {
        return {migraph::auto_contiguous{}};
    }
    migraph::context get_context() const { return {}; }
};

migraph::literal get_2x2()
{
    return migraph::literal{{migraph::shape::float_type, {2, 2}}, {1, 2, 3, 4}};
}

migraph::literal get_2() { return migraph::literal{{migraph::shape::float_type, {2}}, {1, 2}}; }

void after_literal_transpose()
{
    migraph::program p;
    auto l = p.add_literal(get_2x2());
    EXPECT(p.get_shape().standard());
    EXPECT(not p.get_shape().transposed());
    auto t = p.add_instruction(migraph::transpose{{1, 0}}, l);
    p.add_instruction(pass_op{}, t);
    EXPECT(not p.get_shape().standard());
    EXPECT(p.get_shape().transposed());
    p.compile(contiguous_target{});
    EXPECT(p.get_shape().standard());
    EXPECT(not p.get_shape().transposed());
}

void after_literal_broadcast()
{
    migraph::program p;
    auto l1 = p.add_literal(get_2x2());
    auto l2 = p.add_literal(get_2());
    EXPECT(p.get_shape().standard());
    EXPECT(not p.get_shape().broadcasted());
    auto b = p.add_instruction(migraph::broadcast{}, l1, l2);
    p.add_instruction(pass_op{}, b);
    EXPECT(not p.get_shape().standard());
    EXPECT(p.get_shape().broadcasted());
    p.compile(contiguous_target{});
    EXPECT(p.get_shape().standard());
    EXPECT(not p.get_shape().broadcasted());
}

int main()
{
    after_literal_transpose();
    after_literal_broadcast();
}
