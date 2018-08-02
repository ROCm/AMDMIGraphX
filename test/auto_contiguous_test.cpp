#include <migraph/auto_contiguous.hpp>
#include <migraph/operators.hpp>
#include <basic_ops.hpp>
#include <test.hpp>

struct contigous_target
{
    std::string name() const { return "contigous"; }
    std::vector<migraph::pass> get_passes(migraph::context&) const
    {
        return {migraph::auto_contigous{}};
    }
    migraph::context get_context() const { return {}; }
};

migraph::literal get_2x2()
{
    return migraph::literal{{migraph::shape::float_type, {2, 2}}, {1, 2, 3, 4}};
}

void after_literal_transpose()
{
    migraph::program p;
    auto l = p.add_literal(get_2x2());
    EXPECT(p.get_shape().packed());
    p.add_instruction(migraph::transpose{{1, 0}}, l);
    EXPECT(not p.get_shape().packed());
    p.compile(contigous_target{});
    EXPECT(p.get_shape().packed());
}

int main() { after_literal_transpose(); }
