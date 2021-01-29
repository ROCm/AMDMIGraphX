#include <migraphx/eliminate_allocation.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/argument.hpp>
#include <basic_ops.hpp>
#include <test.hpp>

void run_pass(migraphx::module& m, std::size_t align = 32)
{
    migraphx::run_passes(
        m, {migraphx::eliminate_allocation{"allocate", align}, migraphx::dead_code_elimination{}});
}

struct allocate
{
    migraphx::shape s{};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::pack(f(self.s, "shape"));
    }

    std::string name() const { return "allocate"; }
    migraphx::shape compute_shape(const std::vector<migraphx::shape>& inputs) const
    {
        migraphx::check_shapes{inputs, *this}.has(0);
        return s;
    }
    migraphx::argument compute(migraphx::context&,
                               const migraphx::shape& output_shape,
                               const std::vector<migraphx::argument>&) const
    {
        return {output_shape};
    }
};

TEST_CASE(basic)
{
    migraphx::module m;

    auto a1 = m.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {8}}});
    auto m1 = m.add_instruction(pass_op{}, a1);

    auto a2 = m.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {40}}});
    auto m2 = m.add_instruction(pass_op{}, a2, m1);

    auto a3 = m.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {200}}});
    m.add_instruction(pass_op{}, a3, m2);

    run_pass(m);
    EXPECT(m.get_output_shapes().back() == migraphx::shape{migraphx::shape::float_type, {200}});
    EXPECT(m.get_parameter_shape("memory").bytes() == (8 * 4 + 40 * 4 + 200 * 4));
}

TEST_CASE(aligned)
{
    migraphx::module m;

    auto a1 = m.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {1}}});
    auto m1 = m.add_instruction(pass_op{}, a1);

    auto a2 = m.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {2}}});
    auto m2 = m.add_instruction(pass_op{}, a2, m1);

    auto a3 = m.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {200}}});
    m.add_instruction(pass_op{}, a3, m2);

    run_pass(m);
    EXPECT(m.get_output_shapes().back() == migraphx::shape{migraphx::shape::float_type, {200}});
    EXPECT(m.get_parameter_shape("memory").bytes() == (32 + 32 + 200 * 4));
}

TEST_CASE(unaligned)
{
    migraphx::module m;

    auto a1 = m.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {1}}});
    auto m1 = m.add_instruction(pass_op{}, a1);

    auto a2 = m.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {2}}});
    auto m2 = m.add_instruction(pass_op{}, a2, m1);

    auto a3 = m.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {200}}});
    m.add_instruction(pass_op{}, a3, m2);

    run_pass(m, 1);
    EXPECT(m.get_output_shapes().back() == migraphx::shape{migraphx::shape::float_type, {200}});
    EXPECT(m.get_parameter_shape("memory").bytes() == (1 * 4 + 2 * 4 + 200 * 4));
}

TEST_CASE(float_aligned)
{
    migraphx::module m;

    auto a1 = m.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {1}}});
    auto m1 = m.add_instruction(pass_op{}, a1);

    auto a2 = m.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {2}}});
    auto m2 = m.add_instruction(pass_op{}, a2, m1);

    auto a3 = m.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {200}}});
    m.add_instruction(pass_op{}, a3, m2);

    run_pass(m, 4);
    EXPECT(m.get_output_shapes().back() == migraphx::shape{migraphx::shape::float_type, {200}});
    EXPECT(m.get_parameter_shape("memory").bytes() == (1 * 4 + 2 * 4 + 200 * 4));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
