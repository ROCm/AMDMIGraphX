#include <migraphx/eliminate_allocation.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/argument.hpp>
#include <basic_ops.hpp>
#include <test.hpp>

void run_pass(migraphx::program& p, std::size_t align = 32)
{
    migraphx::run_passes(
        p, {migraphx::eliminate_allocation{"allocate", align}, migraphx::dead_code_elimination{}});
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
        migraphx::check_shapes{inputs}.has(0);
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
    migraphx::program p;
    auto a1 = p.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {8}}});
    auto p1 = p.add_instruction(pass_op{}, a1);

    auto a2 = p.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {40}}});
    auto p2 = p.add_instruction(pass_op{}, a2, p1);

    auto a3 = p.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {200}}});
    p.add_instruction(pass_op{}, a3, p2);

    run_pass(p);
    EXPECT(p.get_output_shapes().back() == migraphx::shape{migraphx::shape::float_type, {200}});
    EXPECT(p.get_parameter_shape("memory").bytes() == (8 * 4 + 40 * 4 + 200 * 4));
}

TEST_CASE(aligned)
{
    migraphx::program p;
    auto a1 = p.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {1}}});
    auto p1 = p.add_instruction(pass_op{}, a1);

    auto a2 = p.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {2}}});
    auto p2 = p.add_instruction(pass_op{}, a2, p1);

    auto a3 = p.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {200}}});
    p.add_instruction(pass_op{}, a3, p2);

    run_pass(p);
    EXPECT(p.get_output_shapes().back() == migraphx::shape{migraphx::shape::float_type, {200}});
    EXPECT(p.get_parameter_shape("memory").bytes() == (32 + 32 + 200 * 4));
}

TEST_CASE(unaligned)
{
    migraphx::program p;
    auto a1 = p.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {1}}});
    auto p1 = p.add_instruction(pass_op{}, a1);

    auto a2 = p.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {2}}});
    auto p2 = p.add_instruction(pass_op{}, a2, p1);

    auto a3 = p.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {200}}});
    p.add_instruction(pass_op{}, a3, p2);

    run_pass(p, 1);
    EXPECT(p.get_output_shapes().back() == migraphx::shape{migraphx::shape::float_type, {200}});
    EXPECT(p.get_parameter_shape("memory").bytes() == (1 * 4 + 2 * 4 + 200 * 4));
}

TEST_CASE(float_aligned)
{
    migraphx::program p;
    auto a1 = p.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {1}}});
    auto p1 = p.add_instruction(pass_op{}, a1);

    auto a2 = p.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {2}}});
    auto p2 = p.add_instruction(pass_op{}, a2, p1);

    auto a3 = p.add_instruction(allocate{migraphx::shape{migraphx::shape::float_type, {200}}});
    p.add_instruction(pass_op{}, a3, p2);

    run_pass(p, 4);
    EXPECT(p.get_output_shapes().back() == migraphx::shape{migraphx::shape::float_type, {200}});
    EXPECT(p.get_parameter_shape("memory").bytes() == (1 * 4 + 2 * 4 + 200 * 4));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
