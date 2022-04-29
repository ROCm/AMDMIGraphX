#include <migraphx/allocation_model.hpp>
#include <migraphx/replace_allocate.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/register_op.hpp>
#include <basic_ops.hpp>
#include <test.hpp>

struct cpu_allocate : migraphx::auto_register_op<cpu_allocate>
{
    migraphx::shape s{};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::pack(f(self.s, "shape"));
    }

    std::string name() const { return "cpu::allocate"; }
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

struct hip_allocate : migraphx::auto_register_op<hip_allocate>
{
    migraphx::shape s{};
    std::string tag{};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::pack(f(self.s, "shape"), f(self.tag, "tag"));
    }

    std::string name() const { return "hip::allocate"; }
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

struct cpu_allocation_model
{
    std::string name() const { return "cpu::allocate"; }
    migraphx::operation allocate(const migraphx::shape& s) const
    {
        return migraphx::make_op(name(), {{"shape", to_value(s)}});
    }
    migraphx::operation preallocate(const migraphx::shape&, const std::string&) const { return {}; }
    std::string copy() const { return {}; }
};

struct gpu_allocation_model
{
    std::string name() const { return "hip::allocate"; }
    migraphx::operation allocate(const migraphx::shape& s) const
    {
        return migraphx::make_op(name(), {{"shape", to_value(s)}});
    }
    migraphx::operation preallocate(const migraphx::shape&, const std::string&) const { return {}; }
    std::string copy() const { return {}; }
};

void run_pass(migraphx::module& m, migraphx::allocation_model model, bool offload_copy = false)
{
    migraphx::run_passes(m,
                         {migraphx::replace_allocate{std::move(model), offload_copy},
                          migraphx::dead_code_elimination{}});
}

migraphx::module create_simple_program()
{
    migraphx::module m;
    migraphx::shape s{migraphx::shape::float_type, {5}};
    auto x = m.add_parameter("x", s);
    auto y = m.add_parameter("y", s);
    auto alloc =
        m.add_instruction(migraphx::make_op("allocate", {{"shape", migraphx::to_value(s)}}));
    m.add_instruction(pass_op{}, x, y, alloc);
    return m;
}

TEST_CASE(cpu_allocate)
{
    migraphx::module m = create_simple_program();
    run_pass(m, cpu_allocation_model{});

    EXPECT(std::any_of(m.begin(), m.end(), [](const migraphx::instruction& ins) {
        return migraphx::contains(ins.name(), "cpu::allocate");
    }));
}

TEST_CASE(hip_out_param)
{
    migraphx::module m = create_simple_program();
    run_pass(m, gpu_allocation_model{});

    EXPECT(std::none_of(m.begin(), m.end(), [](const migraphx::instruction& ins) {
        return migraphx::contains(ins.name(), "allocate");
    }));
}

TEST_CASE(hip_out_param_return)
{
    migraphx::module m = create_simple_program();
    m.add_return({std::prev(m.end())});
    run_pass(m, gpu_allocation_model{});

    EXPECT(std::none_of(m.begin(), m.end(), [](const migraphx::instruction& ins) {
        return migraphx::contains(ins.name(), "allocate");
    }));
}

TEST_CASE(hip_allocate)
{
    migraphx::module m;
    migraphx::shape s{migraphx::shape::float_type, {5}};
    auto x = m.add_parameter("x", s);
    auto y = m.add_parameter("y", s);
    auto z = m.add_parameter("z", s);
    auto alloc =
        m.add_instruction(migraphx::make_op("allocate", {{"shape", migraphx::to_value(s)}}));
    auto pass1 = m.add_instruction(pass_op{}, x, y, alloc);
    auto alloc2 =
        m.add_instruction(migraphx::make_op("allocate", {{"shape", migraphx::to_value(s)}}));
    m.add_instruction(pass_op{}, z, pass1, alloc2);
    run_pass(m, gpu_allocation_model{});

    EXPECT(std::any_of(m.begin(), m.end(), [](const migraphx::instruction& ins) {
        return migraphx::contains(ins.name(), "hip::allocate");
    }));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
