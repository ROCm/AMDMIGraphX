#include <migraphx/schedule.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <basic_ops.hpp>
#include <test.hpp>

struct unary_op
{
    std::string name() const { return "unary"; }
    migraphx::argument
    compute(migraphx::context&, const migraphx::shape&, std::vector<migraphx::argument> args) const
    {
        if(args.empty())
            return {};
        return args.front();
    }

    migraphx::shape compute_shape(std::vector<migraphx::shape> inputs) const
    {
        if(inputs.empty())
            return {};
        return inputs.front();
    }
    int output_alias(const std::vector<migraphx::shape>&) const { return 0; }
};

struct binary_op
{
    std::string name() const { return "binary"; }
    migraphx::argument
    compute(migraphx::context&, const migraphx::shape&, std::vector<migraphx::argument> args) const
    {
        if(args.empty())
            return {};
        return args.front();
    }

    migraphx::shape compute_shape(std::vector<migraphx::shape> inputs) const
    {
        if(inputs.empty())
            return {};
        return inputs.front();
    }
};

using instruction_map = std::unordered_map<migraphx::instruction_ref, std::size_t>;

struct schedule_model_test
{
    instruction_map* ins2stream;
    std::size_t concurrency() const
    {
        return 4;
    }
    void schedule_instruction(migraphx::program& p, migraphx::instruction_ref ins, std::size_t n) const
    {
        (*ins2stream)[ins] = n;
    }
    void wait(migraphx::program& p,
              migraphx::instruction_ref ins,
              std::size_t wait_on,
              const std::vector<std::size_t>& wait_for) const
    {

    }
    std::size_t weight(const migraphx::operation& op) const
    {
        if (op.name() == "binary" or op.name() == "unary")
            return 4;
        else
            return 1;
    }
};

struct schedule_target
{
    instruction_map* ins2stream;
    std::string name() const { return "schedule"; }
    std::vector<migraphx::pass> get_passes(migraphx::context&) const
    {
        return {migraphx::schedule{schedule_model_test{ins2stream}}};
    }
    migraphx::context get_context() const { return {}; }
};

bool check_conflicts(migraphx::program& p, migraphx::instruction_ref x, migraphx::instruction_ref y)
{
    for(auto ins:migraphx::iterator_for(p))
    {
        if (ins->name() != "identity")
            continue;
        if (ins->inputs().size() != 2)
            continue;
        if (ins->inputs() == std::vector<migraphx::instruction_ref>{x, y})
            return true;
        if (ins->inputs() == std::vector<migraphx::instruction_ref>{y, x})
            return true;
    }
    return false;
}

TEST_CASE(test1)
{
    instruction_map ins2stream;
    migraphx::program p;
    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    auto onep = p.add_instruction(unary_op{}, one);
    auto twop = p.add_instruction(unary_op{}, two);
    auto binary = p.add_instruction(binary_op{}, onep, twop);
    p.compile(schedule_target{&ins2stream});
    EXPECT(ins2stream.at(onep) != ins2stream.at(twop));
    EXPECT(ins2stream.at(binary) == 0);
    EXPECT(check_conflicts(p, onep, twop));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
