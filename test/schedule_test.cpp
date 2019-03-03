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

struct wait_event
{
    std::vector<std::size_t> wait_for;
    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::pack(f(self.wait_for, "wait_for"));
    }
    std::string name() const { return "wait_event"; }
    migraphx::shape compute_shape(const std::vector<migraphx::shape>&) const { return {}; }

    migraphx::argument compute(migraphx::context&,
                               const migraphx::shape&,
                               const std::vector<migraphx::argument>&) const
    {
        return {};
    }
};

using instruction_map = std::unordered_map<migraphx::instruction_ref, std::size_t>;

struct schedule_model_test
{
    instruction_map* ins2stream;
    std::size_t concurrency() const { return 4; }
    void
    schedule_instruction(migraphx::program&, migraphx::instruction_ref ins, std::size_t n) const
    {
        (*ins2stream)[ins] = n;
    }
    void wait(migraphx::program& p,
              migraphx::instruction_ref ins,
              std::size_t wait_on,
              const std::vector<std::size_t>& wait_for) const
    {
        (*ins2stream)[ins] = wait_on;
        p.insert_instruction(ins, wait_event{wait_for});
    }
    std::size_t weight(const migraphx::operation& op) const
    {
        if(op.name() == "binary" or op.name() == "unary")
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
    for(auto ins : migraphx::iterator_for(p))
    {
        if(ins->name() != "identity")
            continue;
        if(ins->inputs().size() != 2)
            continue;
        if(ins->inputs() == std::vector<migraphx::instruction_ref>{x, y})
            return true;
        if(ins->inputs() == std::vector<migraphx::instruction_ref>{y, x})
            return true;
    }
    return false;
}

std::vector<std::size_t> get_wait_for(std::size_t wait_on, std::vector<std::size_t> wait_for)
{
    wait_for.erase(std::find(wait_for.begin(), wait_for.end(), wait_on));
    std::sort(wait_for.begin(), wait_for.end());
    return wait_for;
}

std::vector<std::size_t> get_wait_for(migraphx::instruction_ref ins)
{
    auto wait_ins = std::prev(ins);
    if(wait_ins->name() != "wait_event")
        return {};
    auto wf = migraphx::any_cast<wait_event>(wait_ins->get_operator()).wait_for;
    std::sort(wf.begin(), wf.end());
    return wf;
}

TEST_CASE(test1)
{
    instruction_map stream;
    migraphx::program p;
    auto one    = p.add_literal(1);
    auto two    = p.add_literal(2);
    auto onep   = p.add_instruction(unary_op{}, one);
    auto twop   = p.add_instruction(unary_op{}, two);
    auto binary = p.add_instruction(binary_op{}, onep, twop);
    p.compile(schedule_target{&stream});
    EXPECT(stream.at(onep) != stream.at(twop));
    EXPECT(stream.at(binary) == 0);
    EXPECT(get_wait_for(binary) == get_wait_for(stream[binary], {stream[onep], stream[twop]}));
    EXPECT(check_conflicts(p, onep, twop));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
